import os
import re
from io import BytesIO
from typing import Dict, List, Tuple

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


# -----------------------------
# Configuration
# -----------------------------
APP_TITLE = "Open-Source RAG PDF Assistant"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE_TOKENS = 450
CHUNK_OVERLAP_TOKENS = 75
TOP_K = 5
MAX_CONTEXT_CHARS = 18000


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
)

st.title("📚 Open-Source RAG PDF Assistant")
st.caption(
    "Upload a PDF → extract text → tokenize → chunk → embed → FAISS search → "
    "answer with an open-weight model on Groq."
)


# -----------------------------
# Cached open-source models
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Loading tokenizer...")
def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL)


# -----------------------------
# PDF processing
# -----------------------------
def extract_pdf_text(uploaded_file) -> Tuple[str, List[Dict]]:
    """Extract text page-by-page and retain page numbers."""
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    full_text = "\n".join(item["text"] for item in pages)
    return full_text, pages


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_text(text: str, tokenizer) -> List[int]:
    """Tokenize without adding special model tokens."""
    return tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
    )


def detokenize(token_ids: List[int], tokenizer) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()


def build_token_chunks(
    pages: List[Dict],
    tokenizer,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> List[Dict]:
    """
    Create token-based chunks while preserving approximate page metadata.
    Chunks are created independently per page to make citations easier.
    """
    chunks = []

    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    chunk_id = 0

    for page_item in pages:
        page_number = page_item["page"]
        text = normalize_text(page_item["text"])

        if not text:
            continue

        token_ids = tokenize_text(text, tokenizer)

        start = 0
        while start < len(token_ids):
            end = min(start + chunk_size, len(token_ids))
            current_ids = token_ids[start:end]
            chunk_text = detokenize(current_ids, tokenizer)

            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_number,
                        "text": chunk_text,
                        "token_count": len(current_ids),
                    }
                )
                chunk_id += 1

            if end >= len(token_ids):
                break

            start = end - overlap

    return chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def create_faiss_index(
    chunks: List[Dict],
    embedding_model: SentenceTransformer,
):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    ).astype("float32")

    dimension = embeddings.shape[1]

    # Inner product over normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings


def retrieve_chunks(
    query: str,
    index,
    chunks: List[Dict],
    embedding_model: SentenceTransformer,
    top_k: int = TOP_K,
) -> List[Dict]:
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        result = dict(chunks[int(idx)])
        result["score"] = float(score)
        results.append(result)

    return results


# -----------------------------
# Groq generation
# -----------------------------
def get_groq_client() -> Groq:
    api_key = None

    # Streamlit Cloud / local .streamlit/secrets.toml
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

    # Local environment variable fallback
    api_key = api_key or os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Add it to Streamlit Secrets "
            "or set it as an environment variable."
        )

    return Groq(api_key=api_key)


def build_context(retrieved_chunks: List[Dict]) -> str:
    context_parts = []
    total_chars = 0

    for item in retrieved_chunks:
        block = (
            f"[Source: Page {item['page']}, "
            f"Chunk {item['chunk_id']}, "
            f"Similarity {item['score']:.3f}]\n"
            f"{item['text']}"
        )

        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break

        context_parts.append(block)
        total_chars += len(block)

    return "\n\n---\n\n".join(context_parts)


def answer_question(
    question: str,
    retrieved_chunks: List[Dict],
) -> str:
    client = get_groq_client()
    context = build_context(retrieved_chunks)

    system_prompt = """You are a careful document question-answering assistant.

Answer the user's question using ONLY the supplied document context.
Do not invent facts that are not supported by the context.

Rules:
1. If the answer is not present in the context, say:
   "I couldn't find that information in the uploaded document."
2. Be concise but useful.
3. Cite supporting page numbers in the form [Page X].
4. If multiple pages support the answer, cite each relevant page.
5. Do not mention your internal retrieval process unless asked.
"""

    user_prompt = f"""DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1000,
    )

    return completion.choices[0].message.content.strip()


# -----------------------------
# Session state
# -----------------------------
if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ RAG Settings")

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=TOP_K,
        help="Number of relevant chunks sent to the LLM.",
    )

    st.write(f"**Embedding model:** `{EMBEDDING_MODEL}`")
    st.write(f"**Generation model:** `{GROQ_MODEL}`")
    st.write(f"**Chunk size:** `{CHUNK_SIZE_TOKENS}` tokens")
    st.write(f"**Chunk overlap:** `{CHUNK_OVERLAP_TOKENS}` tokens")
    st.write("**Vector database:** FAISS (local/in-memory)")

    if st.session_state.index is not None:
        st.success(
            f"Indexed {len(st.session_state.chunks)} chunks "
            f"from `{st.session_state.document_name}`"
        )

    if st.button("🗑️ Clear document", use_container_width=True):
        st.session_state.index = None
        st.session_state.chunks = []
        st.session_state.document_name = None
        st.session_state.messages = []
        st.rerun()


# -----------------------------
# Upload + indexing
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="For best results, use a text-based PDF. Scanned image-only PDFs require OCR.",
)

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.document_name:
        with st.status("Processing PDF...", expanded=True) as status:
            st.write("1️⃣ Extracting PDF text...")
            full_text, pages = extract_pdf_text(uploaded_file)

            if not full_text.strip():
                status.update(
                    label="No extractable text found",
                    state="error",
                )
                st.error(
                    "This PDF appears to be scanned/image-only or contains no "
                    "extractable text. Add OCR support before using this document."
                )
                st.stop()

            st.write(f"Extracted {len(full_text):,} characters from {len(pages)} pages.")

            st.write("2️⃣ Tokenizing and creating chunks...")
            tokenizer = load_tokenizer()
            chunks = build_token_chunks(pages, tokenizer)

            if not chunks:
                status.update(
                    label="No chunks created",
                    state="error",
                )
                st.error("No usable text chunks were created from the PDF.")
                st.stop()

            st.write(f"Created {len(chunks)} token-based chunks.")

            st.write("3️⃣ Creating embeddings...")
            embedding_model = load_embedding_model()
            index, _ = create_faiss_index(chunks, embedding_model)

            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.document_name = uploaded_file.name
            st.session_state.messages = []

            status.update(
                label="PDF indexed successfully",
                state="complete",
            )

        st.success(
            f"✅ `{uploaded_file.name}` is ready. "
            f"Ask questions about the document below."
        )


# -----------------------------
# Chat interface
# -----------------------------
if st.session_state.index is None:
    st.info("👆 Upload a PDF to build the RAG index.")

    with st.expander("How this application works"):
        st.markdown(
            """
            **1. PDF extraction** → `pypdf` extracts text page by page.

            **2. Tokenization** → a Hugging Face tokenizer converts text into tokens.

            **3. Chunking** → text is split into overlapping token-based chunks.

            **4. Embeddings** → `all-MiniLM-L6-v2` converts each chunk into a vector.

            **5. Vector search** → FAISS finds the most similar chunks to the question.

            **6. Generation** → retrieved context is sent to
            `openai/gpt-oss-120b` through Groq.

            **7. Answer** → the model answers using the retrieved document context
            and includes page citations.
            """
        )

    st.stop()


# Render previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask a question about your PDF...")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document and generating an answer..."):
            try:
                embedding_model = load_embedding_model()

                retrieved = retrieve_chunks(
                    question,
                    st.session_state.index,
                    st.session_state.chunks,
                    embedding_model,
                    top_k=top_k,
                )

                answer = answer_question(question, retrieved)

                st.markdown(answer)

                with st.expander("🔎 Retrieved sources"):
                    for item in retrieved:
                        st.markdown(
                            f"**Page {item['page']} · "
                            f"Chunk {item['chunk_id']} · "
                            f"Similarity {item['score']:.3f}**"
                        )
                        st.write(item["text"])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as exc:
                error_message = (
                    f"Something went wrong: `{type(exc).__name__}: {exc}`"
                )
                st.error(error_message)
                st.info(
                    "If this is a Groq error, verify that GROQ_API_KEY is present "
                    "and that the selected model is available to your API key."
                )
