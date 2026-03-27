"""
ingest.py — Document Ingestion Pipeline
----------------------------------------
Loads .txt and .pdf files from /data, splits them into chunks,
generates OpenAI embeddings, and persists the FAISS vector store to /vectorstore.
Run this script directly to build or rebuild the knowledge base:
    python ingest.py
"""

import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ── Configuration ─────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR       = Path("data")
VECTORSTORE_DIR = Path("vectorstore")

# Chunk size / overlap tuned for customer-support docs.
# Smaller chunks = more precise retrieval; larger overlap = better context continuity.
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_documents(data_dir: Path) -> list:
    """
    Recursively load all .txt and .pdf files from *data_dir*.
    Returns a flat list of LangChain Document objects.
    Raises FileNotFoundError when the directory is missing or empty.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory '{data_dir}' does not exist. "
            "Create it and add your .txt / .pdf knowledge-base files."
        )

    documents = []

    # ── .txt files ──
    txt_loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        silent_errors=True,   # skip unreadable files instead of crashing
    )
    txt_docs = txt_loader.load()
    logger.info("Loaded %d .txt document(s).", len(txt_docs))
    documents.extend(txt_docs)

    # ── .pdf files ──
    pdf_files = list(data_dir.rglob("*.pdf"))
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pdf_docs = loader.load()
            logger.info("Loaded PDF '%s' (%d page(s)).", pdf_path.name, len(pdf_docs))
            documents.extend(pdf_docs)
        except Exception as exc:
            logger.warning("Could not load '%s': %s", pdf_path, exc)

    if not documents:
        raise ValueError(
            f"No .txt or .pdf files found in '{data_dir}'. "
            "Add at least one knowledge-base document and run ingest.py again."
        )

    logger.info("Total documents loaded: %d", len(documents))
    return documents


# ── Splitting ─────────────────────────────────────────────────────────────────

def split_documents(documents: list) -> list:
    """
    Split documents into overlapping chunks suitable for embedding.
    RecursiveCharacterTextSplitter tries paragraph → sentence → word
    boundaries before making a hard cut.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunk(s) (size=%d, overlap=%d).",
                len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


# ── Embeddings & Vector Store ─────────────────────────────────────────────────

def build_vectorstore(chunks: list, persist_dir: Path) -> FAISS:
    """
    Embed *chunks* with OpenAI text-embedding-3-small and persist the
    FAISS index to *persist_dir* so it can be loaded without re-embedding.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Add it to your .env file."
        )

    logger.info("Generating embeddings — this may take a moment …")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    persist_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(persist_dir))
    logger.info("Vector store saved to '%s'.", persist_dir)

    return vectorstore


# ── Public entry-point ────────────────────────────────────────────────────────

def ingest(data_dir: Path = DATA_DIR, vectorstore_dir: Path = VECTORSTORE_DIR) -> FAISS:
    """
    Full ingestion pipeline: load → split → embed → persist.
    Returns the FAISS vector store for optional immediate use.
    """
    logger.info("═══ Starting document ingestion ═══")

    documents = load_documents(data_dir)
    chunks    = split_documents(documents)
    vs        = build_vectorstore(chunks, vectorstore_dir)

    logger.info("═══ Ingestion complete (%d chunk(s) indexed) ═══", len(chunks))
    return vs


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        ingest()
    except (FileNotFoundError, ValueError, EnvironmentError) as err:
        logger.error(err)
        sys.exit(1)
