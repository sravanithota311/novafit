"""Build the searchable index: load -> split -> embed -> store.

Uses whichever embedding model the configured provider supplies (local
HuggingFace or cloud Gemini), so the index always matches the query side.

Run manually with:  python ingest.py
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, INDEX_DIR
from rag.providers import get_embeddings

SUPPORTED = {".txt": TextLoader, ".md": TextLoader, ".pdf": PyPDFLoader}


def load_documents(data_dir: Path) -> list:
    docs = []
    files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        raise FileNotFoundError(
            f"No documents found in {data_dir}. Add .md, .txt, or .pdf files."
        )
    for path in files:
        loader_cls = SUPPORTED[path.suffix.lower()]
        loader = (
            loader_cls(str(path), encoding="utf-8")
            if loader_cls is TextLoader
            else loader_cls(str(path))
        )
        loaded = loader.load()
        docs.extend(loaded)
        print(f"  loaded {path.name}  ({len(loaded)} section(s))")
    return docs


# Kept for backward compatibility; other modules may import this name.
def build_embeddings():
    return get_embeddings()


def build_index() -> Path:
    print(f"Loading documents from {DATA_DIR} ...")
    docs = load_documents(DATA_DIR)

    print(f"Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"  {len(chunks)} chunks created")

    print("Embedding and building FAISS index ...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"Done. Index saved to {INDEX_DIR}")
    return INDEX_DIR


def ensure_index() -> None:
    """Build the index if it doesn't exist yet (used on cloud first-run)."""
    if not Path(INDEX_DIR).exists():
        build_index()


if __name__ == "__main__":
    build_index()