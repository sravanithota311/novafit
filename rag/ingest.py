"""Build the searchable index: load -> split -> embed -> store.

Reads every supported file in DATA_DIR, splits it into chunks, embeds them with
a local HuggingFace model, and saves a FAISS index to INDEX_DIR.

Run it whenever your documents change:  python ingest.py
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBED_MODEL,
    INDEX_DIR,
)

SUPPORTED = {".txt": TextLoader, ".md": TextLoader, ".pdf": PyPDFLoader}


def load_documents(data_dir: Path) -> list:
    """Load every supported file in data_dir into LangChain Documents."""
    docs = []
    files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        raise FileNotFoundError(
            f"No documents found in {data_dir}. "
            f"Add .md, .txt, or .pdf files and re-run."
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


def build_embeddings() -> HuggingFaceEmbeddings:
    """The local embedding model. Reused in ingest AND at query time.

    Using the SAME embedding model in both phases is essential — otherwise the
    question vectors and document vectors live in different spaces and
    retrieval returns nonsense.
    """
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_index() -> Path:
    """Full indexing pipeline. Returns the path to the saved index."""
    print(f"Loading documents from {DATA_DIR} ...")
    docs = load_documents(DATA_DIR)

    print(f"Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"  {len(chunks)} chunks created")

    print(f"Embedding with {EMBED_MODEL} (first run downloads the model) ...")
    embeddings = build_embeddings()

    print("Building FAISS index ...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"Done. Index saved to {INDEX_DIR}")
    return INDEX_DIR


if __name__ == "__main__":
    build_index()
