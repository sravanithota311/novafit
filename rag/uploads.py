"""Handle user-uploaded documents: index them so the agent can search them.

Uploaded files are chunked, embedded, and stored in their own FAISS index
(separate from the built-in knowledge base) under storage/uploads_index/.
The agent searches both the base knowledge base and this uploads index.
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import BASE_DIR, CHUNK_OVERLAP, CHUNK_SIZE

UPLOADS_DIR = BASE_DIR / "storage" / "uploads"
UPLOADS_INDEX = BASE_DIR / "storage" / "uploads_index"

SUPPORTED = {".txt": TextLoader, ".md": TextLoader, ".pdf": PyPDFLoader}


def save_upload(filename: str, data: bytes) -> Path:
    """Write raw uploaded bytes to disk, return the saved path."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOADS_DIR / filename
    path.write_bytes(data)
    return path


def add_to_index(path: Path, embeddings) -> int:
    """Chunk + embed a saved file and add it to the uploads index.

    Returns the number of chunks added, or 0 if the type is unsupported.
    """
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        return 0
    loader_cls = SUPPORTED[ext]
    loader = (
        loader_cls(str(path), encoding="utf-8")
        if loader_cls is TextLoader
        else loader_cls(str(path))
    )
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(docs)
    if not chunks:
        return 0

    if UPLOADS_INDEX.exists():
        vs = FAISS.load_local(
            str(UPLOADS_INDEX), embeddings, allow_dangerous_deserialization=True
        )
        vs.add_documents(chunks)
    else:
        vs = FAISS.from_documents(chunks, embeddings)
    UPLOADS_INDEX.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(UPLOADS_INDEX))
    return len(chunks)


def load_uploads_retriever(embeddings, k: int = 4):
    """Return a retriever over uploaded docs, or None if nothing uploaded."""
    if not UPLOADS_INDEX.exists():
        return None
    vs = FAISS.load_local(
        str(UPLOADS_INDEX), embeddings, allow_dangerous_deserialization=True
    )
    return vs.as_retriever(search_kwargs={"k": k})


def list_uploaded_files() -> list[str]:
    if not UPLOADS_DIR.exists():
        return []
    return sorted(p.name for p in UPLOADS_DIR.glob("*") if p.suffix.lower() in SUPPORTED)


def clear_uploads() -> None:
    """Remove all uploaded files and their index."""
    import shutil
    for d in (UPLOADS_DIR, UPLOADS_INDEX):
        if d.exists():
            shutil.rmtree(d)