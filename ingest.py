"""Run `python ingest.py` to (re)build the index from everything in ./data.

Re-run this whenever you add, remove, or change documents.
"""
from rag.ingest import build_index

if __name__ == "__main__":
    build_index()
