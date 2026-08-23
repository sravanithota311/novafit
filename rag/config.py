"""Central configuration.

Reads from environment variables and (under Streamlit) from secrets, so the
same code runs locally with Ollama and in the cloud with Gemini.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default=None):
    val = os.getenv(key)
    if val is not None and val != "":
        return val
    try:
        import streamlit as st
        sv = st.secrets.get(key)
        if sv is not None and sv != "":
            return sv
    except Exception:
        pass
    return default


# --- Paths -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(_get("DATA_DIR", BASE_DIR / "data"))
INDEX_DIR = Path(_get("INDEX_DIR", BASE_DIR / "faiss_index"))

# --- Provider: "ollama" (local) or "gemini" (cloud) --------------------------
LLM_PROVIDER = _get("LLM_PROVIDER", "ollama")

# --- Local (Ollama) ----------------------------------------------------------
LLM_MODEL = _get("LLM_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = _get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = _get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Cloud (Gemini) ----------------------------------------------------------
GOOGLE_API_KEY = _get("GOOGLE_API_KEY", "")
# Set GEMINI_MODEL in Secrets. Defaults to the "latest" alias so it never goes stale.
GEMINI_MODEL = _get("GEMINI_MODEL", "gemini-flash-latest")

# --- Shared behavior ---------------------------------------------------------
KEEP_ALIVE = _get("KEEP_ALIVE", "10m")
MAX_WEB_RESULTS = int(_get("MAX_WEB_RESULTS", "5"))
CHUNK_SIZE = int(_get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "150"))
TOP_K = int(_get("TOP_K", "4"))
TEMPERATURE = float(_get("TEMPERATURE", "0"))

# --- Branding ----------------------------------------------------------------
ASSISTANT_NAME = _get("ASSISTANT_NAME", "NovaFit")
KNOWLEDGE_BASE_TOPIC = _get(
    "KNOWLEDGE_BASE_TOPIC",
    "health and fitness fundamentals: exercise, nutrition, sleep, and recovery",
)
