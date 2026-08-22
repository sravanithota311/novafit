"""Central configuration.

Reads from environment variables and (when running under Streamlit) from
Streamlit secrets, so the same code runs locally with Ollama and in the cloud
with Gemini just by changing settings — no code edits.

For Gemini, the exact model names available depend on the API key / version,
so we auto-discover working chat and embedding models at import time (with
sensible fallbacks) to avoid "model not found" errors.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default=None):
    """Read a setting from env vars first, then Streamlit secrets if available."""
    val = os.getenv(key)
    if val is not None and val != "":
        return val
    try:
        import streamlit as st  # only present when running the app
        secret_val = st.secrets.get(key)
        if secret_val is not None and secret_val != "":
            return secret_val
    except Exception:
        pass
    return default


# --- Paths -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(_get("DATA_DIR", BASE_DIR / "data"))
INDEX_DIR = Path(_get("INDEX_DIR", BASE_DIR / "faiss_index"))

# --- Provider: "ollama" (local) or "gemini" (cloud) --------------------------
LLM_PROVIDER = _get("LLM_PROVIDER", "ollama")

# --- Local (Ollama) settings -------------------------------------------------
LLM_MODEL = _get("LLM_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = _get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = _get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Cloud (Gemini) settings -------------------------------------------------
GOOGLE_API_KEY = _get("GOOGLE_API_KEY", "")
# These are used as-is IF explicitly set; otherwise we auto-discover below.
GEMINI_MODEL = _get("GEMINI_MODEL", "")
GEMINI_EMBED_MODEL = _get("GEMINI_EMBED_MODEL", "")


def _resolve_gemini_models(chat_override: str, embed_override: str):
    """Pick a working chat + embedding model for the current API key.

    If the user explicitly set names, respect them. Otherwise ask the API which
    models are available and choose good ones, with hard fallbacks.
    """
    chat_fallback = chat_override or "gemini-2.0-flash"
    embed_fallback = embed_override or "models/text-embedding-004"

    # If both are explicitly provided, trust them.
    if chat_override and embed_override:
        return chat_override, embed_override

    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        chat_candidates, embed_candidates = [], []
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                chat_candidates.append(m.name)
            if "embedContent" in methods:
                embed_candidates.append(m.name)

        def pick(cands, prefer, fallback):
            for p in prefer:
                for c in cands:
                    if p in c:
                        return c
            return cands[0] if cands else fallback

        chat = chat_override or pick(
            chat_candidates,
            ["gemini-2.0-flash", "flash-latest", "2.5-flash", "1.5-flash", "flash"],
            chat_fallback,
        )
        embed = embed_override or pick(
            embed_candidates,
            ["text-embedding-004", "gemini-embedding-001", "embedding-001", "embedding"],
            embed_fallback,
        )
        return chat, embed
    except Exception:
        return chat_fallback, embed_fallback


# Only bother resolving when we're actually using Gemini and have a key.
if LLM_PROVIDER == "gemini" and GOOGLE_API_KEY:
    GEMINI_MODEL, GEMINI_EMBED_MODEL = _resolve_gemini_models(
        GEMINI_MODEL, GEMINI_EMBED_MODEL
    )
else:
    GEMINI_MODEL = GEMINI_MODEL or "gemini-2.0-flash"
    GEMINI_EMBED_MODEL = GEMINI_EMBED_MODEL or "models/text-embedding-004"

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
