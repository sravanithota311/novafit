"""Central configuration.

Everything tunable lives here so the app, the API, and the ingest script all
read from one place. Any value can be overridden with an environment variable
(see .env.example).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", BASE_DIR / "faiss_index"))

# --- Models (all local, no API keys) ----------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chat model served by Ollama. Agent mode needs a tool-calling model.
# Recommended:  ollama pull qwen2.5:3b
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# How many web results the web-search tool pulls per query.
MAX_WEB_RESULTS = int(os.getenv("MAX_WEB_RESULTS", "5"))

# How long Ollama keeps the model warm in RAM after a request.
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "10m")

# --- Retrieval knobs ---------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

# --- Branding ----------------------------------------------------------------
# The assistant's display name. Change this to rename it everywhere.
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "NovaFit")

# A short description of what's in the private knowledge base (the files in
# data/). The agent uses this to decide when to search documents vs. the web.
# Update it whenever you swap the documents. It is not shown prominently in UI.
KNOWLEDGE_BASE_TOPIC = os.getenv(
    "KNOWLEDGE_BASE_TOPIC",
    "health and fitness fundamentals: exercise, nutrition, sleep, and recovery",
)
