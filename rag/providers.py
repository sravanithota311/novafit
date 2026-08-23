"""Model providers: local Ollama or cloud Gemini for chat; local embeddings."""
from __future__ import annotations

from rag.config import (
    EMBED_MODEL,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    KEEP_ALIVE,
    LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    TEMPERATURE,
)


def get_chat_model():
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=TEMPERATURE,
            google_api_key=GOOGLE_API_KEY,
        )
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
        keep_alive=KEEP_ALIVE,
    )


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)
