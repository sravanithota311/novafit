"""Model providers: run locally with Ollama, or in the cloud with Gemini.

Which one is used is controlled by LLM_PROVIDER in config (default "ollama").
Imports are done lazily inside the functions so that a deployment only needs
the packages for the provider it actually uses — the cloud build doesn't need
torch/sentence-transformers, and a local build doesn't need google-genai.
"""
from __future__ import annotations

from rag.config import (
    EMBED_MODEL,
    GEMINI_EMBED_MODEL,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    KEEP_ALIVE,
    LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    TEMPERATURE,
)


def get_chat_model():
    """Return the chat LLM for the configured provider."""
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=TEMPERATURE,
            google_api_key=GOOGLE_API_KEY,
        )
    # default: local Ollama
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
        keep_alive=KEEP_ALIVE,
    )


def get_embeddings():
    """Return the embedding model for the configured provider.

    IMPORTANT: the index must be built and queried with the same embeddings,
    so this is used both in ingest and in the agent.
    """
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBED_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    # default: local HuggingFace
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)