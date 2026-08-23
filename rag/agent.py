"""The assistant core — tool-free routing (works on any model, incl. Gemini).

Instead of LLM function-calling (which newer Gemini models reject without a
"thought_signature"), we route manually:

  1. Retrieve from the knowledge base and try to answer from it.
  2. If the answer isn't in the documents, the model replies SEARCH_WEB,
     and we do a web search and answer from that instead.

This keeps the same behavior (documents first, web fallback, cited sources,
conversation memory) with plain LLM calls that work everywhere.
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag.config import (
    ASSISTANT_NAME,
    INDEX_DIR,
    KNOWLEDGE_BASE_TOPIC,
    MAX_WEB_RESULTS,
    TOP_K,
)
from rag.ingest import ensure_index
from rag.providers import get_chat_model, get_embeddings
from rag.uploads import load_uploads_retriever

try:
    from ddgs import DDGS
except ImportError:  # older installs
    from duckduckgo_search import DDGS  # type: ignore


WEB_SENTINEL = "SEARCH_WEB"


class Agent:
    """Documents-first assistant with a web fallback — no function-calling."""

    def __init__(self, embeddings=None) -> None:
        ensure_index()
        embeddings = embeddings or get_embeddings()
        base_vs = FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
        self._base_retriever = base_vs.as_retriever(search_kwargs={"k": TOP_K})
        self._uploads_retriever = load_uploads_retriever(embeddings, k=TOP_K)
        self._llm = get_chat_model()

    # --- helpers -------------------------------------------------------------
    @staticmethod
    def _history_msgs(history: list[dict]) -> list:
        msgs = []
        for turn in history:
            if turn["role"] == "user":
                msgs.append(HumanMessage(content=turn["content"]))
            else:
                msgs.append(AIMessage(content=turn["content"]))
        return msgs

    def _kb_search(self, query: str):
        docs = list(self._base_retriever.invoke(query))
        if self._uploads_retriever is not None:
            docs += list(self._uploads_retriever.invoke(query))
        sources = [{
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page"),
            "snippet": d.page_content[:300].strip(),
        } for d in docs]
        context = "\n\n".join(d.page_content for d in docs)
        return context, sources

    def _web_search(self, query: str):
        web_sources, lines = [], []
        results = []
        try:
            with DDGS() as ddgs:
                # Try a couple of backends; datacenter IPs sometimes get thin
                # results from the default one.
                results = list(ddgs.text(query, max_results=MAX_WEB_RESULTS))
                if not results:
                    results = list(ddgs.text(query, max_results=MAX_WEB_RESULTS,
                                             backend="html"))
        except Exception as exc:
            return f"Web search failed: {exc}", []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "") or r.get("snippet", "")
            href = r.get("href", "") or r.get("url", "")
            web_sources.append({"title": title, "url": href})
            lines.append(f"{title}\n{body}\n({href})")
        return "\n\n".join(lines), web_sources

    def _answer(self, system_text: str, history: list[dict], question: str) -> str:
        messages = [SystemMessage(content=system_text)]
        messages += self._history_msgs(history)
        messages.append(HumanMessage(content=question))
        resp = self._llm.invoke(messages)
        return (resp.content or "").strip()

    # --- main entry ----------------------------------------------------------
    def run_stream(self, question: str, history: list[dict] | None = None,
                   user_name: str = "there"):
        history = history or []

        # Step 1: try the knowledge base.
        yield {"type": "tool", "name": "search_knowledge_base"}
        kb_context, doc_sources = self._kb_search(question)

        kb_system = (
            f"You are {ASSISTANT_NAME}, a helpful assistant chatting with a user "
            f"named {user_name}. The knowledge base covers {KNOWLEDGE_BASE_TOPIC}.\n"
            "Answer the user's question using ONLY the context below. "
            f"If the context does not contain the answer, reply with exactly "
            f"'{WEB_SENTINEL}' and nothing else.\n\n"
            f"Context:\n{kb_context if kb_context else '(no documents found)'}"
        )
        answer = self._answer(kb_system, history, question)

        # Step 2: web fallback if the KB couldn't answer.
        if WEB_SENTINEL in answer.upper() and len(answer.strip()) <= 40:
            yield {"type": "tool", "name": "search_the_web"}
            web_context, web_sources = self._web_search(question)
            web_system = (
                f"You are {ASSISTANT_NAME}, a helpful assistant chatting with a "
                f"user named {user_name}. Answer the question using the web "
                "search results below (titles, snippets, and links). Use the "
                "information available even if partial, summarize what the "
                "results say, and mention the most relevant source. Only say you "
                "don't know if the results are truly empty or irrelevant.\n\n"
                f"Web results:\n{web_context}"
            )
            answer = self._answer(web_system, history, question)
            yield {
                "type": "final",
                "answer": answer,
                "doc_sources": [],
                "web_sources": web_sources,
            }
        else:
            yield {
                "type": "final",
                "answer": answer,
                "doc_sources": doc_sources,
                "web_sources": [],
            }

    def ask(self, question: str, history: list[dict] | None = None,
            user_name: str = "there") -> dict:
        result = {"answer": "", "doc_sources": [], "web_sources": []}
        for ev in self.run_stream(question, history, user_name=user_name):
            if ev["type"] == "final":
                result = {
                    "answer": ev["answer"],
                    "doc_sources": ev["doc_sources"],
                    "web_sources": ev["web_sources"],
                }
        return result
