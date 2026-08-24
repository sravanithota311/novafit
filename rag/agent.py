"""The assistant core — tool-free routing (works on any model, incl. Gemini).

Routing priority for each question:
  1. If the user has uploaded documents, search those FIRST and answer from
     them when they're relevant (uploads take priority).
  2. Otherwise (or if uploads don't cover it), search the built-in knowledge
     base.
  3. If neither has the answer, fall back to a web search.

All done with plain LLM calls (no function-calling), so it works on Ollama and
on newer Gemini models alike.
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
except ImportError:
    from duckduckgo_search import DDGS  # type: ignore


WEB_SENTINEL = "SEARCH_WEB"


class Agent:
    """Uploads-first, then knowledge base, then web — no function-calling."""

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

    @staticmethod
    def _to_sources(docs) -> list:
        return [{
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page"),
            "snippet": d.page_content[:300].strip(),
        } for d in docs]

    def _retrieve(self, retriever, query: str):
        docs = list(retriever.invoke(query))
        context = "\n\n".join(d.page_content for d in docs)
        return context, self._to_sources(docs)

    def _web_search(self, query: str):
        web_sources, lines, results = [], [], []
        try:
            with DDGS() as ddgs:
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

    def _answer_from_docs(self, context: str, history, question, user_name,
                          label: str) -> str:
        system = (
            f"You are {ASSISTANT_NAME}, a helpful assistant chatting with a user "
            f"named {user_name}. Below is content from {label}. "
            "Answer the user's question using ONLY this content. If it does not "
            f"contain the answer, reply with exactly '{WEB_SENTINEL}' and nothing "
            f"else.\n\nContent:\n{context if context else '(nothing found)'}"
        )
        return self._answer(system, history, question)

    # --- main entry ----------------------------------------------------------
    def run_stream(self, question: str, history: list[dict] | None = None,
                   user_name: str = "there"):
        history = history or []

        # Step 1: uploaded documents take priority (if any exist).
        if self._uploads_retriever is not None:
            yield {"type": "tool", "name": "search_knowledge_base"}
            up_context, up_sources = self._retrieve(self._uploads_retriever, question)
            answer = self._answer_from_docs(
                up_context, history, question, user_name,
                "the user's uploaded documents",
            )
            if not (WEB_SENTINEL in answer.upper() and len(answer.strip()) <= 40):
                yield {"type": "final", "answer": answer,
                       "doc_sources": up_sources, "web_sources": []}
                return  # answered from uploads — done

        # Step 2: built-in knowledge base.
        yield {"type": "tool", "name": "search_knowledge_base"}
        kb_context, kb_sources = self._retrieve(self._base_retriever, question)
        answer = self._answer_from_docs(
            kb_context, history, question, user_name,
            f"a knowledge base about {KNOWLEDGE_BASE_TOPIC}",
        )
        if not (WEB_SENTINEL in answer.upper() and len(answer.strip()) <= 40):
            yield {"type": "final", "answer": answer,
                   "doc_sources": kb_sources, "web_sources": []}
            return

        # Step 3: web fallback.
        yield {"type": "tool", "name": "search_the_web"}
        web_context, web_sources = self._web_search(question)
        web_system = (
            f"You are {ASSISTANT_NAME}, a helpful assistant chatting with a user "
            f"named {user_name}. Answer the question using the web search results "
            "below (titles, snippets, links). Use the available information even "
            "if partial, summarize what the results say, and mention the most "
            "relevant source. Only say you don't know if the results are truly "
            f"empty or irrelevant.\n\nWeb results:\n{web_context}"
        )
        answer = self._answer(web_system, history, question)
        yield {"type": "final", "answer": answer,
               "doc_sources": [], "web_sources": web_sources}

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
