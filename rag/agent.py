"""The agent: an LLM that chooses between searching documents and the web.

Tools given to the model:
  * search_knowledge_base -> the built-in FAISS index PLUS any uploaded docs
  * search_the_web        -> DuckDuckGo (free, no API key)

The model decides which tool(s) to call for each question. Requires a
tool-calling-capable model (qwen2.5 recommended).
"""
from __future__ import annotations

from pathlib import Path

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

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

# The DuckDuckGo library was renamed from `duckduckgo_search` to `ddgs`.
try:
    from ddgs import DDGS
except ImportError:  # older installs
    from duckduckgo_search import DDGS  # type: ignore


SYSTEM_PROMPT = (
    f"You are {ASSISTANT_NAME}, a helpful AI assistant.\n\n"
    "You MUST answer questions by calling one of your tools. Never answer "
    "from your own memory, and never say you cannot help without trying a "
    "tool first.\n\n"
    "Your tools:\n"
    f"1. search_knowledge_base — use for anything about {KNOWLEDGE_BASE_TOPIC}, "
    "and for any documents the user has uploaded.\n"
    "2. search_the_web — use for EVERYTHING ELSE: general knowledge, people, "
    "companies, current events, definitions, or anything the knowledge base "
    "would not cover.\n\n"
    "How to decide:\n"
    "- If the question relates to the knowledge base or uploaded documents, "
    "call search_knowledge_base.\n"
    "- For ANY other question (e.g. 'who is the CEO of OpenAI', 'latest "
    "news', 'what is X'), you MUST call search_the_web. Do not refuse.\n"
    "- Base your answer only on what the tool returns. Do not invent facts.\n"
    "- Only if a tool returns nothing useful may you say you don't know.\n"
    "- Keep answers concise and clear."
)


class Agent:
    """Tool-calling agent over a knowledge base (+ uploads) and web search."""

    def __init__(self, embeddings=None) -> None:
        # Build the index automatically if it's missing (e.g. first cloud run).
        ensure_index()

        embeddings = embeddings or get_embeddings()
        base_vs = FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
        self._base_retriever = base_vs.as_retriever(search_kwargs={"k": TOP_K})
        # Retriever over uploaded documents (None until something is uploaded).
        self._uploads_retriever = load_uploads_retriever(embeddings, k=TOP_K)

        self._doc_sources: list = []
        self._web_sources: list = []

        llm = get_chat_model()

        tools = self._build_tools()
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT + "\n\nYou are chatting with a user named "
             "{user_name}. Address them by their name naturally and warmly when "
             "it fits, but don't overuse it."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        agent = create_tool_calling_agent(llm, tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            handle_parsing_errors=True,
            max_iterations=4,
            verbose=False,
        )

    # --- Tools ---------------------------------------------------------------
    def _build_tools(self) -> list:
        bot = self

        @tool
        def search_knowledge_base(query: str) -> str:
            """Search the knowledge base and any uploaded documents."""
            docs = list(bot._base_retriever.invoke(query))
            if bot._uploads_retriever is not None:
                docs += list(bot._uploads_retriever.invoke(query))
            if not docs:
                return "No relevant documents found in the knowledge base."
            for d in docs:
                bot._doc_sources.append({
                    "source": d.metadata.get("source", "unknown"),
                    "page": d.metadata.get("page"),
                    "snippet": d.page_content[:300].strip(),
                })
            return "\n\n".join(d.page_content for d in docs)

        @tool
        def search_the_web(query: str) -> str:
            """Search the public web for general or current information."""
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=MAX_WEB_RESULTS))
            except Exception as exc:
                return f"Web search failed: {exc}"
            if not results:
                return "No web results found."
            lines = []
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                bot._web_sources.append({"title": title, "url": href})
                lines.append(f"{title}\n{body}\n({href})")
            return "\n\n".join(lines)

        return [search_knowledge_base, search_the_web]

    # --- History helper ------------------------------------------------------
    @staticmethod
    def _to_messages(history: list[dict]) -> list:
        messages = []
        for turn in history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))
        return messages

    # --- Streaming run (used by the UI) --------------------------------------
    def run_stream(self, question: str, history: list[dict] | None = None,
                   user_name: str = "there"):
        """Run the agent, yielding tool events then a final result dict."""
        self._doc_sources = []
        self._web_sources = []
        payload = {
            "input": question,
            "chat_history": self._to_messages(history or []),
            "user_name": user_name,
        }
        final = ""
        for chunk in self.executor.stream(payload):
            if "actions" in chunk:
                for action in chunk["actions"]:
                    yield {"type": "tool", "name": action.tool}
            if "output" in chunk:
                final = chunk["output"]
        yield {
            "type": "final",
            "answer": final,
            "doc_sources": self._doc_sources,
            "web_sources": self._web_sources,
        }

    # --- Blocking run (used by the API) --------------------------------------
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