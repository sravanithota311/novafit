"""FastAPI service exposing the same agent over HTTP.

Imports the same Agent the Streamlit app uses, so there's no logic duplicated.
Run with:

    uvicorn api:app --reload

Then POST to /chat:
    curl -X POST http://localhost:8000/chat \
      -H "Content-Type: application/json" \
      -d '{"question": "How many vacation days do new employees get?"}'
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from rag.agent import Agent

_bot: Agent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bot
    _bot = Agent()
    yield
    _bot = None


app = FastAPI(title="Agentic Assistant API", lifespan=lifespan)


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[Turn] = []


class DocSource(BaseModel):
    source: str
    page: int | None = None
    snippet: str


class WebSource(BaseModel):
    title: str
    url: str


class ChatResponse(BaseModel):
    answer: str
    doc_sources: list[DocSource]
    web_sources: list[WebSource]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "ready": _bot is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    assert _bot is not None, "Assistant not initialized"
    history = [turn.model_dump() for turn in req.history]
    result = _bot.ask(req.question, history=history)
    return ChatResponse(**result)
