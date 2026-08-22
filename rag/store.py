"""Simple on-disk chat storage — like ChatGPT's saved conversations.

Each user gets a folder under storage/chats/<user>/, and each conversation is
one JSON file. No database needed; it's just files, which keeps the project
easy to run anywhere.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from rag.config import BASE_DIR

CHATS_DIR = BASE_DIR / "storage" / "chats"


def _safe(name: str) -> str:
    cleaned = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
    return cleaned or "user"


def _user_dir(user: str) -> Path:
    d = CHATS_DIR / _safe(user)
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_chat(user: str, title: str = "New chat") -> dict:
    """Create and save a fresh empty conversation, returning it."""
    now = time.time()
    chat = {
        "id": uuid.uuid4().hex[:12],
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    save_chat(user, chat)
    return chat


def save_chat(user: str, chat: dict) -> None:
    chat["updated_at"] = time.time()
    path = _user_dir(user) / f"{chat['id']}.json"
    path.write_text(json.dumps(chat, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chat(user: str, chat_id: str) -> dict | None:
    path = _user_dir(user) / f"{chat_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def delete_chat(user: str, chat_id: str) -> None:
    path = _user_dir(user) / f"{chat_id}.json"
    if path.exists():
        path.unlink()


def list_chats(user: str) -> list[dict]:
    """Return chat summaries (id, title, updated_at), newest first."""
    out = []
    for path in _user_dir(user).glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out.append({
                "id": data["id"],
                "title": data.get("title", "Untitled"),
                "updated_at": data.get("updated_at", 0),
                "count": len(data.get("messages", [])),
            })
        except Exception:
            continue
    out.sort(key=lambda c: c["updated_at"], reverse=True)
    return out


def make_title(first_message: str) -> str:
    """Turn the first user message into a short chat title."""
    text = " ".join(first_message.strip().split())
    return (text[:40] + "…") if len(text) > 40 else (text or "New chat")