"""Streamlit chat UI for NovaFit — agentic assistant with saved chats,
document upload, and streaming answers.

The UI is thin: real work lives in rag/agent.py (the agent), rag/store.py
(saved conversations), and rag/uploads.py (uploaded documents). Run with:

    streamlit run app.py
"""
import time
from pathlib import Path

import streamlit as st

from rag.config import ASSISTANT_NAME, INDEX_DIR
from rag.agent import Agent
from rag.providers import get_embeddings
from rag import store
from rag import uploads

st.set_page_config(page_title=ASSISTANT_NAME, page_icon="✨", layout="centered")

USER_AVATAR = "🧑"
BOT_AVATAR = "✨"

TOOL_LABELS = {
    "search_knowledge_base": "📄 Searching the knowledge base…",
    "search_the_web": "🌐 Searching the web…",
}

# --- Styling -----------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
      .block-container { max-width: 820px; padding-top: 1.5rem; padding-bottom: 5rem; }

      .app-header {
        background: linear-gradient(135deg, #0f6e5c 0%, #17a67f 55%, #2fd0a0 100%);
        color: #fff; border-radius: 18px; padding: 24px 28px;
        box-shadow: 0 8px 28px rgba(15,110,92,.22); margin-bottom: 18px;
        position: relative; overflow: hidden;
      }
      .app-header::after {
        content:""; position:absolute; right:-40px; top:-40px;
        width:190px; height:190px; border-radius:50%; background: rgba(255,255,255,.08);
      }
      .app-header h1 { margin:0; font-size:1.55rem; font-weight:700; letter-spacing:-.4px; }
      .app-header p  { margin:8px 0 0; opacity:.94; font-size:.93rem; line-height:1.5; max-width:94%; }
      .badges { margin-top:12px; }
      .badge {
        display:inline-block; background:rgba(255,255,255,.18);
        padding:4px 12px; border-radius:20px; font-size:.72rem; font-weight:500;
        margin-right:6px; backdrop-filter: blur(4px);
      }

      .stChatMessage { border-radius:14px; padding:2px 4px; margin-bottom:6px; }
      div[data-testid="stChatMessageContent"] { font-size:.96rem; line-height:1.55; }

      .stButton > button {
        border-radius:10px; border:1px solid #d9efe8; background:#f4fbf8;
        color:#22221f; font-size:.84rem; font-weight:500; text-align:left;
        padding:9px 12px; transition:all .15s ease;
      }
      .stButton > button:hover {
        border-color:#17a67f; background:#fff;
        box-shadow:0 3px 10px rgba(23,166,127,.15);
      }

      .src-card {
        background:#eefaf5; border-left:3px solid #17a67f; border-radius:8px;
        padding:10px 14px; margin:8px 0; font-size:.82rem; line-height:1.5;
      }
      .src-card b { color:#0f6e5c; }
      .src-snippet { color:#5f5e5a; margin-top:4px; display:block; }
      .web-card {
        background:#eef3f7; border-left:3px solid #3d6b96; border-radius:8px;
        padding:8px 14px; margin:8px 0; font-size:.82rem;
      }
      .web-card a { color:#2a5580; text-decoration:none; font-weight:600; }

      .welcome { text-align:center; padding:40px 20px 10px; }
      .welcome h1 { font-size:2rem; color:#0f6e5c; margin-bottom:4px; }
      .welcome p { color:#5f5e5a; font-size:1rem; margin-top:0; }

      .app-footer {
        text-align:center; color:#9a978f; font-size:.72rem; margin-top:30px;
        padding-top:14px; border-top:1px solid #eee;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = [
    "How much exercise should adults get each week?",
    "How many hours of sleep do adults need?",
    "What are the latest fitness tracker reviews?",
    "Is creatine backed by research?",
]


# --- Cached resources --------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings_cached():
    return get_embeddings()


@st.cache_resource(show_spinner="Waking up the assistant...")
def get_bot(uploads_version: int) -> Agent:
    # uploads_version is part of the cache key: bumping it rebuilds the agent
    # so newly uploaded documents become searchable.
    return Agent(embeddings=get_embeddings_cached())


def stream_words(text: str):
    """Yield the answer word-by-word for a typing effect."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


def render_sources(doc_sources: list, web_sources: list) -> None:
    if doc_sources:
        with st.expander(f"📄 From the knowledge base ({len(doc_sources)})"):
            for s in doc_sources:
                name = Path(s["source"]).stem.replace("_", " ").title()
                st.markdown(
                    f"<div class='src-card'><b>{name}</b>"
                    f"<span class='src-snippet'>{s['snippet']}…</span></div>",
                    unsafe_allow_html=True,
                )
    if web_sources:
        with st.expander(f"🌐 From the web ({len(web_sources)})"):
            for w in web_sources:
                st.markdown(
                    f"<div class='web-card'>🔗 <a href='{w['url']}' "
                    f"target='_blank'>{w['title']}</a></div>",
                    unsafe_allow_html=True,
                )


def persist_current_chat():
    """Save the active conversation to disk."""
    chat = st.session_state.current_chat
    chat["messages"] = st.session_state.messages
    if chat["title"] in ("New chat", "") and st.session_state.messages:
        first_user = next(
            (m["content"] for m in st.session_state.messages if m["role"] == "user"),
            "",
        )
        if first_user:
            chat["title"] = store.make_title(first_user)
    store.save_chat(st.session_state.user_name, chat)


# --- Guard: index must exist -------------------------------------------------
if not Path(INDEX_DIR).exists():
    st.error("The assistant isn't set up yet. Please contact the administrator.")
    st.stop()

if "uploads_version" not in st.session_state:
    st.session_state.uploads_version = 0

# --- Name screen (shown once at the start) -----------------------------------
if "user_name" not in st.session_state:
    st.markdown(
        f"""
        <div class="welcome">
          <h1>✨ Welcome to {ASSISTANT_NAME}</h1>
          <p>Your personal health &amp; fitness assistant. What should I call you?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    name = st.text_input("Your name", placeholder="e.g. Sravani",
                         label_visibility="collapsed")
    if st.button("Start chatting →", use_container_width=True):
        st.session_state.user_name = name.strip() or "there"
        # Open the most recent chat, or start a new one.
        chats = store.list_chats(st.session_state.user_name)
        if chats:
            loaded = store.load_chat(st.session_state.user_name, chats[0]["id"])
            st.session_state.current_chat = loaded
            st.session_state.messages = loaded["messages"]
        else:
            chat = store.new_chat(st.session_state.user_name)
            st.session_state.current_chat = chat
            st.session_state.messages = []
        st.rerun()
    st.stop()

user_name = st.session_state.user_name

# Safety: make sure a current chat exists.
if "current_chat" not in st.session_state:
    chat = store.new_chat(user_name)
    st.session_state.current_chat = chat
    st.session_state.messages = []

# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### ✨ {ASSISTANT_NAME}")
    st.write(f"Hi **{user_name}** 👋")

    if st.button("➕ New chat", use_container_width=True):
        chat = store.new_chat(user_name)
        st.session_state.current_chat = chat
        st.session_state.messages = []
        st.rerun()

    st.markdown("#### 💬 Your chats")
    chats = store.list_chats(user_name)
    if not chats:
        st.caption("No saved chats yet.")
    for c in chats:
        col1, col2 = st.columns([0.82, 0.18])
        is_current = c["id"] == st.session_state.current_chat["id"]
        label = ("• " if is_current else "") + c["title"]
        if col1.button(label, key=f"open_{c['id']}", use_container_width=True):
            loaded = store.load_chat(user_name, c["id"])
            if loaded:
                st.session_state.current_chat = loaded
                st.session_state.messages = loaded["messages"]
                st.rerun()
        if col2.button("🗑", key=f"del_{c['id']}"):
            store.delete_chat(user_name, c["id"])
            if is_current:
                remaining = store.list_chats(user_name)
                if remaining:
                    loaded = store.load_chat(user_name, remaining[0]["id"])
                    st.session_state.current_chat = loaded
                    st.session_state.messages = loaded["messages"]
                else:
                    chat = store.new_chat(user_name)
                    st.session_state.current_chat = chat
                    st.session_state.messages = []
            st.rerun()

    st.divider()
    st.markdown("#### 📎 Add a document")
    st.caption("Upload a PDF, TXT, or MD file to chat with it.")
    up = st.file_uploader("Upload", type=["pdf", "txt", "md"],
                          label_visibility="collapsed")
    if up is not None:
        if st.button("Add to knowledge base", use_container_width=True):
            with st.spinner("Indexing your document…"):
                path = uploads.save_upload(up.name, up.getvalue())
                added = uploads.add_to_index(path, get_embeddings_cached())
            if added:
                st.session_state.uploads_version += 1  # rebuild the agent
                st.success(f"Added '{up.name}' ({added} chunks).")
                st.rerun()
            else:
                st.warning("Couldn't read that file.")
    existing = uploads.list_uploaded_files()
    if existing:
        st.caption("Uploaded: " + ", ".join(existing))
        if st.button("Clear uploaded documents", use_container_width=True):
            uploads.clear_uploads()
            st.session_state.uploads_version += 1
            st.rerun()

    st.divider()
    st.caption("🔒 Private — the AI runs on this device. Only web searches "
               "(when needed) leave your computer.")
    if st.button("👤 Switch user", use_container_width=True):
        for k in ("user_name", "current_chat", "messages"):
            st.session_state.pop(k, None)
        st.rerun()

# --- Header ------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-header">
      <h1>✨ {ASSISTANT_NAME}</h1>
      <p>Hi {user_name}! Ask me anything about health & fitness — or upload a
      document and ask about that. I'll search the web when needed, and always
      show you where the answer came from.</p>
      <div class="badges">
        <span class="badge">🧠 Agentic</span>
        <span class="badge">📄 Knowledge base + uploads</span>
        <span class="badge">🌐 Web search</span>
        <span class="badge">💬 Saved chats</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

bot = get_bot(st.session_state.uploads_version)

# --- Example prompts (only on an empty chat) ---------------------------------
if not st.session_state.messages:
    st.caption(f"💡 {user_name}, try one of these to get started:")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

# --- Render history ----------------------------------------------------------
for msg in st.session_state.messages:
    avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("doc_sources", []), msg.get("web_sources", []))

# --- Resolve input -----------------------------------------------------------
prompt = st.chat_input(f"Ask {ASSISTANT_NAME} anything...")
if not prompt and "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

# --- Handle a new question ---------------------------------------------------
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        history = st.session_state.messages[:-1]
        answer, doc_sources, web_sources = "", [], []
        with st.status("Thinking…", expanded=True) as status:
            for ev in bot.run_stream(prompt, history=history, user_name=user_name):
                if ev["type"] == "tool":
                    status.write(TOOL_LABELS.get(ev["name"], f"Using {ev['name']}…"))
                else:
                    answer = ev["answer"]
                    doc_sources = ev["doc_sources"]
                    web_sources = ev["web_sources"]
            status.update(label="Done", state="complete", expanded=False)
        # Stream the answer out word-by-word for a GPT-like feel.
        st.write_stream(stream_words(answer))
        render_sources(doc_sources, web_sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "doc_sources": doc_sources,
        "web_sources": web_sources,
    })
    persist_current_chat()
    st.rerun()

# --- Footer ------------------------------------------------------------------
st.markdown(
    f"<div class='app-footer'>{ASSISTANT_NAME} · a local AI agent for health & "
    "fitness · educational information, not medical advice</div>",
    unsafe_allow_html=True,
)