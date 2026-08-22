# Nova — a local agentic AI assistant

Nova is an AI assistant that answers your questions by **deciding for itself**
whether to search a private knowledge base or the live web — then answers with
**citations** so you can see exactly where the information came from.

It runs **fully locally** (the model runs on your machine via
[Ollama](https://ollama.com)), so your documents and questions stay private.
The only thing that ever leaves your computer is a web search, and only when
Nova decides it needs one.

---

## What makes it "agentic"

Nova isn't a fixed pipeline. The model is given two tools and chooses which to
use for each question:

- **`search_knowledge_base`** — retrieves relevant passages from your own
  documents (Retrieval-Augmented Generation over a local FAISS index).
- **`search_the_web`** — searches DuckDuckGo for general knowledge or current
  events the documents don't cover.

Ask about your documents and it searches them; ask about the news and it
searches the web. It also remembers the conversation, so follow-up questions
work naturally.

---

## Architecture

```
nova/
├── rag/                  # reusable core (UI-agnostic)
│   ├── config.py         # all settings + branding in one place
│   ├── ingest.py         # build the index: load → split → embed → store
│   └── agent.py          # the tool-calling agent (documents + web)
├── data/                 # the private knowledge base (.md / .txt / .pdf)
├── ingest.py             # run this to (re)build the index
├── app.py                # Streamlit chat UI
├── api.py                # FastAPI service (same agent, over HTTP)
├── requirements.txt
└── .env.example
```

`app.py` and `api.py` both import the same `Agent` from `rag/agent.py`, so no
logic is duplicated between the UI and the API.

---

## Setup

### 1. Install Ollama and pull a model
Nova's agent needs a tool-calling-capable model. Download Ollama from
https://ollama.com, then:
```bash
ollama pull qwen2.5:3b
```
Leave Ollama running (it serves the model at `http://localhost:11434`).

### 2. Install Python dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Build the index
```bash
python ingest.py
```
First run also downloads a small (~90 MB) embedding model.

### 4. Run it
```bash
streamlit run app.py
```

---

## Try these

Knowledge-base questions (searches your documents):
- "How many vacation days do new employees get?"
- "Summarize the parental leave policy."
- "How much does the company match my 401k?"

Web questions (searches the internet):
- "Who is the current CEO of OpenAI?"
- "What are the latest developments in AI?"

Watch the status box — it shows which tool Nova picks for each question.

---

## Make it your own

- **Swap the knowledge base:** replace the files in `data/` with your own
  `.md`, `.txt`, or `.pdf` files, run `python ingest.py`, and restart.
- **Rename the assistant:** change `ASSISTANT_NAME` in `rag/config.py`.
- **Describe your documents:** update `KNOWLEDGE_BASE_TOPIC` in
  `rag/config.py` so the agent knows when to search documents vs. the web.
- **Tune behavior:** `TOP_K`, `CHUNK_SIZE`, `MAX_WEB_RESULTS`, `LLM_MODEL`, and
  more are all in `rag/config.py` (overridable via a `.env` file).

---

## Notes

- Web search uses DuckDuckGo (free, no API key). It can occasionally
  rate-limit under rapid repeated queries; wait a moment and retry.
- Tool-calling quality depends on the model. `qwen2.5:3b` is a good small
  default; `qwen2.5:7b` is more reliable if you have the RAM.

---

## Tech stack

Python · LangChain (tool-calling agent) · Ollama · FAISS ·
HuggingFace sentence-transformers · DuckDuckGo · Streamlit · FastAPI
