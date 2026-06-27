
# 🔬 Multi-Document Research Assistant 
personal Researcher Assistant chatbot
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-brightgreen?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-Pipeline-4CAF50?style=flat-square)
![ChromaDB](https://img.shields.io/badge/VectorStore-ChromaDB-purple?style=flat-square)
![FastEmbed](https://img.shields.io/badge/Embeddings-FastEmbed_ONNX-orange?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-bge--small--en--v1.5-yellow?style=flat-square)
![Groq](https://img.shields.io/badge/LLM-Groq_API-F55036?style=flat-square)
![Ollama](https://img.shields.io/badge/Local_LLM-Ollama-black?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

A Streamlit-powered RAG (Retrieval-Augmented Generation) application that lets you upload multiple research articles or paste URLs and ask questions across all of them at once. Built with LangChain, LangGraph, ChromaDB, and a 7-model Groq fallback queue.

---
## 🚀 Live Demo

👉 **[click here](https://researcher-assistant-khktgxxvqrfadilhvkfyqz.streamlit.app/)**

---
## ✨ Features

- **Multi-source ingestion** — Upload PDF, Word (.docx), CSV, and TXT files, or paste article URLs directly (PDF links and web pages both supported)
- **Classify → Retrieve → Answer pipeline** — Built with LangGraph; each question is classified into one of six types (comparison, conclusion, relationship, methodology, reference count, general) and answered with a tailored prompt
- **Smart numbered citations** — Answers cite sources as `[1]`, `[2]`, etc., automatically mapped to extracted article titles
- **7-model fallback queue** — Automatically tries Groq models top-to-bottom until one succeeds; or manually pick a model via the ➕ button
- **Local Ollama fallback** — Falls through to a local `llama3` model if all Groq calls fail (requires Ollama installed)
- **PDF title extraction** — Reads font-size metadata from page 1 to extract the real paper title (not just the filename)
- **URL article ingestion** — Fetches web pages and PDF links, extracts article body via trafilatura / BeautifulSoup / regex cascade, and indexes them identically to uploaded files
- **Sliding memory window** — Last 3 Q&A pairs are fed back into every prompt for coherent multi-turn conversations
- **Conversation export** — Download the full session as JSON or PDF (via ReportLab)
- **RAG internals panel** — Sidebar shows retrieved chunks, relevance scores, and page numbers for every answer
- **Editable article titles** — Sidebar lets you correct any auto-extracted title before it appears in citations
- **Repetition guard** — Post-processes LLM output to collapse runaway repetition loops and strip reasoning-model `<think>` tags

---

## 🏗️ Architecture

```
User question
     │
     ▼
classify_question()          ← keyword-based router (6 types)
     │
     ▼
LangGraph pipeline
  ├── retrieve node           ← ChromaDB top-k semantic search (k=10)
  └── answer node             ← type-specific prompt + LLM fallback queue
     │
     ▼
Formatted answer + [N] citations
```

**Embedding model:** `BAAI/bge-small-en-v1.5` via FastEmbed (ONNX, no PyTorch — robust on Windows)  
**Vector store:** ChromaDB (in-memory per session)  
**Chunk size:** 1 000 tokens | **Overlap:** 150 | **Top-k:** 10

---

## 🤖 Model Queue

| Priority | Model | Provider | Notes |
|---|---|---|---|
| 1 | `llama-3.3-70b-versatile` | Groq | Smartest Llama; hits rate limits fastest |
| 2 | `openai/gpt-oss-120b` | Groq | Large, very capable |
| 3 | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq | Llama 4 |
| 4 | `qwen/qwen3-32b` | Groq | Strong reasoning |
| 5 | `openai/gpt-oss-20b` | Groq | Fast + capable |
| 6 | `llama-3.1-8b-instant` | Groq | Fastest, highest limits |
| 7 | `llama3` | Ollama (local) | Unlimited, runs on your machine |

> Groq model IDs change occasionally. Check [console.groq.com/docs/models](https://console.groq.com/docs/models) and update `MODEL_QUEUE` in the script as needed.

---

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/research-assistant.git
cd research-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>Core dependencies</summary>

```
streamlit
langchain
langchain-community
langchain-chroma
langchain-groq
langchain-text-splitters
langchain-core
langgraph
chromadb
fastembed
pypdf
docx2txt
requests
reportlab

# Optional but recommended for better URL extraction
trafilatura
beautifulsoup4

# Optional — only needed for the local Ollama fallback
langchain-ollama
```

</details>

### 4. Configure secrets

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_..."          # Required — get yours at console.groq.com
OLLAMA_BASE_URL = "http://localhost:11434"   # Optional — only for local Ollama
```

> When deploying to **Streamlit Community Cloud**, add these same keys under *Settings → Secrets* in the dashboard instead of using a local file.

---

## 🚀 Running the App

```bash
streamlit run Researchassistant.py
```

The app opens at `http://localhost:8501`.

---

## 🗂️ File Support

| Format | Loader |
|---|---|
| `.pdf` | PyPDFLoader + pypdf (title extraction) |
| `.docx` / `.doc` | Docx2txtLoader |
| `.csv` | CSVLoader |
| `.txt` | TextLoader (UTF-8) |
| URL (web page) | requests + trafilatura / BeautifulSoup |
| URL (PDF link) | requests + PyPDF |

---

## 💬 Question Types

| Type | Detected by | Prompt structure |
|---|---|---|
| `comparison` | "compare", "vs", "difference", "contrast" | Similarities / Differences / Assessment |
| `conclusion` | "conclusion", "finding", "result", "summarize" | Per-paper → common themes → conflicts → takeaway |
| `relationship` | "related", "connection", "overlap", "share" | Topic / Methodology / Citability / Verdict |
| `methodology` | "method", "tool", "dataset", "algorithm" | Markdown table + common / unique / strengths |
| `reference_count` | "how many references", "number of citations" | Deterministic count from reference section |
| `general` | everything else | Direct RAG answer |

---

## 📁 Project Structure

```
research-assistant/
├── Researchassistant.py        # Main app — single-file Streamlit application
├── .streamlit/
│   └── secrets.toml            # API keys (not committed to git)
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

All tuneable parameters are at the top of `Researchassistant.py`:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed ONNX model (swap to `bge-large` for higher quality) |
| `CHUNK_SIZE` | `1000` | Token size per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |
| `TOP_K` | `10` | Number of chunks retrieved per query |
| `MEMORY_WINDOW` | `3` | Number of prior Q&A pairs included in each prompt |
| `URL_FETCH_TIMEOUT` | `30` | Seconds before a URL fetch times out |
| `MODEL_QUEUE` | *(see above)* | Ordered list of LLMs to try |

---

## 🌐 Deployment (Streamlit Community Cloud)

1. Push the repo to GitHub (make sure `.streamlit/secrets.toml` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Point it at `Researchassistant.py`
4. Add `GROQ_API_KEY` (and optionally `OLLAMA_BASE_URL`) under *Settings → Secrets*
5. Deploy — no Docker, no server needed

---

## 🔒 Environment Notes

- `KMP_DUPLICATE_LIB_OK=TRUE` is set automatically to prevent the `libiomp5md.dll` crash on Windows (conda + native wheels conflict)
- `TOKENIZERS_PARALLELISM=false` suppresses HuggingFace tokenizer warnings in multi-threaded Streamlit
- All errors are caught and logged server-side; users see friendly guidance messages instead of raw tracebacks

---

## 📄 License

MIT — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

Built on top of [LangChain](https://github.com/langchain-ai/langchain), [LangGraph](https://github.com/langchain-ai/langgraph), [ChromaDB](https://github.com/chroma-core/chroma), [FastEmbed](https://github.com/qdrant/fastembed), and [Groq](https://groq.com/).
