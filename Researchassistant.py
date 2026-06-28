"""
🔬 Multi-Document Research Assistant — v3
…
Stack:
  • Streamlit   — UI
  • LangChain   — loaders, splitting, embeddings, retrieval (RAG)
  • LangGraph   — the question → classify → retrieve → answer pipeline
  • Chroma      — in-memory vector store
  • HuggingFace — local (free) embeddings
  • requests (+ optional trafilatura / BeautifulSoup) — fetch & extract URL articles
  • LLM queue with automatic fallback (or pick one manually via the ➕ button):
        1. llama-3.3-70b      (Groq — smartest Llama, hits rate limits faster)
        2. gpt-oss-120b       (Groq — large, very capable open model)
        3. llama-4-scout      (Groq — Llama 4)
        4. qwen3-32b          (Groq — strong reasoning)
        5. gpt-oss-20b        (Groq — fast + capable)
        6. llama-3.1-8b       (Groq — fastest, highest limits)
        7. llama3 local       (Ollama — unlimited, runs on your machine)
      * Groq model IDs change over time; edit MODEL_QUEUE below to match
        the live list at https://console.groq.com/docs/models

"""

import io
import os
import html
import json
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import TypedDict
from urllib.parse import urlparse

# ------- Server-side logging ---------
# Users never see raw errors (we show friendly messages instead), but every
# failure is logged here so it shows up in the terminal / Streamlit Cloud logs
# ("Manage app" → logs) where you can diagnose it.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("research_assistant")

# Environment must be set before the embedding backend (onnxruntime) loads.
# KMP_DUPLICATE_LIB_OK avoids the "libiomp5md.dll already initialized" crash that
# conda + native wheels trigger on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st
import requests

from langchain_community.document_loaders import (
    PyMuPDFLoader, Docx2txtLoader, CSVLoader, TextLoader, BSHTMLLoader, ArxivLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings  # ONNX, no torch — robust on Windows
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langgraph.graph import StateGraph, START, END

# Ollama is optional — only needed for the local fallback model.
try:
    from langchain_ollama import ChatOllama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False


# ------- Config --------
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 150
TOP_K           = 10
MEMORY_WINDOW   = 3


URL_FETCH_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.8",
}
URL_FETCH_TIMEOUT = 30   # seconds

# Secrets — read from .streamlit/secrets.toml (or Streamlit Cloud secrets UI).
GROQ_API_KEY    = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
OLLAMA_BASE_URL = st.secrets.get("OLLAMA_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))


MODEL_QUEUE = [
    {"provider": "groq",   "model": "llama-3.3-70b-versatile",                  "label": "Llama 3.3 70B (Groq)"},
    {"provider": "groq",   "model": "openai/gpt-oss-120b",                      "label": "GPT-OSS 120B (Groq)"},
    {"provider": "groq",   "model": "meta-llama/llama-4-scout-17b-16e-instruct","label": "Llama 4 Scout (Groq)"},
    {"provider": "groq",   "model": "qwen/qwen3-32b",                           "label": "Qwen3 32B (Groq)"},
    {"provider": "groq",   "model": "openai/gpt-oss-20b",                       "label": "GPT-OSS 20B (Groq)"},
    {"provider": "groq",   "model": "llama-3.1-8b-instant",                     "label": "Llama 3.1 8B (Groq)"},
    {"provider": "ollama", "model": "llama3",                                   "label": "Llama 3 (Ollama, local)"},
]

AUTO_LABEL = "Auto (smart fallback)"


#  Step 1: Load any file type
def load_file(file_path: str, filename: str) -> list[Document]:
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext in [".doc", ".docx"]:
        loader = Docx2txtLoader(file_path)
    elif ext in [".html", ".htm"]:
        loader = BSHTMLLoader(file_path)
    elif ext == ".csv":
        loader = CSVLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        # Silently skip unsupported types — no on-screen warning, but log it.
        log.warning("Unsupported file type skipped: %s (%s)", ext, filename)
        return []

    docs = loader.load()

    # Attach document-level metadata to every chunk — critical for citations.
    for doc in docs:
        doc.metadata["doc_name"] = filename
        doc.metadata["doc_id"]   = filename
    return docs


#  Extract each article's real title (for the numbered citation list)
_SKIP_PREFIXES = ("abstract", "http", "doi", "www", "journal", "proceedings",
                  "arxiv", "preprint", "vol ", "volume", "copyright", "©",
                  "submitted", "received", "accepted", "published", "issn")


def _looks_like_title(t: str) -> bool:
    t = (t or "").strip()
    if not (6 <= len(t) <= 250):                 return False
    if len(t.split()) < 2:                       return False
    if not any(c.isalpha() for c in t):          return False
    if t.lower() in ("(anonymous)", "untitled", "microsoft word"):  return False
    return True


def _guess_title_from_text(text: str, fallback_filename: str) -> str:
    """First substantial line that isn't a journal header / boilerplate."""
    for raw in (text or "").splitlines():
        line = raw.strip()
        if len(line.split()) >= 3 and not line.lower().startswith(_SKIP_PREFIXES):
            return line[:200]
    return Path(fallback_filename).stem


def _pdf_title_from_doc(doc, fallback_filename: str) -> str:
    """Title from an ALREADY-OPEN PyMuPDF document (so we don't re-parse the file)."""
    # 1) Metadata title — only if it looks real.
    try:
        meta_title = ((doc.metadata or {}).get("title") or "").strip()
        if _looks_like_title(meta_title):
            return meta_title[:200]
    except Exception:
        pass

    # 2) Largest-font text near the top of page 1.
    try:
        page = doc[0]
        spans = []  # (font_size, y_position, text)
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if text and text.strip():
                        size = round(float(span.get("size", 0.0)), 1)
                        y = span.get("bbox", (0, 0, 0, 0))[1]
                        spans.append((size, y, text))

        if spans:
            max_size = max(s[0] for s in spans)
            top = [s for s in spans if s[0] >= max_size - 0.5]   # the biggest-font run(s)
            top.sort(key=lambda s: s[1])                         # top of page first
            title = " ".join(" ".join(t.strip() for _, _, t in top).split())
            if _looks_like_title(title):
                return title[:200]

        # 3) Fall back to the first sensible plain-text line.
        return _guess_title_from_text(page.get_text() or "", fallback_filename)
    except Exception:
        return Path(fallback_filename).stem


#  Fetch an article from a URL
class URLSource:
    """Mimics a Streamlit UploadedFile so URL articles reuse the whole pipeline."""

    def __init__(self, name: str, data: bytes, title: str, url: str):
        self.name = name
        self._data = data
        self.size = len(data)
        self.title = title
        self.url = url

    def getvalue(self) -> bytes:
        return self._data


def _sanitize_filename(text: str, ext: str) -> str:
    """Turn a title/URL into a safe, short filename with the given extension."""
    keep = "".join(c if (c.isalnum() or c in " -_") else "_" for c in (text or "")).strip()
    keep = " ".join(keep.split())[:80] or "article"
    return f"{keep}{ext}"


def _unique_name(name: str, taken: set) -> str:
    """Ensure a source name is unique among the already-loaded sources."""
    if name not in taken:
        return name
    stem, ext = os.path.splitext(name)
    i = 2
    while f"{stem}_{i}{ext}" in taken:
        i += 1
    return f"{stem}_{i}{ext}"


def _arxiv_id_from_url(url: str) -> str:
    import re
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", url, re.I)
    return m.group(1) if m else ""


def _title_from_pdf_bytes(data: bytes, url: str) -> str:
    """Reuse the PDF title cascade on bytes downloaded from a URL."""
    try:
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        return _pdf_title_from_doc(doc, Path(urlparse(url).path).name or "article.pdf")
    except Exception:
        return Path(urlparse(url).path).stem or "Article"


def _extract_html_title_and_text(html_text: str, url: str) -> tuple[str, str]:
    """Pull (title, main_body_text) out of a web page, best-effort.

    Tries, in order: trafilatura (best article extractor, optional) →
    BeautifulSoup (article/main + headings/paragraphs) → a plain regex strip.
    """
    title, body = "", ""

    # 1) trafilatura — best-in-class article extraction (optional dependency).
    try:
        import trafilatura
        body = trafilatura.extract(html_text, include_comments=False,
                                   include_tables=True, url=url) or ""
        meta = trafilatura.extract_metadata(html_text)
        if meta and getattr(meta, "title", None):
            title = meta.title.strip()
    except Exception:
        pass

    # 2) BeautifulSoup fallback.
    if not body.strip():
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_text, "html.parser")
            if not title:
                og = soup.find("meta", attrs={"property": "og:title"})
                if og and og.get("content"):
                    title = og["content"].strip()
                elif soup.title and soup.title.string:
                    title = soup.title.string.strip()
                else:
                    h1 = soup.find("h1")
                    if h1:
                        title = h1.get_text(strip=True)
            for tag in soup(["script", "style", "nav", "footer",
                             "header", "aside", "form", "noscript"]):
                tag.decompose()
            main = soup.find("article") or soup.find("main") or soup.body or soup
            parts = [el.get_text(" ", strip=True)
                     for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li"])]
            parts = [p for p in parts if p]
            body = "\n".join(parts) if parts else main.get_text("\n", strip=True)
        except Exception:
            pass

    # 3) Last-ditch regex strip (no extra deps).
    if not body.strip():
        import re
        if not title:
            m = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.I | re.S)
            title = (m.group(1).strip() if m else "")
        tmp = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html_text, flags=re.I | re.S)
        tmp = re.sub(r"<[^>]+>", " ", tmp)
        body = re.sub(r"\s+", " ", html.unescape(tmp)).strip()

    title = " ".join((title or "").split())[:200] or (Path(urlparse(url).path).stem or url)
    body = (title + "\n\n" + body).strip()   # keep the title inside the indexed text
    return title, body


def fetch_url_article(url: str) -> URLSource:
    """Download a URL and wrap it as a URLSource (PDF link → .pdf, web page → .txt)."""
    if not urlparse(url).scheme:
        url = "https://" + url

    arxiv_id = _arxiv_id_from_url(url)
    if arxiv_id:
        docs = ArxivLoader(query=arxiv_id, load_max_docs=1).load()
        if docs:
            meta = docs[0].metadata
            title = (meta.get("Title") or "").strip() or f"arXiv {arxiv_id}"
            body = (title + "\n\n" + docs[0].page_content).strip()
            name = _sanitize_filename(title, ".txt")
            return URLSource(name, body.encode("utf-8"), title, url)

    resp = requests.get(url, headers=URL_FETCH_HEADERS, timeout=URL_FETCH_TIMEOUT)
    resp.raise_for_status()

    ctype = resp.headers.get("Content-Type", "").lower()
    is_pdf = "application/pdf" in ctype or urlparse(url).path.lower().endswith(".pdf")

    if is_pdf:
        data = resp.content
        title = _title_from_pdf_bytes(data, url)
        name = _sanitize_filename(title or Path(urlparse(url).path).stem, ".pdf")
        return URLSource(name, data, title, url)

    # HTML / text page → extract the article body and store it as a .txt source.
    title, body = _extract_html_title_and_text(resp.text, url)
    if not body.strip():
        raise ValueError("No readable text could be extracted from that page.")
    name = _sanitize_filename(title, ".txt")
    return URLSource(name, body.encode("utf-8"), title, url)


#   FAST title pass reads ONLY the first page of each PDF
@st.cache_resource(show_spinner="🔖 Reading titles...")
def extract_titles_fast(file_signatures: tuple, _uploaded_files: list) -> list:
    """
    Title only — opens each PDF and reads JUST the first page (where the title is),
    so titles appear immediately, before the heavy full-document indexing runs.
    URL articles already carry a pre-extracted .title, so we just reuse that.
    """
    tmp_dir = tempfile.gettempdir()
    titles = []
    for uf in _uploaded_files:

        pre_title = getattr(uf, "title", None)
        if pre_title:
            titles.append(pre_title)
            continue

        name = uf.name
        ext = Path(name).suffix.lower()
        path = os.path.join(tmp_dir, name)
        with open(path, "wb") as f:
            f.write(uf.getvalue())
        try:
            if ext == ".pdf":
                import fitz
                doc = fitz.open(path)
                titles.append(_pdf_title_from_doc(doc, name))
            elif ext in (".doc", ".docx"):
                import docx2txt
                titles.append(_guess_title_from_text(docx2txt.process(path) or "", name))
            else:  # txt / csv
                titles.append(_guess_title_from_text(uf.getvalue().decode("utf-8", errors="ignore"), name))
        except Exception:
            titles.append(Path(name).stem)
    return titles



@st.cache_resource(show_spinner="📚 Indexing documents…")
def process_documents(file_signatures: tuple, _uploaded_files: list):
    """Parse every page once → (vectorstore for RAG, {filename: full_text})."""
    tmp_dir = tempfile.gettempdir()
    all_docs, texts, errors = [], {}, []

    for uf in _uploaded_files:
        name = uf.name
        ext = Path(name).suffix.lower()
        path = os.path.join(tmp_dir, name)
        with open(path, "wb") as f:
            f.write(uf.getvalue())
        try:
            if ext == ".pdf":
                import fitz
                pages = [(page.get_text() or "") for page in fitz.open(path)]
                docs = [Document(page_content=pg,
                                 metadata={"doc_name": name, "doc_id": name, "page": i})
                        for i, pg in enumerate(pages)]
                full = "\n".join(pages)
            else:  # docx / csv / txt
                docs = load_file(path, name)
                full = "\n".join(d.page_content for d in docs)
        except Exception as e:
            log.exception("Failed to parse document: %s", name)
            errors.append(f"{name} — {type(e).__name__}: {e}")
            docs, full = [], ""
        all_docs.extend(docs)
        texts[name] = full

    if not all_docs:
        return None, texts, errors

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    splits = splitter.split_documents(all_docs)
    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)   # ONNX, no torch
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore, texts, errors


def count_references_in_text(text: str) -> int:
    """Heuristically count entries in the References/Bibliography section."""
    import re
    if not text:
        return 0

    low = text.lower()

    starts = [low.rfind("\nreferences"), low.rfind("references\n"),
              low.rfind("\nbibliography"), low.rfind("bibliography\n"),
              low.rfind("\nreferences cited")]
    idx = max(starts)
    section = text[idx:] if idx != -1 else text

    # 1) Numbered list:  [1] …  or  1. …  → highest number is the reference count.
    nums = re.findall(r'(?m)^\s*\[(\d{1,4})\]', section)
    if not nums:
        nums = re.findall(r'(?m)^\s*(\d{1,4})[\.\)]\s+[A-Z"\']', section)
    if nums:
        ints = [int(n) for n in nums if int(n) < 2000]   # ignore stray years
        if ints:
            return max(ints)


    inline = [int(x) for x in re.findall(r'\[(\d{1,4})\]', section) if int(x) < 2000]
    if inline:
        return max(inline)


    year_entries = re.findall(r'\((?:19|20)\d{2}[a-z]?\)', section)
    return len(year_entries)


# Step 3: Classify question type
def classify_question(question: str) -> str:
    q = question.lower()
    comparison_words = {"difference", "compare", "versus", "vs", "contrast",
                        "better", "worse", "similar", "unlike", "both"}
    conclusion_words = {"conclusion", "finding", "result", "outcome",
                        "summarise", "summarize", "overall", "takeaway"}
    relation_words   = {"related", "connection", "link", "overlap", "common", "share"}
    method_words     = {"tool", "method", "methodology", "technique", "approach",
                        "algorithm", "framework", "used", "implement", "dataset"}

    tokens = set(q.replace("?", " ").split())

    # "how many references / number of citations" → deterministic counting, not the LLM.
    counting = {"how many", "number of", "total", "count", "no of", "no. of"}
    ref_words = {"reference", "references", "citation", "citations", "cited", "bibliography"}
    if (tokens & ref_words) and (any(c in q for c in counting)):
        return "reference_count"

    if tokens & comparison_words: return "comparison"
    if tokens & conclusion_words: return "conclusion"
    if tokens & relation_words:   return "relationship"
    if tokens & method_words:     return "methodology"
    return "general"


#  Step 4: Prompts per question type
PROMPTS = {
    "comparison": """You are a research assistant. Compare and contrast based ONLY on the documents provided.
Structure your answer as:
- **Similarities:** what the papers agree on
- **Differences:** where they diverge (tools, methods, conclusions)
- **Your assessment:** which approach seems stronger and why

Documents:
{context}

Question: {question}""",

    "conclusion": """You are a research assistant. Synthesise the key conclusions from the provided documents.
Structure your answer as:
- **Conclusion per paper**
- **Common themes across all papers**
- **Conflicting findings** (if any)
- **Overall takeaway**

Documents:
{context}

Question: {question}""",

    "relationship": """You are a research assistant. Analyse whether these documents are related.
Structure your answer as:
- **Topic overlap:** do they study the same problem?
- **Methodology connection:** do they use similar approaches?
- **Citability:** could one paper support/challenge another?
- **Verdict:** related / partially related / unrelated

Documents:
{context}

Question: {question}""",

    "methodology": """You are a research assistant. Extract and compare the methodology, tools, datasets and techniques used.
Structure your answer as:
- **Methodology / tools / datasets per paper** (use a Markdown table if possible)
- **What is common across papers**
- **Unique approaches in each paper**
- **Methodological strengths & weaknesses**

Documents:
{context}

Question: {question}""",

    "general": """You are a research assistant. Answer the question based ONLY on the provided documents.
If the answer isn't in any document, say so clearly.

Documents:
{context}

Question: {question}""",
}

# Appended to every prompt. The citation style depends on how many docs there are.
CITATION_MULTI = """

CITATION RULES:
- Each SOURCE above is numbered, e.g. "SOURCE [1]".
- When you use information from a source, cite it inline with its bracket number — [1], [2], [3] — matching those numbers.
- Do NOT write out file names in your answer; use the bracket numbers only.
"""

CITATION_SINGLE = """

NOTE: There is only ONE document, so every answer comes from it.
Do NOT mention, repeat, or cite the document's name/filename — just answer the question directly.
"""


ANSWER_STYLE = """

FORMATTING:
- Write in clear, concise prose with short paragraphs and bullet points where helpful.
- Use **bold** only for short inline labels. Do NOT use markdown headers (#, ##, ###).
- Do not repeat sentences. Keep the answer focused and well-structured.
"""


#  Step 5: LLM factory + fallback queue

@st.cache_resource(show_spinner=False)
def get_llm(provider: str, model: str):
    """Build (and cache) one chat model. Returns None if it can't be created."""
    if provider == "groq":
        if not GROQ_API_KEY:
            return None
        return ChatGroq(
            api_key=GROQ_API_KEY, model=model,
            temperature=0.3, max_tokens=1536,
            model_kwargs={"frequency_penalty": 0.5, "presence_penalty": 0.3},
        )
    if provider == "ollama":
        if not _OLLAMA_AVAILABLE:
            return None
        return ChatOllama(
            model=model, base_url=OLLAMA_BASE_URL,
            temperature=0.3, num_predict=1536, repeat_penalty=1.3,
        )
    return None


def _collapse_repetition(text: str) -> str:
    """Safety net: strip reasoning tags + collapse runaway repetition."""
    import re
    if not text:
        return text

    # 0) Remove reasoning-model chain-of-thought (e.g. Qwen3 emits <think>…</think>).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)   # drop an unclosed tag too
    text = text.strip()

    # 1) Drop near-identical consecutive lines (keep the first occurrence).
    out_lines, prev_norm, run = [], None, 0
    for ln in text.split("\n"):
        norm = re.sub(r"\s+", " ", ln.strip().lower())
        if norm and norm == prev_norm:
            run += 1
            if run >= 1:        # second identical line onward → skip
                continue
        else:
            run, prev_norm = 0, norm
        out_lines.append(ln)
    text = "\n".join(out_lines)

    # 2) Collapse a sentence/phrase repeated back-to-back within the text.
    text = re.sub(r"(.{15,200}?[.!?])(\s*\1){2,}", r"\1", text)
    # 3) Collapse the same token repeated absurdly many times (e.g. "the the the…").
    text = re.sub(r"\b(\w{1,20})(\s+\1\b){4,}", r"\1", text, flags=re.IGNORECASE)
    return text.strip()


def _start_stream(llm, messages):
    """Begin a streaming generation and pull the FIRST chunk eagerly.

    Pulling the first token here means a model that fails immediately (rate limit,
    bad id, Ollama down) raises *now* — so generate_with_fallback can fall through to
    the next model before we hand the live generator to the UI. Returns a generator
    that yields plain text chunks (first token first, then the rest of the stream).
    """
    stream_iter = llm.stream(messages)
    first = next(stream_iter)   # may raise (or StopIteration on an empty stream) → caller falls back

    def _gen():
        yield getattr(first, "content", str(first))
        for chunk in stream_iter:
            yield getattr(chunk, "content", str(chunk))

    return _gen()


def generate_with_fallback(messages, preferred_label: str = "", stream: bool = False):
    """Try models until one answers. If preferred_label is set, try it first, then the rest.

    stream=False → returns (cleaned_text, label).
    stream=True  → returns (text_chunk_generator, label); the caller iterates it (e.g.
                   via st.write_stream) and is responsible for collapsing repetition on
                   the final assembled string.
    """
    queue = MODEL_QUEUE
    if preferred_label and preferred_label != AUTO_LABEL:
        chosen = [m for m in MODEL_QUEUE if m["label"] == preferred_label]
        rest   = [m for m in MODEL_QUEUE if m["label"] != preferred_label]
        queue  = chosen + rest   # picked model first, others as safety fallback

    last_err = None
    for spec in queue:
        llm = get_llm(spec["provider"], spec["model"])
        if llm is None:
            continue
        try:
            if stream:
                return _start_stream(llm, messages), spec["label"]
            resp = llm.invoke(messages)
            text = getattr(resp, "content", str(resp))
            if text and text.strip():
                return _collapse_repetition(text), spec["label"]
        except Exception as e:   # rate limit, bad model id, Ollama not running…
            last_err = e
            log.warning("Model %s failed, trying next: %s", spec["label"], e)
            continue
    raise RuntimeError(
        "All models in the queue failed. Check your GROQ_API_KEY / model IDs, "
        f"or start Ollama for the local fallback. Last error: {last_err}"
    )


#  Step 6: Retrieval helper
def retrieve_and_format(vectorstore, question: str, name_to_num: dict, name_to_title: dict, k: int = TOP_K):
    """Retrieve chunks and label each source by its article number + title, e.g. SOURCE [1]: <title>.

    Uses similarity_search_with_relevance_scores so we can surface a relevance score
    (0–1) on every source and expose the raw scored chunks to the RAG debug panel.
    Returns (formatted_context, source_labels, raw_chunks) where raw_chunks is a list
    of (text, score, page) tuples for inspection.
    """
    results = vectorstore.similarity_search_with_relevance_scores(question, k=k)


    grouped: dict[str, list[str]] = {}
    best_score: dict[str, float] = {}
    pages: dict[str, set] = {}
    raw_chunks: list = []
    seen: set = set()
    for doc, score in results:
        fingerprint = " ".join(doc.page_content.split()).lower()[:300]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        name = doc.metadata.get("doc_name", "Unknown")
        score = round(float(score), 3)
        page = doc.metadata.get("page")
        grouped.setdefault(name, []).append(doc.page_content)
        best_score[name] = max(best_score.get(name, -1.0), score)
        if page is not None:
            pages.setdefault(name, set()).add(page)
        raw_chunks.append((doc.page_content, score, page))

    # Order the sources by their article number so [1] is always the first article.
    ordered = sorted(grouped.items(), key=lambda kv: name_to_num.get(kv[0], 999))

    formatted, sources = "", []
    for doc_name, contents in ordered:
        num = name_to_num.get(doc_name, "?")
        title = name_to_title.get(doc_name, doc_name)   # title, fallback to filename
        formatted += f"\n\n{'='*60}\nSOURCE [{num}]: {title}\n{'='*60}\n"
        formatted += "\n---\n".join(contents)
        label = f"[{num}] {title}"
        if pages.get(doc_name):                                   # surface page numbers (1-based)
            label += f"  (pp. {', '.join(str(p + 1) for p in sorted(pages[doc_name]))})"
        if doc_name in best_score:                                # surface best relevance score
            label += f"  ·  relevance: {best_score[doc_name]:.3f}"
        sources.append(label)
    return formatted, sources, raw_chunks


#  Step 7: LangGraph pipeline
class GraphState(TypedDict):
    question: str
    q_type: str
    context: str
    sources: list
    answer: str
    model_used: str
    model_pref: str
    raw_chunks: list
    prompt_used: str
    llm_messages: list


@st.cache_resource(show_spinner=False)
def build_graph(doc_names: tuple, doc_titles: tuple, _vectorstore, _doc_texts: dict):
    """Compile the classify → retrieve → generate graph (cached per upload set)."""
    # Map each filename to its article number (1-based, in upload order) and title.
    name_to_num = {name: i + 1 for i, name in enumerate(doc_names)}
    name_to_title = {name: doc_titles[i] for i, name in enumerate(doc_names)}
    single_doc = len(doc_names) == 1

    def classify_node(state: GraphState) -> GraphState:
        state["q_type"] = classify_question(state["question"])
        return state

    def retrieve_node(state: GraphState) -> GraphState:
        # Only reached for non-reference_count questions (see route_after_classify).
        context, sources, raw_chunks = retrieve_and_format(
            _vectorstore, state["question"], name_to_num, name_to_title)
        state["context"], state["sources"], state["raw_chunks"] = context, sources, raw_chunks
        return state

    def _recent_history_text() -> str:
        """Last MEMORY_WINDOW Q/A pairs (excluding the current question) as plain text.

        Gives the model short-term conversational memory — a lightweight sliding-window
        equivalent of ConversationBufferWindowMemory, sourced from the chat transcript.
        """
        history = st.session_state.get("messages", [])[-(MEMORY_WINDOW * 2):-1]
        lines = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
                 for m in history]
        return "\n".join(lines)

    def generate_node(state: GraphState) -> GraphState:
        state.setdefault("sources", [])
        state.setdefault("raw_chunks", [])

        # Deterministic reference count — computed in code, not guessed by the LLM.
        if state["q_type"] == "reference_count":
            lines, total = [], 0
            for name in doc_names:
                n = count_references_in_text(_doc_texts.get(name, ""))
                total += n
                if single_doc:
                    lines.append(f"This document contains **{n} references**.")
                else:
                    lines.append(f"- **[{name_to_num[name]}] {name_to_title[name]}** — {n} references")
            if single_doc:
                state["answer"] = lines[0]
            else:
                state["answer"] = ("**References cited per document:**\n\n"
                                   + "\n".join(lines)
                                   + f"\n\n**Total across all {len(doc_names)} documents: {total} references**")
            state["model_used"], state["prompt_used"], state["llm_messages"] = \
                "reference counter (exact)", "reference_count", []
            return state


        rules = CITATION_SINGLE if single_doc else CITATION_MULTI
        template = PROMPTS[state["q_type"]] + rules + ANSWER_STYLE

        history_text = _recent_history_text()
        if history_text:
            # Escape braces so prior turns can't break ChatPromptTemplate's {var} parsing.
            safe_history = history_text.replace("{", "{{").replace("}", "}}")
            template = f"Prior conversation (most recent turns):\n{safe_history}\n\n---\n\n" + template

        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(context=state["context"], question=state["question"])
        state["answer"] = ""                       # filled in by the streaming UI layer
        state["llm_messages"] = messages
        state["prompt_used"] = state["q_type"]
        return state

    # Routing lives in the graph topology (declarative), not inside the node bodies:
    # reference-count questions skip retrieval entirely and go straight to generate.
    def route_after_classify(state: GraphState) -> str:
        return "generate" if state["q_type"] == "reference_count" else "retrieve"

    graph = StateGraph(GraphState)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify, {
        "retrieve": "retrieve",
        "generate": "generate",
    })
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


#  Step 8: Conversation export (JSON + PDF)
def conversation_to_json(title: str, articles: list[str], messages: list[dict]) -> bytes:
    payload = {
        "title": title,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "articles": articles,
        "conversation": messages,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def _pdf_clean(text: str) -> str:
    """Escape HTML and turn **bold** into <b> for reportlab Paragraphs."""
    import re
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return text.replace("\n", "<br/>")


def conversation_to_pdf(title: str, articles: list[str], messages: list[dict]) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=1.8*cm, bottomMargin=1.8*cm)
    styles = getSampleStyleSheet()
    q_style = ParagraphStyle("Q", parent=styles["Normal"], fontSize=11,
                             textColor=colors.HexColor("#1a5276"), spaceBefore=10, spaceAfter=4, leading=15)
    a_style = ParagraphStyle("A", parent=styles["Normal"], fontSize=10.5,
                             spaceAfter=8, leading=14)
    meta_style = ParagraphStyle("M", parent=styles["Normal"], fontSize=8,
                                textColor=colors.grey, spaceAfter=6)

    story = [Paragraph(_pdf_clean(title), styles["Title"]), Spacer(1, 6)]
    story.append(Paragraph(f"Exported: {datetime.now():%Y-%m-%d %H:%M}", meta_style))

    if articles:
        story.append(Paragraph("<b>Articles included:</b>", styles["Heading4"]))
        for a in articles:
            story.append(Paragraph(f"• {_pdf_clean(a)}", styles["Normal"]))
    story.append(Spacer(1, 12))

    for msg in messages:
        if msg["role"] == "user":
            story.append(Paragraph(f"<b>Q:</b> {_pdf_clean(msg['content'])}", q_style))
        else:
            if msg.get("model_used"):
                story.append(Paragraph(f"answered by {_pdf_clean(msg['model_used'])}", meta_style))
            story.append(Paragraph(_pdf_clean(msg["content"]), a_style))

    doc.build(story)
    return buf.getvalue()


# Suggested actions (the chips that appear after upload)
# Each chip is (button label, the full question sent to the pipeline).
SUGGESTIONS_SINGLE = [
    ("📝 Summary",          "Give me a detailed summary of this document."),
    ("🎯 Conclusion",       "What are the main conclusions of this document?"),
    ("📌 Memorize",         "List the key points from this document I should memorize for revision."),
    ("🔬 Methodology",      "What methodology, tools and datasets are used in this document?"),
    ("💡 Key findings",     "What are the key findings and contributions of this document?"),
    ("⚠️ Limitations",      "What limitations or weaknesses are discussed in this document?"),
    ("🔢 Count references", "How many references are cited in this document?"),
]
SUGGESTIONS_MULTI = [
    ("⚖️ Compare all",       "What are the key differences and similarities between these papers?"),
    ("🔬 Methodology",       "Compare the methodology, tools and datasets used across all papers."),
    ("🎯 Conclusions",       "What conclusions do these papers share, and where do they conflict?"),
    ("🔗 Relationship",      "Are these papers related? How do they connect to each other?"),
    ("📝 Summarize each",    "Summarize the findings of each paper separately."),
    ("🏆 Strongest paper",   "Which paper has the strongest approach and why?"),
    ("🔢 Count references",  "How many references are cited in each document?"),
]


def run_query(app, question: str, model_pref: str = ""):
    """Run one question through the LangGraph pipeline, streaming the answer, and store both turns."""
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        # The graph classifies, retrieves and assembles the prompt; the LLM call is
        # streamed here so tokens render live via st.write_stream.
        result = app.invoke({"question": question, "model_pref": model_pref})
        q_type     = result.get("q_type", "general")
        sources    = result.get("sources", [])
        raw_chunks = result.get("raw_chunks", [])
        debug = {
            "q_type": q_type,
            "prompt_template": result.get("prompt_used", q_type),
            "chunks": raw_chunks,
        }

        if result.get("answer"):
            # Deterministic answers (e.g. reference_count) come back fully formed.
            answer = result["answer"]
            model_used = result.get("model_used", "")
            with st.chat_message("assistant"):
                st.markdown(answer)
        else:
            # Stream the LLM response token-by-token.
            with st.chat_message("assistant"):
                stream_gen, model_used = generate_with_fallback(
                    result["llm_messages"], model_pref, stream=True)
                streamed = st.write_stream(stream_gen)
            answer = _collapse_repetition(streamed if isinstance(streamed, str) else "".join(streamed))

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "model_used": model_used,
            "q_type": q_type,
            "sources": sources,
            "debug": debug,
        })
    except Exception:

        log.exception("Query failed for question: %r", question)
        st.session_state.messages.append({
            "role": "assistant",
            "content": ("I couldn't put together an answer just now. "
                        "Please try asking again in a moment, or rephrase your question."),
            "model_used": "", "q_type": "general", "sources": [],
        })


#  Streamlit UI
st.set_page_config(page_title="Research Assistant", page_icon="🔬", layout="wide")

# Professional, light-brown theme.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Palette ── */
    :root {
        --bg:        #F3E9DC;   /* warm cream page */
        --bg2:       #EEE2D2;   /* slightly deeper cream for gradient */
        --surface:   #FBF6EE;   /* ivory cards / inputs */
        --line:      #e4d5bf;   /* hairline borders */
        --line-strong:#c9ab85;  /* input borders */
        --accent:    #8b6f4e;   /* coffee brown */
        --accent-dk: #6f5639;   /* darker coffee (hover) */
        --ink:       #2b2118;   /* primary text */
        --ink-soft:  #6b5b4a;   /* muted text */
        --glow:      rgba(139,111,78,0.20);
    }

    /* Soft vertical gradient instead of a flat fill — adds depth */
    .stApp {
        background: linear-gradient(180deg, #F6EDE1 0%, var(--bg) 45%, var(--bg2) 100%);
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EBDBC6 0%, #E5D4BD 100%);
        border-right: 1px solid #d8c4a8;
    }
    [data-testid="stHeader"] { background: transparent; }

    /* Force every piece of text to be dark (no white text on the light bg) */
    .stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp div,
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
    [data-testid="stChatMessage"], [data-testid="stChatMessage"] *,
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
    .stChatInput textarea, input, textarea, .stExpander * {
        color: var(--ink) !important;
    }
    h1, h2, h3, h4 { color: #5b4636 !important; letter-spacing: -0.01em; }

    /* App title — a touch larger and tighter, with a subtle underline accent */
    .block-container h1:first-of-type {
        font-weight: 700;
        padding-bottom: 0.2rem;
        border-bottom: 2px solid var(--line);
    }

    /* Center the content in a readable column, like ChatGPT / Claude */
    .block-container { max-width: 880px; padding-top: 2.2rem; }

    /* Professional answer typography */
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        font-size: 0.96rem;
        line-height: 1.65;
    }
    /* Tame oversized markdown headings the model might emit inside an answer */
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] h4 {
        font-size: 1.02rem !important;
        font-weight: 700;
        margin: 0.6rem 0 0.3rem 0 !important;
        color: #4a3a2c !important;
    }
    [data-testid="stChatMessage"] p { margin: 0.35rem 0; }
    [data-testid="stChatMessage"] ul, [data-testid="stChatMessage"] ol { margin: 0.2rem 0 0.4rem 1.1rem; }
    [data-testid="stChatMessage"] li { margin: 0.15rem 0; }
    [data-testid="stChatMessage"] code {
        background-color: #efe3d0; padding: 1px 5px; border-radius: 4px; font-size: 0.88rem;
    }

    /* Chat bubbles — softer corners, layered shadow, gentle entrance */
    [data-testid="stChatMessage"] {
        background-color: var(--surface);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 12px 20px;
        box-shadow: 0 1px 2px rgba(91,70,54,0.06), 0 6px 18px rgba(91,70,54,0.06);
        animation: msgIn .25s ease both;
    }
    @keyframes msgIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

    /* ════════════════════════════════════════════════════════════════════
       INPUT FIELDS — the key fix for the "odd corners".
       Streamlit wraps every <input> in a div[data-baseweb="input"] that has
       its OWN square border. Rounding only the inner <input> made the square
       wrapper corners poke out. So we round + border the WRAPPER and make the
       inner input transparent & border-free. Clean pill, no corner artifacts.
       ════════════════════════════════════════════════════════════════════ */
    div[data-baseweb="input"],
    div[data-baseweb="base-input"] {
        border-radius: 24px !important;
        background-color: var(--surface) !important;
        border: 1px solid var(--line-strong) !important;
        box-shadow: 0 1px 2px rgba(91,70,54,0.05);
        overflow: hidden;                       /* clip inner corners to the pill */
        transition: border-color .15s ease, box-shadow .15s ease;
    }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="base-input"]:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--glow);      /* soft focus glow */
    }
    /* Inner input: strip its own border/background so only the wrapper shows */
    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input,
    div[data-testid="stForm"] input,
    .stTextInput input {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        height: 44px;
        padding-left: 18px !important;
        font-size: 0.95rem;
    }
    div[data-baseweb="input"] input::placeholder { color: var(--ink-soft) !important; opacity: .8; }

    div[data-testid="stForm"] { padding: 0 !important; border: none !important; background: transparent !important; }

    /* The ➤ send button */
    div[data-testid="stForm"] button[kind="primaryFormSubmit"],
    div[data-testid="stForm"] button {
        height: 44px; border-radius: 22px !important;
        background: linear-gradient(180deg, var(--accent) 0%, var(--accent-dk) 100%);
        color: var(--surface) !important; border: none;
        font-size: 1.1rem; font-weight: 600;
        box-shadow: 0 2px 6px rgba(111,86,57,0.30);
        transition: transform .12s ease, box-shadow .12s ease, filter .12s ease;
    }
    div[data-testid="stForm"] button:hover {
        filter: brightness(1.06);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(111,86,57,0.38);
    }
    div[data-testid="stForm"] button * { color: var(--surface) !important; }

    /* ── Round "+" model button sitting at the left of the input bar ── */
    div[data-testid="stPopover"] > button {
        border-radius: 50% !important;
        width: 44px; height: 44px;
        font-size: 1.25rem; font-weight: 700;
        padding: 0 !important;
        background-color: var(--surface);
        border: 1px solid var(--line-strong);
        box-shadow: 0 1px 3px rgba(91,70,54,0.12);
        transition: transform .12s ease, border-color .12s ease, box-shadow .12s ease;
    }
    div[data-testid="stPopover"] > button:hover {
        border-color: var(--accent);
        transform: translateY(-1px) rotate(90deg);
        box-shadow: 0 3px 9px rgba(91,70,54,0.20);
    }

    /* Suggestion chips + generic buttons — pill, lift on hover */
    .stButton > button {
        background-color: var(--surface);
        color: var(--ink) !important;
        border: 1px solid var(--line-strong);
        border-radius: 22px;
        padding: 7px 16px;
        font-weight: 500;
        box-shadow: 0 1px 2px rgba(91,70,54,0.06);
        transition: all .15s ease;
    }
    .stButton > button:hover {
        background-color: #ECDCC4;
        color: var(--ink) !important;
        border-color: var(--accent);
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(91,70,54,0.15);
    }
    .stButton > button:active { transform: translateY(0); }
    .stButton > button:hover * { color: var(--ink) !important; }

    /* Article cards — soft hover lift */
    .article-card {
        background-color: var(--surface);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 8px;
        color: var(--ink) !important;
        font-size: 0.9rem;
        box-shadow: 0 1px 2px rgba(91,70,54,0.05);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .article-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(91,70,54,0.12);
    }

    /* File uploader dropzone — match the warm theme */
    [data-testid="stFileUploaderDropzone"] {
        background-color: var(--surface);
        border: 1.5px dashed var(--line-strong);
        border-radius: 14px;
    }

    /* Expanders — rounded, themed */
    [data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        background-color: var(--surface);
        box-shadow: 0 1px 2px rgba(91,70,54,0.05);
    }

    /* Divider a touch warmer */
    hr { border-color: var(--line) !important; }

    /* ── Loading state: dim + lock the controls so they read as "inaccessible" ── */
    .stButton > button:disabled,
    div[data-testid="stForm"] button:disabled,
    div[data-testid="stPopover"] > button:disabled,
    [data-testid="stDownloadButton"] button:disabled {
        opacity: 0.45 !important;
        filter: grayscale(35%);
        cursor: not-allowed !important;
        box-shadow: none !important;
        transform: none !important;
    }
    /* Dim the disabled text input wrapper too (it's the chat box while loading) */
    div[data-baseweb="input"]:has(input:disabled),
    div[data-baseweb="base-input"]:has(input:disabled) {
        opacity: 0.45 !important;
        cursor: not-allowed !important;
        box-shadow: none !important;
        border-color: var(--line-strong) !important;
    }
    input:disabled { cursor: not-allowed !important; }

    .stChatInput textarea { background-color: var(--surface); }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "url_sources" not in st.session_state:
    st.session_state.url_sources = []   # list[URLSource] added via the URL box

#  LEFT: sidebar = document upload + URL articles
with st.sidebar:
    st.header("📁 Documents")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "doc", "docx", "html", "htm", "csv", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    #  Add an article by URL
    st.markdown("**🔗 Add an article by URL**")
    with st.form("url_form", clear_on_submit=True, border=False):
        url_in = st.text_input(
            "Article URL", key="url_input", label_visibility="collapsed",
            placeholder="https://example.com/article  (web page or PDF link)",
        )
        add_url = st.form_submit_button("➕ Add URL", use_container_width=True)

    if add_url and url_in and url_in.strip():
        new_url = url_in.strip()
        if any(s.url == new_url for s in st.session_state.url_sources):
            st.info("That URL is already added.")
        else:
            try:
                with st.spinner("🌐 Fetching article…"):
                    src = fetch_url_article(new_url)
                # Keep every source name unique (so citations & metadata don't collide).
                taken = {f.name for f in (uploaded_files or [])} | {s.name for s in st.session_state.url_sources}
                src.name = _unique_name(src.name, taken)
                st.session_state.url_sources.append(src)
                st.success(f"Added: {src.title[:60]}")
                st.rerun()
            except Exception:
                # Keep it friendly — no raw error text on screen; log it for debugging.
                log.exception("URL fetch failed for: %r", new_url)
                st.info("Couldn't read that link. Please check the URL and try again.")

    # Show / remove the URLs added so far.
    if st.session_state.url_sources:
        st.caption("URLs added:")
        for i, s in enumerate(st.session_state.url_sources):
            c1, c2 = st.columns([0.85, 0.15])
            c1.markdown(f"🔗 **{s.title[:38]}**<br>"
                        f"<span style='font-size:0.72rem;color:#6b5b4a'>{s.url[:46]}</span>",
                        unsafe_allow_html=True)
            if c2.button("✕", key=f"rm_url_{i}", help="Remove this URL"):
                st.session_state.url_sources.pop(i)
                st.rerun()

    if uploaded_files or st.session_state.url_sources:
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


    qa_pairs = []   # (question, assistant_msg)
    pending_q = None
    for m in st.session_state.messages:
        if m["role"] == "user":
            pending_q = m["content"]
        elif m["role"] == "assistant":
            qa_pairs.append((pending_q or "(question)", m))
            pending_q = None

    # Only show enquiries that actually carry RAG details (sources or debug).
    rag_pairs = [(q, m) for (q, m) in qa_pairs if m.get("sources") or m.get("debug")]

    if rag_pairs:
        st.divider()
        with st.expander(f"🔬 RAG internals — {len(rag_pairs)} enquiry(s)", expanded=False):
            for idx, (question, m) in enumerate(rag_pairs, 1):
                st.markdown(f"**Enquiry {idx}:** {question[:120]}"
                            + ("…" if len(question) > 120 else ""))

                debug = m.get("debug", {})
                q_type = m.get("q_type") or debug.get("q_type", "general")
                st.caption(f"Question type: `{q_type}`  ·  "
                           f"prompt template: `{debug.get('prompt_template', q_type)}`")

                if m.get("sources"):
                    st.markdown("_Sources:_")
                    for s in m["sources"]:
                        st.markdown(f"&nbsp;&nbsp;• {s}", unsafe_allow_html=True)

                chunks = debug.get("chunks", [])
                if chunks:
                    st.markdown(f"_Retrieved chunks (top-{len(chunks)}):_")
                    for i, (text, score, page) in enumerate(chunks, 1):
                        page_str = "n/a" if page is None else page + 1
                        st.markdown(f"`Chunk {i}` · relevance: `{score}` · page: `{page_str}`")
                        st.caption(text[:300] + ("…" if len(text) > 300 else ""))
                elif not m.get("sources"):
                    st.caption("No retrieval (answer computed directly).")

                if idx < len(rag_pairs):
                    st.divider()


all_sources = (list(uploaded_files) if uploaded_files else []) + list(st.session_state.url_sources)
article_names = [f.name for f in all_sources]
st.title("🔬 Research Assistant")

if not all_sources:
    st.info("👈 Upload your research articles **or paste an article URL** from the left to begin.")
    st.stop()

signatures = tuple((f.name, f.size) for f in all_sources)

# 1)  read just the first page of each PDF for the title (URL articles already

extracted_titles = extract_titles_fast(signatures, all_sources)


with st.sidebar.expander("✏️ Edit article titles", expanded=False):
    article_titles = []
    for i, name in enumerate(article_names):
        edited = st.text_input(f"[{i+1}] {name}", value=extracted_titles[i], key=f"title::{name}")
        article_titles.append(edited.strip() or extracted_titles[i])

numbered_articles = [f"[{i+1}] {t}  —  {name}"
                     for i, (t, name) in enumerate(zip(article_titles, article_names))]
is_single = len(article_names) == 1

# Title strip listing every uploaded article
st.markdown("**🗂️ Articles in this session:**"
            + ("" if is_single else "  _(answers cite them as [1], [2], …)_"))
acols = st.columns(min(len(article_names), 3))
for i, (title, name) in enumerate(zip(article_titles, article_names)):
    acols[i % len(acols)].markdown(
        f"<div class='article-card'><b>[{i+1}]</b> {title}"
        f"<br><span style='font-size:0.75rem;color:#6b5b4a'>{name}</span></div>",
        unsafe_allow_html=True,
    )

st.divider()

vectorstore, doc_texts, parse_errors = process_documents(signatures, all_sources)
if not vectorstore:
    # Friendly, non-alarming guidance instead of a red error box.
    log.warning("No documents indexed for sources: %s", article_names)
    st.info("I couldn't read text from those sources yet. "
            "Try re-uploading the files or adding a different URL.")
    if parse_errors:
        with st.expander("Why did this fail?"):
            for e in parse_errors:
                st.code(e)
    st.stop()
app = build_graph(tuple(article_names), tuple(article_titles), vectorstore, doc_texts)


pending = None

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("model_used"):
            st.caption(f"answered by {msg['model_used']}")


if "selected_model" not in st.session_state:
    st.session_state.selected_model = AUTO_LABEL
MODEL_OPTIONS = [AUTO_LABEL] + [m["label"] for m in MODEL_QUEUE]
session_title = "Research Assistant — " + ", ".join(article_names[:3]) + (" …" if len(article_names) > 3 else "")

def render_controls(disabled: bool):
    sfx = "_loading" if disabled else ""
    picked = None

    #  Suggested actions
    st.markdown("**✨ Suggested actions** — "
                + ("ask anything about your paper:" if is_single
                   else "explore across all your papers:"))
    suggestions = SUGGESTIONS_SINGLE if is_single else SUGGESTIONS_MULTI
    chip_cols = st.columns(3)
    for i, (label, prompt_text) in enumerate(suggestions):
        if chip_cols[i % 3].button(label, key=f"sug_{i}{sfx}",
                                   use_container_width=True, disabled=disabled):
            picked = prompt_text

    #  Input bar
    bar_plus, bar_box = st.columns([0.07, 0.93], gap="small")
    with bar_plus:

        with st.popover("➕", use_container_width=True, disabled=disabled):
            st.markdown("**Answer with model**")
            if disabled:
                st.radio("model", MODEL_OPTIONS,
                         index=MODEL_OPTIONS.index(st.session_state.selected_model),
                         key="selected_model_loading", label_visibility="collapsed",
                         disabled=True)
            else:
                st.radio("model", MODEL_OPTIONS, key="selected_model",
                         label_visibility="collapsed")
    with bar_box:
        with st.form(f"ask_form{sfx}", clear_on_submit=True, border=False):
            t_col, s_col = st.columns([0.92, 0.08], gap="small")
            typed = t_col.text_input(
                "question", key=f"ask_text{sfx}", label_visibility="collapsed",
                placeholder=f"Ask a question…   ·   model: {st.session_state.selected_model}",
                disabled=disabled,
            )
            submitted = s_col.form_submit_button("➤", use_container_width=True,
                                                 disabled=disabled)
    if submitted and typed and typed.strip():
        picked = typed

    #  Download buttons
    if st.session_state.messages:
        st.divider()
        st.markdown("**⬇️ Download the full conversation:**")
        d1, d2, _ = st.columns([1, 1, 4])
        d1.download_button(
            "JSON",
            data=conversation_to_json(session_title, numbered_articles, st.session_state.messages),
            file_name=f"research_conversation_{datetime.now():%Y%m%d_%H%M}.json",
            mime="application/json", use_container_width=True,
            key=f"dl_json{sfx}", disabled=disabled,
        )


        try:
            pdf_bytes = conversation_to_pdf(session_title, numbered_articles, st.session_state.messages)
            d2.download_button(
                "PDF", data=pdf_bytes,
                file_name=f"research_conversation_{datetime.now():%Y%m%d_%H%M}.pdf",
                mime="application/pdf", use_container_width=True,
                key=f"dl_pdf{sfx}", disabled=disabled,
            )
        except Exception:
            log.exception("PDF export failed (reportlab missing or render error)")

    return picked


controls = st.empty()
with controls.container():
    pending = render_controls(disabled=False)

#  Run whichever question is pending
if pending:

    with controls.container():
        render_controls(disabled=True)
    with st.spinner("🔎 Searching across your documents..."):
        run_query(app, pending, st.session_state.selected_model)
    st.rerun()
