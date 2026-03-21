# CLAUDE.md — Research Agent with RAG (Lesson 5)

## Project Context

This is **homework-lesson-5** of the MULTI-AGENT-SYSTEMS course.
The goal: extend the Research Agent from `MAIS-Lection04/research-agent/` with a **RAG tool** — hybrid search (semantic + BM25) with cross-encoder reranking over a local knowledge base.

Reference implementation: `../MAIS-Lection04/research-agent/` (v0.1.0, custom ReAct loop)
Task spec: `homework-lesson-5/README.md`

## What Must Change (Lesson-4 → Lesson-5)

| Lesson-4 (current)                          | Lesson-5 (target)                                    |
|----------------------------------------------|------------------------------------------------------|
| Tools: `web_search`, `read_url`, `write_report`, `list_reports`, `read_file` | + new tool: `knowledge_search`          |
| Agent searches only the web                  | Agent searches web AND local knowledge base          |
| No document ingestion                        | `ingest.py` pipeline: PDF → chunks → embeddings → FAISS |
| No retrieval module                          | `retriever.py`: hybrid search (semantic + BM25) + reranking |
| No vector DB                                 | FAISS index persisted to `index/` directory          |
| No embeddings                                | `text-embedding-3-small` via OpenAI                  |

## Architecture (Target)

```
research-agent/
├── config.py          # Settings + SYSTEM_PROMPT (updated with knowledge_search)
├── tools.py           # 6 tools: existing 5 + knowledge_search
├── agent.py           # Custom ReAct loop (unchanged from L4)
├── retriever.py       # NEW: hybrid retrieval (semantic + BM25) + cross-encoder reranking
├── ingest.py          # NEW: document ingestion pipeline
├── main.py            # Console REPL (unchanged from L4)
├── app.py             # FastAPI web UI (unchanged from L4)
├── requirements.txt   # Extended with RAG dependencies
├── Dockerfile
├── docker-compose.yml
├── VERSION
├── data/              # NEW: source documents (PDF/TXT) for ingestion
├── index/             # NEW: persisted FAISS index + BM25 chunks (gitignored)
├── output/            # Generated reports
└── logs/              # Agent logs
```

## Implementation Plan

### 1. Dependencies (`requirements.txt`)

Keep from L4: `openai`, `ddgs`, `trafilatura`, `pydantic-settings`, `fastapi`, `uvicorn`

Add for RAG pipeline:
- `langchain` — text splitting, document abstractions
- `langchain-core` — base document types
- `langchain-openai` — OpenAI embeddings wrapper
- `langchain-community` — FAISS vector store wrapper
- `faiss-cpu` — vector similarity search
- `rank_bm25` — BM25 lexical search
- `sentence-transformers` — cross-encoder reranking model
- `pypdf` — PDF document loading

**Note:** LangChain is used ONLY for RAG components (embeddings, text splitting, FAISS, document loading). The agent loop stays custom (no `create_react_agent`, no LangGraph).

### 2. Knowledge Ingestion Pipeline (`ingest.py`)

Standalone script that processes documents and builds the search index:

```python
"""
Usage: python ingest.py

1. Load PDFs from data/ directory (PyPDFLoader)
2. Split into chunks (RecursiveCharacterTextSplitter)
   - chunk_size=500, chunk_overlap=100
3. Generate embeddings (text-embedding-3-small via OpenAI)
4. Build FAISS vector store
5. Save FAISS index to index/ directory
6. Save raw chunks for BM25 retriever (pickle)
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def ingest():
    # Load all PDFs from data/
    docs = []
    for pdf_file in Path(settings.data_dir).glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs.extend(loader.load())

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    # Build FAISS index with OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.api_key.get_secret_value(),
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(settings.index_dir)

    # Save chunks for BM25 (pickle)
    save_chunks_for_bm25(chunks, settings.index_dir)
```

Key points:
- Run once: `python ingest.py` — creates `index/` directory
- Rerunning overwrites the index (no incremental updates)
- Index persists to disk — no need to re-embed on every startup

### 3. Hybrid Retrieval + Reranking (`retriever.py`)

The core RAG module:

```python
"""
Hybrid retrieval: semantic (FAISS) + lexical (BM25) + cross-encoder reranking.

Pipeline:
  query → [semantic search] → top_k candidates
  query → [BM25 search]     → top_k candidates
        → [merge & deduplicate]
        → [cross-encoder reranker] → top_n final results
"""

def get_retriever():
    # 1. Load FAISS index from disk
    embeddings = OpenAIEmbeddings(model=settings.embedding_model, ...)
    vectorstore = FAISS.load_local(settings.index_dir, embeddings)

    # 2. Create semantic retriever
    semantic_retriever = vectorstore.as_retriever(
        search_kwargs={"k": settings.retrieval_top_k}
    )

    # 3. Load chunks and create BM25 retriever
    chunks = load_chunks_for_bm25(settings.index_dir)
    bm25_retriever = BM25Retriever.from_documents(chunks, k=settings.retrieval_top_k)

    # 4. Combine into ensemble (e.g., EnsembleRetriever or manual merge)
    ensemble = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

    # 5. Add cross-encoder reranker
    # e.g., BAAI/bge-reranker-base or cross-encoder/ms-marco-MiniLM-L-6-v2
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base", top_n=settings.rerank_top_n)

    return ensemble, reranker


def retrieve(query: str) -> list[dict]:
    """Run hybrid retrieval + reranking. Returns top_n documents."""
    ensemble, reranker = get_retriever()
    candidates = ensemble.invoke(query)
    reranked = reranker.rerank(query, candidates)
    return reranked
```

### 4. New Tool: `knowledge_search` (`tools.py`)

Add to existing tools:

```python
def knowledge_search(query: str) -> str:
    """Search the local knowledge base using hybrid retrieval + reranking."""
    results = retrieve(query)  # from retriever.py
    if not results:
        return "No relevant documents found in the knowledge base."
    formatted = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"{i}. [Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

KNOWLEDGE_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "knowledge_search",
        "description": (
            "Search the local knowledge base of ingested documents. "
            "Use for questions about topics covered in the uploaded PDFs. "
            "Returns relevant text chunks with source and page references."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for the knowledge base.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
```

Add to `TOOL_FUNCTIONS` and `TOOL_SCHEMAS` registries.

### 5. Update System Prompt (`config.py`)

Add `knowledge_search` to the `<tools>` section:

```
6. **knowledge_search(query)** — Search the local knowledge base of ingested documents.
   Returns relevant text chunks with source file and page references.
   Use this FIRST for questions about topics that might be covered in uploaded documents
   (RAG, LLMs, LangChain, etc.), then supplement with web_search if needed.
```

Update `<strategy>` to include:
```
1. **Check knowledge base first** — if the topic might be in the ingested documents,
   start with knowledge_search before going to the web.
2. **Combine sources** — use both knowledge base and web results for comprehensive reports.
```

### 6. Config Updates (`config.py`)

Add RAG-specific settings to `Settings`:

```python
# RAG settings
embedding_model: str = "text-embedding-3-small"
data_dir: str = "data"
index_dir: str = "index"
chunk_size: int = 500
chunk_overlap: int = 100
retrieval_top_k: int = 10  # candidates before reranking
rerank_top_n: int = 3      # final results after reranking
```

### 7. Test Data (`data/`)

The homework skeleton provides 3 PDFs:
- `langchain.pdf`
- `large-language-model.pdf`
- `retrieval-augmented-generation.pdf`

Copy these to `research-agent/data/`.

### 8. Agent Loop (`agent.py`) — NO CHANGES

The custom ReAct loop from L4 is tool-agnostic. Adding `knowledge_search` to `TOOL_SCHEMAS` and `TOOL_FUNCTIONS` is sufficient — no agent.py changes needed.

### 9. Web UI (`app.py`) — Minor Updates

Add display formatting for `knowledge_search` tool calls (color, status text) in both HTML and `_format_tool_event()`.

### 10. Logging

Add `knowledge_search` to `_format_tool_status()` in `main.py` and `_format_tool_event()` in `app.py`:
```python
if name == "knowledge_search":
    count = result.count("---")
    return f"  [knowledge_search]{args_part}— {count + 1} documents found"
```

---

## Files to Copy from Lesson-4

- `agent.py` — custom ReAct loop (no changes needed)
- `main.py` — console REPL (minor: add knowledge_search display)
- `app.py` — FastAPI web UI (minor: add knowledge_search display)
- `Dockerfile` (adjust for new deps if needed)
- `docker-compose.yml` (add data/ and index/ volume mounts)
- `.env.example`
- `.gitignore` (add `index/`)
- `.dockerignore`
- `VERSION` (reset to `0.1.0`)
- `example_output/report.md`
- `ARCHITECTURE.md` (update with RAG components)

## Files to Create New

- `ingest.py` — document ingestion pipeline
- `retriever.py` — hybrid retrieval + reranking
- `data/` — copy PDFs from homework skeleton

## Files to Modify

- `config.py` — add RAG settings + update system prompt with knowledge_search
- `tools.py` — add `knowledge_search` function + schema + register in dispatch dict
- `requirements.txt` — add RAG dependencies
- `main.py` — add knowledge_search display formatting
- `app.py` — add knowledge_search display formatting

## Conventions

- Python 3.12+
- Ruff for linting/formatting (line-length=100)
- Pre-commit hooks (copy from lesson-4 root)
- DEVLOG.md — update after every significant change
- VERSION file — single source of truth, start at `0.1.0`
- LangChain used ONLY for RAG pipeline (embeddings, splitting, FAISS, loaders) — NOT for agent loop

## Acceptance Criteria

1. `python ingest.py` loads PDFs from `data/`, creates FAISS index in `index/`
2. Index persists to disk and loads without re-embedding
3. Hybrid search combines semantic (FAISS) + lexical (BM25)
4. Cross-encoder reranker filters top_n results
5. `knowledge_search` tool is available to the agent via JSON Schema
6. Agent autonomously decides when to use `knowledge_search` vs `web_search`
7. Agent combines results from both sources in reports
8. `python main.py` starts REPL — agent can answer questions about ingested docs
9. Custom ReAct loop (no `create_react_agent`, no LangGraph)
10. Reports saved to `output/` with source citations
11. FastAPI web UI works with knowledge_search events displayed
12. Docker build works (with data/ and index/ mounted)

## Expected Agent Behavior

```
You: Що таке RAG і які є підходи до retrieval?

  [knowledge_search]("RAG retrieval approaches") — 3 documents found
  [web_search]("RAG retrieval techniques 2026") — 5 results found
  [read_url]("https://example.com/advanced-rag") — extracted 5,000 chars
  [write_report]("rag_approaches") — Report saved to output/2026-03-21_1430_rag_approaches.md

Agent: RAG — це техніка, де...
```

The agent checks the knowledge base first for local expertise, supplements with web search for latest information, and combines both into a comprehensive report.
