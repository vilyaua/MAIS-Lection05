# Research Agent with RAG

Custom ReAct agent (OpenAI SDK, no LangGraph) extended with a RAG tool — hybrid search (FAISS + BM25) with cross-encoder reranking over local PDF documents.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — set API_KEY (required)

# 3. Ingest documents into the knowledge base
python ingest.py

# 4. Run (pick one)
python main.py              # CLI
uvicorn app:app --reload    # Web UI at http://localhost:8000
```

## Project Structure

```
research-agent/
├── agent.py           # Custom ReAct loop (tool-agnostic, unchanged from L4)
├── config.py          # Settings (pydantic-settings) + system prompt
├── tools.py           # 6 tools: knowledge_search, web_search, read_url,
│                      #          write_report, list_reports, read_file
├── retriever.py       # Hybrid retrieval (FAISS + BM25) + cross-encoder reranking
├── ingest.py          # Document ingestion: PDF → chunks → embeddings → FAISS index
├── main.py            # Console REPL interface
├── app.py             # FastAPI web UI with SSE streaming
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── VERSION
├── ARCHITECTURE.md    # Detailed component diagram and data flow
├── data/              # Source PDFs for ingestion
│   ├── langchain.pdf
│   ├── large-language-model.pdf
│   └── retrieval-augmented-generation.pdf
├── index/             # Generated FAISS index + BM25 chunks (gitignored)
├── output/            # Generated research reports
└── logs/              # Agent logs (rotating, 5 MB)
```

## How It Works

1. **Ingestion** (`python ingest.py`) — loads PDFs, splits into chunks (500 chars, 100 overlap), generates embeddings via `text-embedding-3-small`, builds FAISS index, pickles chunks for BM25
2. **Retrieval** — on `knowledge_search` call, runs hybrid search (FAISS semantic + BM25 lexical, weights 0.5/0.5), then reranks with cross-encoder (`BAAI/bge-reranker-base`), returns top 3
3. **Agent loop** — custom ReAct: LLM decides which tool to call, executes it, feeds result back, repeats until final answer. The agent checks the knowledge base first, supplements with web search

## Tools

| Tool | Description |
|------|-------------|
| `knowledge_search(query)` | Search local knowledge base (hybrid FAISS + BM25 + reranking) |
| `web_search(query)` | Search the web via DuckDuckGo |
| `read_url(url)` | Extract text from a web page |
| `write_report(description, content)` | Save Markdown report to output/ |
| `list_reports()` | List saved reports |
| `read_file(filename)` | Read a saved report |

## Configuration

All settings via `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | *(required)* | OpenAI API key |
| `MODEL_NAME` | `gpt-4.1-mini` | LLM model |
| `BASE_URL` | *(none)* | OpenAI-compatible provider URL |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `500` | Text chunk size |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `10` | Candidates before reranking |
| `RERANK_TOP_N` | `3` | Final results after reranking |

## Docker

```bash
docker compose up --build
# Web UI at http://localhost:8000
# data/ and index/ are mounted as volumes
```

## What Changed from L4

- Added `ingest.py` — document ingestion pipeline
- Added `retriever.py` — hybrid retrieval + cross-encoder reranking
- Added `knowledge_search` tool to `tools.py`
- Updated `config.py` — RAG settings + "check KB first" strategy in system prompt
- Updated `main.py` + `app.py` — knowledge_search display formatting
- Extended `requirements.txt` — langchain RAG stack, faiss-cpu, sentence-transformers, pypdf
- `agent.py` — **zero changes** (tool-agnostic design)

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed component diagram and design decisions.
