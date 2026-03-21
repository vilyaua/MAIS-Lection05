# Architecture — Research Agent with RAG

## Overview

Extends the L4 Research Agent (custom ReAct loop, OpenAI SDK) with a RAG tool that performs hybrid search (semantic FAISS + BM25) with cross-encoder reranking over local PDF documents.

## Components

```
┌─────────────────────────────────────────────────────────┐
│                      User Interface                      │
│  main.py (CLI REPL)  │  app.py (FastAPI + SSE Web UI)   │
└─────────────┬────────────────────┬──────────────────────┘
              │                    │
              ▼                    ▼
┌─────────────────────────────────────────────────────────┐
│                    agent.py                              │
│              Custom ReAct Loop                           │
│  Reason → Act (call tool) → Observe → repeat            │
│  (Tool-agnostic: dispatches via TOOL_FUNCTIONS dict)     │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    tools.py                              │
│  knowledge_search │ web_search │ read_url │ write_report │
│  list_reports     │ read_file  │                         │
└───────┬───────────────┬────────────────┬────────────────┘
        │               │                │
        ▼               ▼                ▼
┌──────────────┐ ┌─────────────┐  ┌──────────────┐
│ retriever.py │ │  DuckDuckGo │  │  trafilatura  │
│              │ │    (ddgs)   │  │  (URL reader) │
│ FAISS (sem.) │ └─────────────┘  └──────────────┘
│ BM25 (lex.)  │
│ CrossEncoder │
│  (reranker)  │
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐
│   index/     │ ◄── │  ingest.py   │
│ FAISS index  │     │ PDF → chunks │
│ BM25 chunks  │     │  → embeddings│
└──────────────┘     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │    data/     │
                     │  *.pdf files │
                     └──────────────┘
```

## Data Flow

### Ingestion (offline, run once)
```
data/*.pdf → PyPDFLoader → pages → RecursiveCharacterTextSplitter (500/100)
           → chunks → OpenAIEmbeddings (text-embedding-3-small) → FAISS index
                    → pickle dump → BM25 chunks
```

### Retrieval (at query time)
```
query → EnsembleRetriever ──┬── FAISS semantic search (top_k=10)
                            └── BM25 lexical search (top_k=10)
      → merge candidates
      → CrossEncoder reranking (BAAI/bge-reranker-base)
      → top_n=3 results with source metadata
```

### Agent Loop (unchanged from L4)
```
User message → [system prompt + history] → OpenAI API
            → tool_calls? → execute tools → append results → repeat
            → no tool_calls? → return final answer
```

## Key Design Decisions

1. **LangChain for RAG only** — The agent loop is custom (no create_react_agent, no LangGraph). LangChain is used only for document loading, text splitting, embeddings, FAISS, and BM25.

2. **Lazy initialization** — The retriever and cross-encoder model are loaded once on first `knowledge_search` call and cached at module level. This avoids slow startup when the RAG tool isn't needed.

3. **Hybrid search** — Combines semantic (embedding-based) and lexical (BM25) search with equal weights (0.5/0.5) to capture both meaning and exact terms.

4. **Cross-encoder reranking** — A separate reranking step with BAAI/bge-reranker-base to improve precision from the initial candidate set.

5. **Tool-agnostic agent** — Adding `knowledge_search` required zero changes to `agent.py`. Only `tools.py`, `config.py`, and display formatting needed updates.

## Configuration

All settings in `config.py` via pydantic-settings (`.env` file):

| Setting | Default | Description |
|---------|---------|-------------|
| `embedding_model` | `text-embedding-3-small` | OpenAI embedding model |
| `data_dir` | `data` | Source documents directory |
| `index_dir` | `index` | Persisted index directory |
| `chunk_size` | `500` | Text chunk size (chars) |
| `chunk_overlap` | `100` | Overlap between chunks |
| `retrieval_top_k` | `10` | Candidates per retriever |
| `rerank_top_n` | `3` | Final results after reranking |
