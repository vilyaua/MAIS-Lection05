# Development Log

## 2026-03-21

### 21:30 — L5 implementation: Research Agent with RAG
- Scaffolded `research-agent/` from L4 (custom ReAct loop preserved)
- Created `ingest.py` — PDF ingestion pipeline (PyPDFLoader → chunking → FAISS + BM25)
- Created `retriever.py` — hybrid retrieval (semantic + BM25) with cross-encoder reranking
- Added `knowledge_search` tool to `tools.py` (6th tool, lazy-loaded retriever)
- Updated `config.py` — RAG settings + system prompt with "check KB first" strategy
- Updated `main.py` + `app.py` — knowledge_search display formatting (orange in web UI)
- Extended `requirements.txt` with langchain RAG stack, faiss-cpu, sentence-transformers
- Updated `docker-compose.yml` with data/ and index/ volume mounts
- Created `ARCHITECTURE.md` with component diagram
- Copied `pyproject.toml` + `.pre-commit-config.yaml` from L4

### Project scaffolding
- Created repo with `homework-lesson-5/` skeleton (stubs for agent.py, tools.py, config.py, main.py, ingest.py, retriever.py)
- Test data: 3 PDFs in `homework-lesson-5/data/` (langchain, LLM, RAG)
- Created `CLAUDE.md` — implementation guide for RAG extension of L4's custom ReAct agent
- Created `README.md` — project overview with quick start
- Added MIT `LICENSE`
