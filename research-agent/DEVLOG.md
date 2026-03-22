# Development Log

## 2026-03-21

### 21:30 — Initial implementation: L4 → L5 with RAG

- Scaffolded from `MAIS-Lection04/research-agent/` (v0.1.0)
- Copied verbatim: `agent.py`, `VERSION`, `Dockerfile`, `.dockerignore`, `.env.example`
- Created `ingest.py` — document ingestion pipeline (PyPDFLoader → RecursiveCharacterTextSplitter → OpenAIEmbeddings → FAISS + BM25 pickle)
- Created `retriever.py` — hybrid retrieval (FAISS semantic + BM25 lexical, weights 0.5/0.5) with cross-encoder reranking (BAAI/bge-reranker-base)
- Modified `tools.py` — added `knowledge_search` tool (6th tool), lazy import of retriever
- Modified `config.py` — added RAG settings (embedding_model, data_dir, index_dir, chunk_size, chunk_overlap, retrieval_top_k, rerank_top_n), updated system prompt with knowledge_search tool and "check KB first" strategy
- Modified `main.py` — added knowledge_search display formatting
- Modified `app.py` — added knowledge_search CSS color (orange) and SSE event formatting
- Extended `requirements.txt` — added langchain, langchain-openai, langchain-community, langchain-text-splitters, faiss-cpu, rank_bm25, sentence-transformers, pypdf
- Updated `.gitignore` — added `index/`
- Updated `docker-compose.yml` — added `data/` and `index/` volume mounts
- Copied test data: 3 PDFs (langchain, LLM, RAG) from homework skeleton
- Created `ARCHITECTURE.md` with component diagram and data flow
