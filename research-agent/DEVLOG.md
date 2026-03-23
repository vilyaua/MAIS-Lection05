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

## 2026-03-23

### 13:00 — Add truncation to web_search and knowledge_search (`feat/web-search-truncation`)

Addressed teacher feedback: "варто додати truncation у web_search, без нього може непередбачувано забиватись контекст"

- `config.py`: added `max_search_content_length: int = 3000` — separate limit from `max_url_content_length` (8000), since search results and full page content have different optimal sizes
- `config.py`: added `app_name: str = "Research Agent L05 (RAG)"` for `/api/info` identification
- `tools.py`: added truncation to `web_search` return value using `max_search_content_length`
- `tools.py`: added truncation to `knowledge_search` return value using same limit — consistent pattern across all search tools
- `app.py`: added `app` field to `/api/info` endpoint

### 13:50 — Test run: 10 queries with truncation

- Ran all 10 test queries against L05 with RAG
- All queries succeeded, total: 306,355 tokens in 475s
- L05 used 15% more tokens than L04 but was 13% faster (knowledge_search is instant vs network latency)
- Biggest win: RAG comparison query used 21,790 tokens (vs 65,812 in L04) thanks to knowledge base
- Biggest concern: Vélez-Málaga vs Motril (51,950 tokens) — agent called knowledge_search on irrelevant topic, accumulating context without saving a report
