# MAIS-Lection05

## Homework: Research Agent with RAG

Extension of the Research Agent from Lesson 4 — adds a **RAG tool** with hybrid search (semantic + BM25) and cross-encoder reranking, so the agent can search both the web and a local knowledge base.

### What's New vs Lesson 4

| Lesson 4                                    | Lesson 5                                      |
|---------------------------------------------|-----------------------------------------------|
| Tools: web_search, read_url, write_report   | + new tool: `knowledge_search`                |
| Agent searches only the web                 | Agent searches web AND local knowledge base   |
| No document ingestion                       | `ingest.py` pipeline: PDF → chunks → FAISS    |
| No retrieval module                         | `retriever.py`: hybrid search + reranking     |

### Project Structure

- **`homework-lesson-5/`** — Original homework skeleton with task description and test PDFs
- **`research-agent/`** — Completed implementation
- **`CLAUDE.md`** — Implementation guide and specifications

### Quick Start

```bash
cd research-agent
pip install -r requirements.txt
cp .env.example .env   # add your API key

# 1. Ingest documents into the knowledge base
python ingest.py

# 2. Run the agent
python main.py
```

See [`research-agent/README.md`](research-agent/README.md) for full setup and architecture details.
