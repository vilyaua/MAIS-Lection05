"""Hybrid retrieval: semantic (FAISS) + lexical (BM25) + cross-encoder reranking.

Pipeline:
    query → EnsembleRetriever(semantic + BM25, weights 0.5/0.5)
          → CrossEncoder reranking (BAAI/bge-reranker-base)
          → top_n results
"""

import logging
import pickle
from pathlib import Path

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import CrossEncoder

from config import Settings

logger = logging.getLogger("research_agent")

# Module-level cache — initialized once on first call
_retriever_cache: dict = {}


def _get_components():
    """Lazy-load and cache retriever components (FAISS, BM25, cross-encoder)."""
    if _retriever_cache:
        return _retriever_cache["ensemble"], _retriever_cache["reranker"]

    settings = Settings()
    index_dir = Path(settings.index_dir)

    # 1. Load FAISS index
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.api_key.get_secret_value(),
    )
    vectorstore = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)

    # 2. Semantic retriever
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retrieval_top_k})

    # 3. Load chunks and create BM25 retriever
    chunks_path = index_dir / "bm25_chunks.pkl"
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)  # noqa: S301
    bm25_retriever = BM25Retriever.from_documents(chunks, k=settings.retrieval_top_k)

    # 4. Ensemble retriever (semantic + BM25)
    ensemble = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

    # 5. Cross-encoder reranker
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    _retriever_cache["ensemble"] = ensemble
    _retriever_cache["reranker"] = reranker
    _retriever_cache["top_n"] = settings.rerank_top_n

    logger.info(
        "Retriever initialized: %d chunks, top_k=%d, top_n=%d",
        len(chunks),
        settings.retrieval_top_k,
        settings.rerank_top_n,
    )

    return ensemble, reranker


def retrieve(query: str) -> list:
    """Run hybrid retrieval + reranking. Returns top_n LangChain Documents."""
    ensemble, reranker = _get_components()
    top_n = _retriever_cache.get("top_n", 3)

    # Get candidates from ensemble retriever
    candidates = ensemble.invoke(query)
    if not candidates:
        return []

    # Rerank with cross-encoder
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    # Sort by score descending, take top_n
    scored_docs = sorted(zip(scores, candidates, strict=False), key=lambda x: x[0], reverse=True)
    results = [doc for _score, doc in scored_docs[:top_n]]

    logger.info("Retrieved %d candidates, reranked to %d results", len(candidates), len(results))
    return results
