from __future__ import annotations

from resident.config import settings
from resident.kb.parse import load_chunks
from resident.retrieval.base import Retriever
from resident.retrieval.bm25 import BM25Retriever
from resident.schemas import Chunk


def build_retriever(kind: str | None = None, chunks: list[Chunk] | None = None) -> Retriever:
    """Создает поисковик bm25, dense или hybrid, по умолчанию из .env."""
    kind = kind or settings.retriever
    chunks = chunks or load_chunks()
    if kind == "bm25":
        return BM25Retriever(chunks)
    from resident.retrieval.dense import DenseRetriever
    from resident.retrieval.hybrid import HybridRetriever

    dense = DenseRetriever(chunks)
    if kind == "dense":
        return dense
    if kind == "hybrid":
        return HybridRetriever([BM25Retriever(chunks), dense])
    raise ValueError(f"unknown retriever: {kind}")


__all__ = ["Retriever", "build_retriever"]
