from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from resident.schemas import Chunk, SearchHit


def section_mask(chunks: list[Chunk], sections: list[str] | None) -> np.ndarray:
    """Возвращает маску чанков из указанных разделов КР и их подразделов."""
    if not sections:
        return np.ones(len(chunks), dtype=bool)
    return np.array([
        any(c.section_num == s or c.section_num.startswith(s + ".") for s in sections)
        for c in chunks
    ])


def doc_text(chunk: Chunk) -> str:
    """Возвращает индексируемый текст чанка: путь заголовков, тезис и комментарий."""
    return " / ".join(chunk.section_path) + "\n" + chunk.full_text


class Retriever(ABC):
    name: str

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    @abstractmethod
    def scores(self, query: str) -> np.ndarray:
        """Возвращает оценку релевантности каждого чанка запросу."""

    def search(self, query: str, k: int = 8, sections: list[str] | None = None) -> list[SearchHit]:
        """Возвращает top-k чанков по запросу с фильтром по разделам."""
        scores = self.scores(query).astype(float)
        scores[~section_mask(self.chunks, sections)] = -np.inf
        order = np.argsort(-scores)[:k]
        return [
            SearchHit(chunk=self.chunks[i], score=float(scores[i]), rank=r + 1)
            for r, i in enumerate(order)
            if np.isfinite(scores[i])
        ]
