from __future__ import annotations

import numpy as np

from resident.retrieval.base import Retriever


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(self, retrievers: list[Retriever], k_rrf: int = 60, depth: int = 50):
        super().__init__(retrievers[0].chunks)
        self.retrievers = retrievers
        self.k_rrf = k_rrf
        self.depth = depth

    def scores(self, query: str) -> np.ndarray:
        """Объединяет ранги поисковиков через Reciprocal Rank Fusion."""
        fused = np.zeros(len(self.chunks))
        for r in self.retrievers:
            scores = r.scores(query)
            order = [i for i in np.argsort(-scores)[: self.depth] if np.isfinite(scores[i])]
            fused[order] += 1.0 / (self.k_rrf + np.arange(1, len(order) + 1))
        return fused
