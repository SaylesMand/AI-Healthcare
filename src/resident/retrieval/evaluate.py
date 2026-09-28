from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from resident.config import EVAL_DIR
from resident.retrieval.base import Retriever

GOLD_PATH = EVAL_DIR / "retrieval_gold.jsonl"


def load_gold(path=GOLD_PATH) -> list[dict]:
    """Загружает эталонный набор запросов для оценки поиска."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def query_metrics(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    """Считает Recall@5, Recall@10, MRR@10 и nDCG@5 для одного запроса."""
    hits = [cid in relevant for cid in ranked]
    first = next((i for i, h in enumerate(hits) if h), None)
    dcg = sum(1 / np.log2(i + 2) for i, h in enumerate(hits[:5]) if h)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), 5)))
    return {
        "Recall@5": sum(hits[:5]) / len(relevant),
        "Recall@10": sum(hits[:10]) / len(relevant),
        "MRR@10": 1 / (first + 1) if first is not None and first < 10 else 0.0,
        "nDCG@5": dcg / idcg,
    }


def evaluate(retriever: Retriever, gold: list[dict] | None = None) -> tuple[dict, pd.DataFrame]:
    """Считает средние метрики поисковика и метрики по каждому запросу."""
    gold = gold or load_gold()
    rows = []
    for g in gold:
        ranked = [h.chunk.chunk_id for h in retriever.search(g["query"], k=10)]
        rows.append({"query": g["query"], **query_metrics(ranked, set(g["relevant"]))})
    per_query = pd.DataFrame(rows)
    summary = per_query.drop(columns="query").mean().round(3).to_dict()
    return summary, per_query


def compare(kinds: list[str]) -> pd.DataFrame:
    """Сравнивает метрики нескольких вариантов поиска."""
    from resident.retrieval import build_retriever

    results = {}
    for kind in kinds:
        summary, _ = evaluate(build_retriever(kind))
        results[kind] = summary
    return pd.DataFrame(results).T


if __name__ == "__main__":
    kinds = sys.argv[1:] or ["bm25", "dense", "hybrid"]
    print(compare(kinds).to_string())
