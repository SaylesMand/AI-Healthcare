from __future__ import annotations

import json
import re
from functools import lru_cache

import numpy as np
import pymorphy3
from rank_bm25 import BM25Okapi

from resident.config import ABBR_PATH
from resident.retrieval.base import Retriever, doc_text
from resident.schemas import Chunk

_morph = pymorphy3.MorphAnalyzer()
_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.I)
STOPWORDS = set(
    "и в во не на с со по к ко о об от до для из у за при а но или ли же бы то это как так что чем "
    "его ее её их он она оно они мы вы я который которая которые также если может быть был была было "
    "были есть нет более менее все всех всем этот эта эти этом этой том тот те этих под над между "
    "через после перед без т е др т.е. и/или".split()
)


@lru_cache(maxsize=100_000)
def lemma(token: str) -> str:
    """Возвращает нормальную форму слова через pymorphy3."""
    return _morph.parse(token)[0].normal_form


# Сокращения, которых нет в списке КР, но которые встречаются в запросах врачей.
EXTRA_ABBREVIATIONS = {
    "ИПП": "ингибиторы протонного насоса",
    "ДАТТ": "двойная антитромбоцитарная терапия",
    "НМГ": "низкомолекулярный гепарин эноксапарин",
    "ОПП": "острое повреждение почек",
    "КИН": "контраст-индуцированная нефропатия",
    "СКФ": "скорость клубочковой фильтрации",
}


def load_abbreviations() -> dict[str, str]:
    """Загружает сокращения КР и дополнительные сокращения врачей."""
    abbr = json.loads(ABBR_PATH.read_text(encoding="utf-8")) if ABBR_PATH.exists() else {}
    return {**EXTRA_ABBREVIATIONS, **abbr}


class Tokenizer:
    def __init__(self, abbreviations: dict[str, str] | None = None):
        abbr = abbreviations if abbreviations is not None else load_abbreviations()
        # раскрываем только "чистые" аббревиатуры: ДАТТ, рСКФ, НФГ
        self.abbr = {k.lower(): v for k, v in abbr.items() if re.fullmatch(r"[А-ЯЁA-Zа-яё0-9\-]{2,10}", k)}

    def __call__(self, text: str) -> list[str]:
        """Разбивает текст на леммы и добавляет расшифровки сокращений."""
        tokens = []
        for raw in _TOKEN_RE.findall(text.lower()):
            if raw in STOPWORDS:
                continue
            tokens.append(lemma(raw))
            if raw in self.abbr:
                tokens += [lemma(t) for t in _TOKEN_RE.findall(self.abbr[raw].lower()) if t not in STOPWORDS]
        return tokens


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, chunks: list[Chunk], tokenizer: Tokenizer | None = None):
        super().__init__(chunks)
        self.tokenize = tokenizer or Tokenizer()
        self.bm25 = BM25Okapi([self.tokenize(doc_text(c)) for c in chunks])

    def scores(self, query: str) -> np.ndarray:
        """Возвращает оценки BM25, для чанков без общих терминов -inf."""
        scores = np.asarray(self.bm25.get_scores(self.tokenize(query)), dtype=float)
        scores[scores <= 0] = -np.inf  # ни одного общего термина - не кандидат
        return scores
