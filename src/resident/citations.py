from __future__ import annotations

import re

from rapidfuzz import fuzz

from resident.drugs import classes_of
from resident.schemas import Chunk, Citation, PlanItemDraft

MIN_QUOTE_LEN = 15
FUZZY_THRESHOLD = 90
ALIGN_SET_RATIO = 85
ALIGN_RATIO = 40


def normalize(text: str) -> str:
    """Приводит текст к нижнему регистру и унифицирует кавычки, тире и пробелы для сравнения."""
    text = text.lower().replace("ё", "е").replace("**", "").replace("\xa0", " ")
    text = re.sub(r"[«»\"“”„'`]", "", text)
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    return re.sub(r"\s+", " ", text).strip(" .;,")


def chunk_haystack(chunk: Chunk) -> str:
    """Возвращает нормализованный текст тезиса и комментария чанка."""
    return normalize(chunk.full_text)


def strip_asides(text: str) -> str:
    """Удаляет из текста уточнения в скобках и ссылки на литературу."""
    prev = None
    while prev != text:  # вложенные скобки
        prev, text = text, re.sub(r"\([^()]*\)", " ", text)
    text = re.sub(r"\[[\d,\s\-–]*\]", " ", text)
    return re.sub(r"\s+", " ", text).strip(" .;,")


ELLIPSIS_RE = re.compile(r"\s*(?:\.\.\.|…)\s*")


def quote_in_chunk(quote: str, chunk: Chunk) -> bool:
    """Проверяет, что цитата есть в чанке дословно, допуская пропуск скобок и фрагментов через многоточие."""
    q, hay = normalize(quote), chunk_haystack(chunk)
    if len(q) < MIN_QUOTE_LEN:
        return False
    if q in hay:
        return True
    loose_hay = strip_asides(hay)
    parts = [strip_asides(p) for p in ELLIPSIS_RE.split(q) if p.strip(" .…")]
    if not parts or sum(map(len, parts)) < MIN_QUOTE_LEN:
        return False
    pos = 0
    for part in parts:
        idx = loose_hay.find(part, pos)
        if idx < 0:
            # опечатки источника / одиночные расхождения - только для длинных фрагментов
            if len(part) < 40 or fuzz.partial_ratio(part, loose_hay[pos:]) < FUZZY_THRESHOLD:
                return False
            continue
        pos = idx + len(part)
    return True


def sentences(text: str) -> list[str]:
    """Разбивает текст на предложения длиннее 20 символов."""
    return [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if len(s.strip()) > 20]


def align_quote(quote: str, chunk: Chunk) -> tuple[str, float] | None:
    """Находит в чанке предложение, близкое к пересказу LLM, и возвращает его с оценкой сходства."""
    q = strip_asides(normalize(quote))
    best = None
    for sent in sentences(chunk.full_text):
        s = strip_asides(normalize(sent))
        set_r, r = fuzz.token_set_ratio(q, s), fuzz.ratio(q, s)
        if set_r >= ALIGN_SET_RATIO and r >= ALIGN_RATIO and (best is None or set_r > best[1]):
            best = (sent, set_r)
    return best


def check_citations(citations: list[Citation], allowed: dict[str, Chunk]) -> tuple[list[Chunk], list[str], list[str]]:
    """Проверяет ссылки на чанки из контекста и заменяет пересказы дословным текстом КР."""
    ok, problems, info = [], [], []
    for c in citations:
        chunk = allowed.get(c.chunk_id)
        if chunk is None:
            problems.append(f"ссылка на {c.chunk_id}: чанк не передавался в контексте")
        elif quote_in_chunk(c.quote, chunk):
            ok.append(chunk)
        elif aligned := align_quote(c.quote, chunk):
            info.append(f"цитата {c.chunk_id} выровнена по тексту КР (сходство {aligned[1]:.0f}%)")
            c.quote = aligned[0]
            ok.append(chunk)
        else:
            problems.append(f"цитата не найдена в {c.chunk_id}: «{c.quote[:80]}»")
    return ok, problems, info


def _stem(word: str) -> str:
    """Возвращает основу слова без двух последних букв для поиска с учетом падежей."""
    w = normalize(word)
    return w[: max(4, len(w) - 2)]  # тикагрелор -> тикагрел(ора/ор)


def drug_mentioned(name: str, chunks: list[Chunk]) -> bool:
    """Проверяет, что препарат или его фармгруппа упоминается в процитированных чанках."""
    words = [w for w in re.findall(r"[a-zа-яё]+", normalize(name)) if len(w) >= 4]
    if not words:
        return True
    stem = _stem(words[0])  # МНН - первое слово: "эноксапарин натрия"
    if any(stem in chunk_haystack(c) for c in chunks):
        return True
    item_classes = classes_of(name) - {"contrast", "hydration"}
    return bool(item_classes) and any(item_classes & classes_of(c.full_text) for c in chunks)


def dose_numbers(dose: str | None) -> list[str]:
    """Извлекает числа из строки дозы."""
    if not dose:
        return []
    return [n.replace(",", ".") for n in re.findall(r"\d+(?:[.,]\d+)?", dose)]


def dose_supported(dose: str | None, chunks: list[Chunk]) -> bool:
    """Проверяет, что все числа дозы есть в процитированных чанках."""
    nums = dose_numbers(dose)
    if not nums:
        return True
    hay = " ".join(chunk_haystack(c) for c in chunks).replace(",", ".")
    tokens = set(re.findall(r"\d+(?:\.\d+)?", hay))
    return all(n in tokens for n in nums)


def verify_item(item: PlanItemDraft, allowed: dict[str, Chunk]) -> tuple[bool, list[str]]:
    """Проверяет ссылки, препарат и дозу назначения и возвращает результат с замечаниями."""
    chunks, problems, info = check_citations(item.citations, allowed)
    if not chunks:
        problems.append("нет ни одной подтверждённой ссылки на КР")
        return False, problems + info
    valid_ids = {c.chunk_id for c in chunks}
    item.citations = [c for c in item.citations if c.chunk_id in valid_ids]
    if item.kind == "drug":
        if not drug_mentioned(item.name, chunks):
            problems.append(f"препарат «{item.name}» не упоминается в процитированных пунктах")
        if not dose_supported(item.dose, chunks):
            problems.append(f"доза «{item.dose}» не подтверждена процитированным текстом")
    return not problems, problems + info
