"""Заглушка LLM для офлайн-тестов: отвечает по схеме, цитируя переданные чанки."""
from __future__ import annotations

import re

from resident.schemas import (
    Citation,
    FindingVerdict,
    LLMAuditDraft,
    PlanChangeDraft,
    PlanDraft,
    PlanItemDraft,
    StopOrder,
    VerdictDraft,
)

CHUNK_RE = re.compile(r'<chunk id="([^"]+)"[^>]*>\nРаздел: [^\n]*\nТекст: ([^\n]+)')
KNOWN_DRUGS = ["тикагрелор", "клопидогрел", "прасугрел", "эноксапарин натрия", "фондапаринукс натрия",
               "морфин", "метопролол", "ингибиторы протонного насоса", "ацетилсалициловая кислота"]


def _drug_in(text: str) -> str | None:
    low = text.lower()
    return next((d for d in KNOWN_DRUGS if d[:8] in low), None)


class FakeLLM:
    """Для плана: по одному назначению на каждый из первых двух чанков контекста.

    `bad_quote=True` - первое назначение получает выдуманную цитату (проверка верификатора).
    """

    def __init__(self, drug_names: dict[str, str] | None = None, bad_quote: bool = False, stop_ids: list[str] | None = None):
        self.drug_names = drug_names or {}
        self.bad_quote = bad_quote
        self.stop_ids = stop_ids or []
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, user, schema):
        self.calls.append((schema.__name__, user))
        chunks = CHUNK_RE.findall(user)
        if schema is PlanDraft:
            m = re.search(r"kind\): (\w+)", user)
            if m is None:  # запрос на исправление - «не удалось обосновать»
                return PlanDraft(items=[])
            kind, items = m.group(1), []
            for n, (cid, text) in enumerate(chunks[:2]):
                quote = "выдуманная цитата, которой нет в КР" if self.bad_quote and n == 0 else text[:60]
                name = self.drug_names.get(cid) or (_drug_in(text) if kind == "drug" else f"назначение по {cid}")
                if name:
                    items.append(PlanItemDraft(kind=kind, name=name, rationale="тест",
                                               citations=[Citation(chunk_id=cid, quote=quote)]))
            return PlanDraft(items=items)
        if schema is PlanChangeDraft:
            cid, text = chunks[0]
            cit = [Citation(chunk_id=cid, quote=text[:60])]
            return PlanChangeDraft(
                stop=[StopOrder(item_id=i, reason="тест", citations=cit) for i in self.stop_ids],
                add=[PlanItemDraft(kind="lab", name="Креатинин ежедневно", rationale="тест", citations=cit)],
                rationale="тестовый пересмотр",
            )
        if schema is LLMAuditDraft:
            return LLMAuditDraft(findings=[])
        if schema is VerdictDraft:  # проверяющая модель подтверждает все находки
            return VerdictDraft(verdicts=[
                FindingVerdict(index=int(n), reason="тест", condition_met=True, fact_correct=True)
                for n in re.findall(r"### Находка (\d+)", user)])
        raise AssertionError(schema)
