from __future__ import annotations

from dataclasses import dataclass, field

from resident.citations import check_citations, normalize, verify_item
from resident.drugs import classes_of
from resident.kb.parse import load_chunks
from resident.llm.client import LLMClient
from resident.llm.context import format_chunks, format_patient, format_plan, load_prompt
from resident.retrieval.base import Retriever
from resident.schemas import (
    Chunk,
    Deviation,
    LabValue,
    Patient,
    PlanChange,
    PlanChangeDraft,
    PlanItem,
    StopOrder,
    TreatmentPlan,
)

# Для каждого типа отклонения - поисковый запрос и разделы КР.
DEVIATION_QUERIES: dict[str, tuple[str, list[str]]] = {
    "aki": ("острое повреждение почек после контраста: гидратация, объём контраста, отмена нефротоксичных "
            "препаратов, коррекция доз антитромботических средств при нарушении функции почек",
            ["3.1.3", "3.2.3.2", "3.3", "А3"]),
    "egfr_fondaparinux": ("антикоагулянт при выраженном нарушении функции почек: фондапаринукс противопоказан, НФГ",
                          ["3.2.3.2", "А3"]),
    "egfr_enoxaparin": ("эноксапарин при рСКФ ниже 30: коррекция дозы, альтернатива НФГ", ["3.2.3.2", "А3"]),
    "planned_contrast_risk": ("КГ и ЧКВ у пациентов с ХБП: оценка риска и пользы, ограничение объёма контраста",
                              ["3.1.2", "3.1.3"]),
    "hb_drop": ("кровотечение на фоне антитромботической терапии: ингибиторы протонного насоса, гемотрансфузия, "
                "переход на клопидогрел", ["3.2.3.1", "3.3"]),
    "hyperkalemia": ("контроль калия, ингибиторы АПФ и антагонисты альдостерона при гиперкалиемии", ["2.3", "3.2.7"]),
    "thrombocytopenia": ("тромбоцитопения: антиагреганты не рекомендуются", ["3.2.3.1"]),
    "reinfarction": ("рецидив инфаркта миокарда, ранняя постинфарктная стенокардия: коронарография", ["3.4.6", "3.1.1"]),
}


@dataclass
class ReplanTrace:
    queries: list[str]
    chunk_ids: list[str]
    rejected_stops: list[str] = field(default_factory=list)
    unsupported_adds: list[str] = field(default_factory=list)


class Replanner:
    def __init__(self, retriever: Retriever, llm: LLMClient, k: int = 6, max_context: int = 14):
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.max_context = max_context
        self.system = load_prompt("replanner_system")
        self._kb = {c.chunk_id: c for c in load_chunks()}

    def context(self, deviations: list[Deviation]) -> tuple[list[str], list[Chunk]]:
        """Собирает чанки КР по порогам и запросам для выявленных отклонений."""
        chunks: dict[str, Chunk] = {}
        for d in deviations:  # пункты КР, задающие порог, - всегда в контексте
            for cid in d.chunk_ids:
                if cid in self._kb:
                    chunks.setdefault(cid, self._kb[cid])
        queries = []
        for code in dict.fromkeys(d.code for d in deviations):
            if code in DEVIATION_QUERIES:
                q, sections = DEVIATION_QUERIES[code]
                queries.append(q)
                for hit in self.retriever.search(q, k=self.k, sections=sections):
                    chunks.setdefault(hit.chunk.chunk_id, hit.chunk)
        return queries, list(chunks.values())[: self.max_context]

    def replan(self, patient: Patient, plan: TreatmentPlan, deviations: list[Deviation],
               labs: list[LabValue], t_now: float) -> tuple[PlanChange, ReplanTrace]:
        """Запрашивает у LLM пересмотр плана и проверяет ссылки отмен и новых назначений."""
        queries, chunks = self.context(deviations)
        allowed = {c.chunk_id: c for c in chunks}
        dev_text = "\n".join(f"- [{d.severity}] {d.message} (порог: {d.source})" for d in deviations)
        user = (
            f"## Пациент\n{format_patient(patient, labs)}\n\n"
            f"## Действующий план (время {t_now:g} ч от госпитализации)\n{format_plan(plan.active_items())}\n\n"
            f"## Отклонения на контрольной точке\n{dev_text}\n\n"
            f"## Фрагменты КР\n{format_chunks(chunks)}\n\n"
            "## Задание\nПересмотри план: что отменить/отложить и что назначить."
        )
        draft = self.llm.complete(self.system, user, PlanChangeDraft)
        trace = ReplanTrace(queries, list(allowed))

        active = {i.item_id: i for i in plan.active_items()}
        stops: list[StopOrder] = []
        for s in draft.stop:
            ok_chunks, _, _ = check_citations(s.citations, allowed)
            if s.item_id not in active or not ok_chunks:
                trace.rejected_stops.append(f"{s.item_id}: нет в плане или ссылка не подтверждена")
            elif not stop_refers_to(active[s.item_id], s, allowed):
                trace.rejected_stops.append(f"{s.item_id}: причина отмены относится к другому назначению")
            else:
                stops.append(s)

        added: list[PlanItem] = []
        for n, d in enumerate(draft.add, 1):
            ok, notes = verify_item(d, allowed)
            item = PlanItem(
                **d.model_dump(), item_id=f"re-{n}", start_h=t_now, source="replanner",
                verification="verified" if ok else "unsupported", verification_notes=notes,
            )
            if not ok:
                trace.unsupported_adds.append(item.item_id)
            added.append(item)
        change = PlanChange(t_h=t_now, deviations=deviations, stopped=stops, added=added, rationale=draft.rationale)
        return change, trace


def stop_refers_to(item: PlanItem, stop: StopOrder, allowed: dict[str, Chunk]) -> bool:
    """Проверяет, что причина отмены или процитированный тезис относятся к отменяемому назначению."""
    text = " ".join([stop.reason] + [allowed[c.chunk_id].full_text for c in stop.citations if c.chunk_id in allowed])
    if item_classes := classes_of(item.name):
        return bool(item_classes & classes_of(text))
    return normalize(item.name)[:6] in normalize(text)


def apply_change(plan: TreatmentPlan, change: PlanChange) -> TreatmentPlan:
    """Возвращает копию плана с примененными отменами и новыми назначениями."""
    new = plan.model_copy(deep=True)
    for s in change.stopped:
        item = new.get(s.item_id)
        item.stop_h = change.t_h
        item.verification_notes.append(f"отменено при пересмотре плана: {s.reason}")
    new.items.extend(change.added)
    return new
