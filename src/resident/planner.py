from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz, utils

from resident.citations import normalize, verify_item
from resident.config import CR_ID
from resident.drugs import EXCLUSIVE_GROUPS, classes_of
from resident.llm.client import LLMClient
from resident.llm.context import format_chunks, format_patient, load_prompt
from resident.mandatory import RequiredGroup, required_groups
from resident.retrieval.base import Retriever
from resident.schemas import Chunk, Patient, PlanDraft, PlanItem, PlanItemDraft, TreatmentPlan


@dataclass(frozen=True)
class Domain:
    key: str
    title: str
    sections: list[str]
    queries: list[str]  # подзапросы по подтемам области: у каждой свои фрагменты в контексте
    kinds: list[str]


DOMAINS = [
    Domain("lab", "Лабораторная диагностика и контроль", ["2.3"],
           ["лабораторные исследования: динамика тропонина, общий анализ крови, креатинин и рСКФ, "
            "электролиты калий натрий, липиды, глюкоза, коагулограмма"], ["lab"]),
    Domain("instr", "Инструментальная диагностика и оценка риска", ["2.4", "2.5"],
           ["ЭКГ, мониторирование ЭКГ, эхокардиография с оценкой ФВ ЛЖ, стратификация риска шкала GRACE"],
           ["instrumental"]),
    Domain("inv", "Инвазивная стратегия и защита почек", ["3.1"],
           ["сроки коронарографии и ЧКВ в зависимости от категории риска, лучевой доступ, "
            "профилактика контраст-индуцированной нефропатии"], ["procedure", "regimen"]),
    Domain("apt", "Антиагрегантная терапия и защита ЖКТ", ["3.2.3.1", "А3"],
           ["ацетилсалициловая кислота всем пациентам вне зависимости от стратегии лечения, нагрузочная и "
            "поддерживающая доза АСК",
            "ингибитор P2Y12 в добавление к АСК: тикагрелор, прасугрел, клопидогрел, нагрузочные и "
            "поддерживающие дозы"], ["drug"]),
    Domain("acg", "Парентеральные антикоагулянты", ["3.2.3.2", "А3"],
           ["парентеральные антикоагулянты: нефракционированный гепарин, эноксапарин, фондапаринукс; "
            "выбор и дозы с учетом стратегии и функции почек"], ["drug"]),
    Domain("base", "Вторичная профилактика: бета-адреноблокаторы, блокаторы РААС, статины",
           ["3.2.5", "3.2.7", "3.2.8", "А3"],
           ["бета-адреноблокаторы при сниженной ФВ ЛЖ",
            "ингибиторы АПФ у всех пациентов с ОКСбпST",
            "антагонисты рецепторов ангиотензина II (АРА) валсартан в случае непереносимости иАПФ",
            "антагонисты минералокортикоидных рецепторов эплеренон при ФВ ЛЖ 40% и менее",
            "ингибиторы ГМГ-КоА-редуктазы в высокой дозе вне зависимости от исходного холестерина, эзетимиб"],
           ["drug"]),
    Domain("sym", "Симптоматическая терапия и коррекция нарушений", ["3.2.1", "3.2.2", "3.2.4", "3.3", "А3"],
           ["обезболивание морфин при сохраняющейся боли", "кислородотерапия при гипоксемии",
            "органические нитраты при продолжающейся ишемии", "контроль гликемии инсулин, отмена НПВП"],
           ["drug", "regimen"]),
]


def missing_groups(domain_key: str, items: list[PlanItem], patient: Patient) -> list[RequiredGroup]:
    """Возвращает обязательные для пациента группы области, которых нет среди подтвержденных назначений."""
    given = set().union(*(classes_of(i.name) for i in items if i.kind == "drug" and i.verification == "verified"))
    return [g for g in required_groups(patient) if g.domain == domain_key and not g.classes & given]


def allergy_conflicts(item: PlanItemDraft, patient: Patient) -> list[str]:
    """Возвращает замечания, если препарат из группы аллергена пациента."""
    notes = []
    item_classes = classes_of(item.name)
    for a in patient.allergies:
        if item_classes & classes_of(a.substance) or normalize(a.substance)[:6] in normalize(item.name):
            notes.append(f"противопоказано: аллергия на {a.substance} ({a.reaction}) - выбери альтернативу из КР")
    return notes


def exclusive_conflict(item: PlanItemDraft, existing: list[PlanItem]) -> str | None:
    """Возвращает причину отказа, если препарат той же группы P2Y12 или антикоагулянтов уже назначен."""
    item_classes = classes_of(item.name)
    for group, members in EXCLUSIVE_GROUPS.items():
        if item_classes & members:
            for e in existing:
                if e.kind == "drug" and classes_of(e.name) & members:
                    return f"{group} уже назначен: {e.name} ({e.item_id})"
    return None


def is_duplicate(item: PlanItemDraft, existing: list[PlanItem]) -> bool:
    """Проверяет, дублирует ли назначение уже включенное в план."""
    ids = {c.chunk_id for c in item.citations}
    for e in existing:
        if e.kind != item.kind:
            continue
        if normalize(e.name) == normalize(item.name):
            return True
        if ids & {c.chunk_id for c in e.citations} and fuzz.token_set_ratio(e.name, item.name, processor=utils.default_process) >= 60:
            return True
    return False


def patient_modifiers(p: Patient) -> str:
    """Собирает особенности пациента для поискового запроса."""
    mods = []
    egfr = p.lab("egfr")
    if egfr is not None and egfr < 60:
        mods.append(f"ХБП сниженная рСКФ {egfr:g} нарушение функции почек")
    if p.gi_bleeding_risk:
        mods.append("язвенная болезнь риск желудочно-кишечного кровотечения ингибиторы протонного насоса")
    if p.high_bleeding_risk:
        mods.append("высокий риск кровотечений")
    if p.age >= 75:
        mods.append("пожилой возраст")
    if p.diabetes:
        mods.append("сахарный диабет")
    for a in p.allergies:
        mods.append(f"непереносимость {a.substance}")
    if p.very_high_risk:
        mods.append("критерии очень высокого риска")
    return " ".join(mods)


@dataclass
class DomainTrace:
    domain: str
    queries: list[str]
    chunk_ids: list[str]
    drafted: int = 0
    repaired: int = 0
    unsupported: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)  # отсеяно защитой от взаимоисключающих назначений
    completed: list[str] = field(default_factory=list)  # добавлено после проверки полноты


class Planner:
    def __init__(self, retriever: Retriever, llm: LLMClient, k: int = 8, max_context: int = 14):
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.max_context = max_context
        self.system = load_prompt("planner_system")

    def retrieve(self, domain: Domain, patient: Patient) -> tuple[list[str], list[Chunk]]:
        """Ищет чанки КР по подзапросам области с учетом особенностей пациента и чередует их выдачу."""
        mods = patient_modifiers(patient)
        k = max(3, self.k // len(domain.queries))
        queries, ranked = [], []
        for base in domain.queries:
            topic = [base, f"{base} {mods}"] if mods else [base]
            queries += topic
            ranked.append([h.chunk for q in topic for h in self.retriever.search(q, k=k, sections=domain.sections)])
        # по кругу: сначала лучшие фрагменты каждой подтемы, чтобы ни одна не выпала из контекста
        chunks: dict[str, Chunk] = {}
        for rank in range(max(map(len, ranked))):
            for topic_chunks in ranked:
                if rank < len(topic_chunks):
                    chunks.setdefault(topic_chunks[rank].chunk_id, topic_chunks[rank])
        return queries, list(chunks.values())[: self.max_context]

    def draft(self, domain: Domain, patient: Patient, chunks: list[Chunk]) -> list[PlanItemDraft]:
        """Запрашивает у LLM назначения для области плана по найденным чанкам."""
        user = (
            f"## Пациент\n{format_patient(patient)}\n\n"
            f"## Область плана: {domain.title}\n"
            f"Допустимые типы назначений (kind): {', '.join(domain.kinds)}\n\n"
            f"## Фрагменты КР\n{format_chunks(chunks)}\n\n"
            "## Задание\nСоставь назначения для этой области плана."
        )
        return self.llm.complete(self.system, user, PlanDraft).items

    def repair(self, domain: Domain, patient: Patient, chunks: list[Chunk],
               failed: list[tuple[PlanItemDraft, list[str]]]) -> list[PlanItemDraft]:
        """Просит LLM исправить назначения, не прошедшие проверку."""
        problems = "\n".join(
            f"- {d.model_dump_json()}\n  Замечания: {'; '.join(notes)}" for d, notes in failed
        )
        user = (
            f"## Пациент\n{format_patient(patient)}\n\n"
            f"## Фрагменты КР\n{format_chunks(chunks)}\n\n"
            f"## Назначения с ошибками\n{problems}\n\n## Задание\n{load_prompt('planner_repair')}"
        )
        return self.llm.complete(self.system, user, PlanDraft).items

    @staticmethod
    def check(draft: PlanItemDraft, allowed: dict[str, Chunk], patient: Patient) -> tuple[bool, list[str]]:
        """Проверяет ссылки назначения на КР и аллергии пациента."""
        ok, notes = verify_item(draft, allowed)
        conflicts = allergy_conflicts(draft, patient) if draft.kind == "drug" else []
        return ok and not conflicts, conflicts + notes

    def plan(self, patient: Patient) -> tuple[TreatmentPlan, list[DomainTrace]]:
        """Строит план по всем клиническим областям с проверкой, исправлением и защитой от дублей."""
        plan = TreatmentPlan(patient_id=patient.patient_id)
        traces = []
        for domain in DOMAINS:
            queries, chunks = self.retrieve(domain, patient)
            allowed = {c.chunk_id: c for c in chunks}
            trace = DomainTrace(domain.key, queries, list(allowed))
            drafts = self.draft(domain, patient, chunks)
            trace.drafted = len(drafts)

            drafts = [d for d in drafts if d.name.strip()]
            checked = [(d, *self.check(d, allowed, patient)) for d in drafts]
            failed = [(d, notes) for d, ok, notes in checked if not ok]
            if failed:
                repaired = self.repair(domain, patient, chunks, failed)
                trace.repaired = len(repaired)
                checked = [(d, ok, n) for d, ok, n in checked if ok]
                checked += [(d, *self.check(d, allowed, patient)) for d in repaired if d.name.strip()]

            self._accept(checked, domain, plan, trace, start=1)

            # проверка полноты: группы, которые КР рекомендует этому пациенту, должны быть в плане
            if missing := missing_groups(domain.key, plan.items, patient):
                by_id = {c.chunk_id: c for c in self.retriever.chunks}
                ids = [f"{CR_ID}:{g.ref}" for g in missing]
                chunks += [by_id[cid] for cid in ids if cid not in allowed and cid in by_id]
                allowed = {c.chunk_id: c for c in chunks}
                added = [d for d in self.complete_omissions(patient, chunks, missing) if d.name.strip()]
                trace.completed = [d.name for d in added]
                self._accept([(d, *self.check(d, allowed, patient)) for d in added], domain, plan, trace,
                             start=len(checked) + 1)
            traces.append(trace)
        return plan, traces

    def complete_omissions(self, patient: Patient, chunks: list[Chunk],
                           missing: list[RequiredGroup]) -> list[PlanItemDraft]:
        """Просит LLM дополнить план группами, которые КР рекомендует этому пациенту без противопоказаний."""
        groups = "\n".join(f"- {g.label} (тезис {CR_ID}:{g.ref})" for g in missing)
        user = (
            f"## Пациент\n{format_patient(patient)}\n\n"
            f"## Фрагменты КР\n{format_chunks(chunks)}\n\n"
            f"## Нет в плане\n{groups}\n\n## Задание\n{load_prompt('planner_omission')}"
        )
        return self.llm.complete(self.system, user, PlanDraft).items

    @staticmethod
    def _accept(checked: list[tuple[PlanItemDraft, bool, list[str]]], domain: Domain, plan: TreatmentPlan,
                trace: DomainTrace, start: int) -> None:
        """Добавляет проверенные назначения в план, пропуская дубли и взаимоисключающие препараты."""
        # подтверждённые - первыми, чтобы при дублях оставался вариант с проверенной ссылкой
        checked = sorted(checked, key=lambda x: not x[1])
        for n, (d, ok, notes) in enumerate(checked, start):
            if is_duplicate(d, plan.items):
                continue
            if d.kind == "drug" and (conflict := exclusive_conflict(d, plan.items)):
                trace.dropped.append(f"{d.name}: {conflict}")
                continue
            item = PlanItem(
                **d.model_dump(), item_id=f"{domain.key}-{n}", source="planner",
                verification="verified" if ok else "unsupported", verification_notes=notes,
            )
            if not ok:
                trace.unsupported.append(item.item_id)
            plan.items.append(item)
