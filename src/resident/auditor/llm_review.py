from __future__ import annotations

from resident.auditor.rules import AuditContext
from resident.citations import check_citations
from resident.llm.client import LLMClient
from resident.llm.context import format_chunks, format_labs, format_patient, format_plan, load_prompt
from resident.retrieval.base import Retriever
from resident.schemas import Chunk, Deviation, Finding, LLMAuditDraft, PlanChange, VerdictDraft

AUDIT_QUERIES = [
    ("противопоказания и безопасность антитромботической терапии, риск кровотечений, дозы при ХБП",
     ["3.2.3", "А3"]),
    ("обязательные исследования и контроль: тропонин, креатинин рСКФ, электролиты, ЭхоКГ", ["2.3", "2.4"]),
    ("инвазивная стратегия, контраст и защита почек", ["3.1"]),
    ("базисная терапия: статины, бета-адреноблокаторы, ингибиторы АПФ, НПВП не рекомендуются", ["3.2.5", "3.2.7", "3.2.8", "3.3"]),
]


def _track_text(ctx: AuditContext) -> str:
    """Форматирует все назначения трека и согласия в текст для LLM-аудитора."""
    lines = []
    for i in ctx.plan.items:
        who = f", назначил: {i.ordered_by}" if i.ordered_by else ""
        ver = "" if i.verification == "not_checked" else f", ссылка на КР: {i.verification}"
        extra = f", контраст {i.contrast_ml:g} мл" if i.contrast_ml else ""
        stop = f", причина отмены: {i.verification_notes[-1]}" if not i.active and i.verification_notes else ""
        lines.append(f"{format_plan([i])}{extra}{who}{ver}{stop}")
    consents = "; ".join(f"{c.procedure} ({c.procedure_id}, {c.t_h:g} ч)" for c in ctx.case.consents) or "нет"
    return "\n".join(lines) + f"\nИнформированные согласия: {consents}"


def _labs_text(ctx: AuditContext) -> str:
    """Форматирует динамику анализов по времени в текст для LLM-аудитора."""
    times = sorted({v.t_h for v in ctx.sim.labs})
    return "\n".join(f"- {t:g} ч: {format_labs([v for v in ctx.sim.labs if v.t_h == t])}" for t in times)


class LLMAuditor:
    def __init__(self, retriever: Retriever, llm: LLMClient, verifier: LLMClient | None = None,
                 k: int = 5, max_context: int = 16):
        self.retriever = retriever
        self.llm = llm
        self.verifier = verifier  # модель другого семейства, которая перепроверяет находки
        self.k = k
        self.max_context = max_context
        self.system = load_prompt("auditor_system")

    def context(self) -> list[Chunk]:
        """Собирает фрагменты КР для аудита по фиксированному набору запросов."""
        chunks: dict[str, Chunk] = {}
        for q, sections in AUDIT_QUERIES:
            for hit in self.retriever.search(q, k=self.k, sections=sections):
                chunks.setdefault(hit.chunk.chunk_id, hit.chunk)
        return list(chunks.values())[: self.max_context]

    def review(self, ctx: AuditContext, rule_findings: list[Finding], deviations: list[Deviation],
               changes: list[PlanChange]) -> tuple[list[Finding], list[Finding]]:
        """Запрашивает у LLM дополнительные находки, проверяет ссылки и возвращает принятые и отклоненные."""
        chunks = self.context()
        allowed = {c.chunk_id: c for c in chunks}
        known = "\n".join(f"- {f.rule_id}: {f.message}" for f in rule_findings) or "нет"
        devs = "\n".join(f"- {d.t_h:g} ч: {d.message}" for d in deviations) or "нет"
        chg = "\n".join(
            f"- {c.t_h:g} ч: отменено {[s.item_id for s in c.stopped]}, добавлено {[a.name for a in c.added]}. {c.rationale}"
            for c in changes
        ) or "нет"
        user = (
            f"## Пациент\n{format_patient(ctx.case.patient)}\n\n"
            f"## Трек назначений\n{_track_text(ctx)}\n\n"
            f"## Динамика анализов\n{_labs_text(ctx)}\n\n"
            f"## Отклонения\n{devs}\n\n## Изменения тактики\n{chg}\n\n"
            f"## Уже найдено правилами\n{known}\n\n"
            f"## Фрагменты КР\n{format_chunks(chunks)}\n\n"
            "## Задание\nПроведи аудит трека и верни дополнительные находки."
        )
        draft = self.llm.complete(self.system, user, LLMAuditDraft)
        findings = []
        item_ids = {i.item_id for i in ctx.plan.items}
        for f in draft.findings:
            ok_chunks, _, _ = check_citations(f.citations, allowed)
            findings.append(Finding(
                rule_id="LLM", severity=f.severity, category=f.category,
                item_ids=[i for i in f.item_ids if i in item_ids], message=f"{f.message} (факт: {f.evidence})",
                citations=f.citations, source="llm", verified=bool(ok_chunks),
            ))
        return self.verify(ctx, findings, allowed) if self.verifier and findings else (findings, [])

    def verify(self, ctx: AuditContext, findings: list[Finding],
               allowed: dict[str, Chunk]) -> tuple[list[Finding], list[Finding]]:
        """Отдает находки второй модели и оставляет те, где условие тезиса и факт из трека подтверждены."""
        blocks = []
        for n, f in enumerate(findings):
            theses = "\n".join(f"  [{c.chunk_id}] {allowed[c.chunk_id].full_text}"
                               for c in f.citations if c.chunk_id in allowed)
            blocks.append(f"### Находка {n}\n{f.message}\nЦитируемые тезисы КР:\n{theses or '  нет'}")
        user = (
            f"## Пациент\n{format_patient(ctx.case.patient)}\n\n"
            f"## Трек назначений\n{_track_text(ctx)}\n\n"
            f"## Динамика анализов\n{_labs_text(ctx)}\n\n"
            f"## Находки на проверку\n" + "\n\n".join(blocks)
        )
        verdicts = {v.index: v for v in self.verifier.complete(load_prompt("auditor_verifier"), user, VerdictDraft).verdicts}
        kept, rejected = [], []
        for n, f in enumerate(findings):
            v = verdicts.get(n)
            f.review = v.reason if v else "проверяющая модель не вернула вердикт"
            (kept if v and v.condition_met and v.fact_correct else rejected).append(f)
        return kept, rejected
