from __future__ import annotations

from resident.auditor.llm_review import LLMAuditor
from resident.auditor.rules import AuditContext, run_rules
from resident.schemas import AuditReport, Deviation, Finding, PlanChange

SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}


def audit(ctx: AuditContext, llm_auditor: LLMAuditor | None = None,
          deviations: list[Deviation] | None = None, changes: list[PlanChange] | None = None) -> AuditReport:
    """Запускает правила и LLM-аудитор и возвращает отчет с находками по убыванию тяжести."""
    findings, rejected = run_rules(ctx), []
    if llm_auditor is not None:
        kept, rejected = llm_auditor.review(ctx, findings, deviations or [], changes or [])
        findings += kept
    findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], f.source != "rules", f.rule_id))
    return AuditReport(case_id=ctx.case.case_id, findings=findings, rejected=rejected)


def expected_coverage(report: AuditReport, expected: list[str]) -> dict:
    """Считает, какие заложенные в кейс дефекты аудитор нашел."""
    found = {f.rule_id for f in report.findings}
    hit = [r for r in expected if r in found]
    return {"expected": expected, "found": hit, "missed": [r for r in expected if r not in found],
            "recall": len(hit) / len(expected) if expected else None}


def finding_line(f: Finding) -> str:
    """Форматирует находку аудитора в строку для вывода."""
    cites = "; ".join(f"{c.chunk_id}: «{c.quote[:90]}…»" for c in f.citations)
    ref = f.legal_ref or cites or "-"
    flag = "" if f.verified else " [ссылка не подтверждена]"
    review = f"\n    проверка второй моделью: {f.review}" if f.review else ""
    return (f"[{f.severity.upper()}] {f.rule_id} ({f.category}) {', '.join(f.item_ids)} - {f.message}{flag}"
            f"\n    основание: {ref}{review}")
