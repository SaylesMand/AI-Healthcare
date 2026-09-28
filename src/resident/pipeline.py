from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from resident.auditor.llm_review import LLMAuditor
from resident.auditor.report import audit, expected_coverage, finding_line
from resident.auditor.rules import AuditContext
from resident.config import CASES_DIR, settings
from resident.drugs import PARENTERAL_ANTICOAGULANT, classes_of
from resident.llm.client import CachedLLM, LLMClient
from resident.monitor import detect_deviations
from resident.planner import DomainTrace, Planner
from resident.replanner import Replanner, ReplanTrace, apply_change
from resident.retrieval import build_retriever
from resident.retrieval.base import Retriever
from resident.schemas import AuditReport, Case, Deviation, PlanChange, PlanItem, TreatmentPlan
from resident.simulator.engine import SimulationResult, simulate


def load_case(case_id: str) -> Case:
    """Загружает кейс из data/cases по id."""
    return Case.model_validate_json((CASES_DIR / f"{case_id}.json").read_text(encoding="utf-8"))


def schedule_items(case: Case) -> list[PlanItem]:
    """Превращает процедуры расписания и внешние назначения кейса в элементы плана."""
    items = [
        PlanItem(
            item_id=p.procedure_id, kind="procedure", name=p.name, start_h=p.t_h, source="external",
            ordered_by=p.ordered_by, contrast_ml=p.contrast_ml, invasive=p.invasive,
            rationale="по расписанию рентгенэндоваскулярной операционной", citations=[],
        )
        for p in case.procedures
    ]
    return items + [o.model_copy(update={"source": "external"}) for o in case.external_orders]


def apply_protocol(plan: TreatmentPlan, horizon_h: float) -> None:
    """Отменяет парентеральные антикоагулянты плана после выполненного ЧКВ."""
    pcis = [i for i in plan.items if i.kind == "procedure" and i.invasive and i.start_h <= horizon_h
            and ("чкв" in i.name.lower() or "стент" in i.name.lower())]
    if not pcis:
        return
    t_stop = pcis[0].start_h + 1
    for item in plan.items:
        if item.source != "external" and item.active and classes_of(item.name) & PARENTERAL_ANTICOAGULANT:
            item.stop_h = max(t_stop, item.start_h)
            item.verification_notes.append("прекращено после ЧКВ (КР п. 3.2.3.2, тезис 2)")


@dataclass
class CaseRun:
    case: Case
    plan_initial: TreatmentPlan
    planner_trace: list[DomainTrace]
    sim_initial: SimulationResult
    deviations: list[Deviation]
    changes: list[PlanChange] = field(default_factory=list)
    replan_traces: list[ReplanTrace] = field(default_factory=list)
    plan_final: TreatmentPlan | None = None
    sim_final: SimulationResult | None = None
    audit: AuditReport | None = None

    @property
    def coverage(self) -> dict:
        """Возвращает покрытие заложенных дефектов находками аудитора."""
        return expected_coverage(self.audit, self.case.expected_findings)


class Pipeline:
    def __init__(self, retriever: Retriever | None = None, llm: LLMClient | None = None,
                 auditor_llm: LLMClient | None = None, use_llm_auditor: bool = True):
        self.retriever = retriever or build_retriever()
        self.llm = llm or CachedLLM()
        # аудитор - модель другого семейства, чтобы ошибки планировщика и аудитора не совпадали
        self.auditor_llm = auditor_llm or (llm if llm is not None else CachedLLM(model=settings.auditor_model))
        self.planner = Planner(self.retriever, self.llm)
        self.replanner = Replanner(self.retriever, self.llm)
        # находки аудитора перепроверяет модель планировщика: две модели разных семейств контролируют друг друга
        self.llm_auditor = (LLMAuditor(self.retriever, self.auditor_llm, verifier=self.llm)
                            if use_llm_auditor else None)

    def run(self, case: Case) -> CaseRun:
        """Прогоняет кейс через план, симуляцию, пересмотр и аудит."""
        t_check = case.horizon_h
        # Контур 1: протокольный план + расписание процедур и внешние назначения
        plan, ptrace = self.planner.plan(case.patient)
        plan.items = schedule_items(case) + plan.items
        apply_protocol(plan, t_check)

        # Контур 2: симуляция до контрольной точки и реакция на отклонения
        sim = simulate(case, plan)
        deviations = detect_deviations(case, plan, sim, t_check)
        run = CaseRun(case, plan, ptrace, sim, deviations)
        final_plan, final_sim = plan, sim
        if deviations:
            change, rtrace = self.replanner.replan(case.patient, plan, deviations, sim.panel(t_check), t_check)
            final_plan = apply_change(plan, change)
            final_sim = simulate(case, final_plan)  # прошлое не меняется: изменения действуют с t_check
            run.changes.append(change)
            run.replan_traces.append(rtrace)
        run.plan_final, run.sim_final = final_plan, final_sim

        # Контур 3: аудит полного трека
        ctx = AuditContext(case=case, plan=final_plan, sim=final_sim, t_end=t_check)
        run.audit = audit(ctx, self.llm_auditor, deviations, run.changes)
        return run


def main(case_ids: list[str]) -> None:
    """Прогоняет кейсы и печатает находки аудитора."""
    pipe = Pipeline()
    for cid in case_ids:
        run = pipe.run(load_case(cid))
        print(f"\n===== {cid}: {run.case.title}")
        print(f"план: {len(run.plan_initial.items)} назначений; отклонения: {[d.code for d in run.deviations]}")
        for f in run.audit.findings:
            print(finding_line(f))
        for f in run.audit.rejected:
            print("[ОТКЛОНЕНО ПРОВЕРКОЙ]", finding_line(f))
        print("покрытие заложенных дефектов:", json.dumps(run.coverage, ensure_ascii=False))
    print(f"\nLLM {settings.llm_model}:", pipe.llm.stats)
    if pipe.auditor_llm is not pipe.llm:
        print(f"LLM-аудитор {settings.auditor_model}:", pipe.auditor_llm.stats)


if __name__ == "__main__":
    main(sys.argv[1:] or ["case1", "case2", "case3"])
