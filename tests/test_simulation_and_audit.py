import re
from pathlib import Path

import pytest

from fakes import FakeLLM
from resident.auditor.rules import cite
from resident.llm.client import CachedLLM, CacheMiss
from resident.pipeline import Pipeline, load_case
from resident.schemas import PlanDraft, TreatmentPlan
from resident.simulator.engine import simulate
from resident.simulator.kinetics import egfr_ckd_epi_2021


def test_ckd_epi_reference_values():
    # креатинин 1,0 мг/дл (88,4 мкмоль/л), мужчина 50 лет -> ~92 по CKD-EPI 2021
    assert egfr_ckd_epi_2021(88.4, 50, "M") == pytest.approx(92, abs=1.5)
    assert egfr_ckd_epi_2021(88.4, 50, "F") < egfr_ckd_epi_2021(70, 50, "F")


def test_simulation_is_deterministic():
    case = load_case("case3")
    plan = TreatmentPlan(patient_id="x")
    assert simulate(case, plan).labs == simulate(case, plan).labs


def test_rule_citations_exist_in_kb():
    src = Path("src/resident/auditor/rules.py").read_text(encoding="utf-8")
    calls = re.findall(r'cite\("([^"]+)", "([^"]+)"\)', src)
    assert len(calls) >= 10
    for chunk_id, phrase in calls:
        cite(chunk_id, phrase)  # бросит ValueError, если цитаты нет


@pytest.fixture(scope="module")
def runs(bm25):
    pipe = Pipeline(retriever=bm25, llm=FakeLLM(stop_ids=["proc-2"]))
    return {cid: pipe.run(load_case(cid)) for cid in ("case1", "case2", "case3")}


def test_baseline_case_is_stable(runs):
    run = runs["case1"]
    assert run.deviations == [] and run.changes == []
    tn = run.sim_final.series("troponin_hs")
    assert max(v.value for v in tn) > tn[0].value > tn[-1].value  # подъём и спад


def test_reactive_case_triggers_replanning(runs):
    run = runs["case3"]
    codes = {d.code for d in run.deviations}
    assert {"aki", "planned_contrast_risk"} <= codes
    change = run.changes[0]
    assert [s.item_id for s in change.stopped] == ["proc-2"]  # этапное ЧКВ отложено
    assert not run.plan_final.get("proc-2").active


@pytest.mark.parametrize("case_id", ["case2", "case3"])
def test_auditor_finds_seeded_defects(runs, case_id):
    assert runs[case_id].coverage["recall"] == 1.0


def test_invented_quotes_never_reach_plan(bm25):
    llm = FakeLLM(bad_quote=True)
    run = Pipeline(retriever=bm25, llm=llm, use_llm_auditor=False).run(load_case("case1"))
    assert any("Назначения с ошибками" in user for _, user in llm.calls)  # был запрос на исправление
    planner_items = [i for i in run.plan_initial.items if i.source == "planner"]
    assert planner_items and all(i.verification == "verified" for i in planner_items)
    assert not any("выдуманная" in c.quote for i in planner_items for c in i.citations)


def test_replay_mode_requires_cache(tmp_path):
    llm = CachedLLM(mode="replay", cache_dir=tmp_path)
    with pytest.raises(CacheMiss):
        llm.complete("s", "u", PlanDraft)


def test_planner_guards():
    from resident.planner import allergy_conflicts, exclusive_conflict
    from resident.schemas import PlanItem, PlanItemDraft

    patient = load_case("case2").patient  # ангиоотёк на эналаприл
    ramipril = PlanItemDraft(kind="drug", name="рамиприл", rationale="", citations=[])
    valsartan = PlanItemDraft(kind="drug", name="валсартан", rationale="", citations=[])
    assert allergy_conflicts(ramipril, patient) and not allergy_conflicts(valsartan, patient)

    ticagrelor = PlanItem(item_id="apt-1", kind="drug", name="тикагрелор", rationale="", citations=[])
    clopidogrel = PlanItemDraft(kind="drug", name="клопидогрел", rationale="", citations=[])
    assert "P2Y12" in exclusive_conflict(clopidogrel, [ticagrelor])


def test_omission_rule_suggests_arb_for_acei_allergy(runs):
    omissions = [f for f in runs["case2"].audit.findings if f.rule_id == "R16_omission"]
    assert any("валсартан" in f.message and f.citations[0].chunk_id == "154_4:3.2.7:t03" for f in omissions)


def test_required_groups_follow_lvef_and_contraindications():
    from resident.mandatory import required_groups

    def labels(p):
        return {g.label.split(" (")[0] for g in required_groups(p)}

    case1, case2 = load_case("case1").patient, load_case("case2").patient
    assert {"бета-адреноблокатор", "антагонист альдостерона"} <= labels(case2)  # ФВ 38%, ХСН
    assert "бета-адреноблокатор" not in labels(case1)  # ФВ 55%
    labs = [v.model_copy(update={"value": 5.6}) if v.code == "potassium" else v for v in case2.baseline_labs]
    assert "антагонист альдостерона" not in labels(case2.model_copy(update={"baseline_labs": labs}))  # K+ > 5


def test_r13_flags_anticoagulant_started_after_pci():
    from resident.auditor.rules import AuditContext, r13_anticoagulant_after_pci
    from resident.pipeline import schedule_items
    from resident.schemas import PlanItem

    case = load_case("case1")  # ЧКВ на 10 ч
    ufh = PlanItem(item_id="re-1", kind="drug", name="Нефракционированный гепарин", rationale="замена",
                   citations=[], start_h=72, source="replanner")
    plan = TreatmentPlan(patient_id=case.patient.patient_id, items=schedule_items(case) + [ufh])
    found = r13_anticoagulant_after_pci(AuditContext(case=case, plan=plan, sim=simulate(case, plan), t_end=72))
    assert [f.item_ids for f in found] == [["re-1"]] and "начат на 72 ч" in found[0].message


def test_verifier_rejects_unconfirmed_llm_findings(runs, bm25):
    from fakes import CHUNK_RE
    from resident.auditor.llm_review import LLMAuditor
    from resident.auditor.rules import AuditContext
    from resident.schemas import Citation, FindingVerdict, LLMAuditDraft, LLMFindingDraft, VerdictDraft

    class Auditor:
        def complete(self, system, user, schema):
            cid, text = CHUNK_RE.findall(user)[0]
            return LLMAuditDraft(findings=[
                LLMFindingDraft(evidence="факт", severity="major", category="clinical", item_ids=[], message=m,
                                citations=[Citation(chunk_id=cid, quote=text[:40])])
                for m in ("верная", "спорная")])

    class Verifier:
        def complete(self, system, user, schema):
            assert "### Находка 1" in user
            return VerdictDraft(verdicts=[
                FindingVerdict(index=0, reason="подтверждено", condition_met=True, fact_correct=True),
                FindingVerdict(index=1, reason="условие тезиса не выполнено", condition_met=False, fact_correct=True)])

    run = runs["case1"]
    ctx = AuditContext(case=run.case, plan=run.plan_final, sim=run.sim_final, t_end=72)
    kept, rejected = LLMAuditor(bm25, Auditor(), verifier=Verifier()).review(ctx, [], [], [])
    assert [f.message.split(" (")[0] for f in kept] == ["верная"]
    assert rejected[0].review == "условие тезиса не выполнено"


def test_replanner_rejects_stop_with_reason_about_other_item():
    from resident.kb.parse import load_chunks
    from resident.replanner import stop_refers_to
    from resident.schemas import Citation, PlanItem, StopOrder

    kb = {c.chunk_id: c for c in load_chunks()}
    aspirin = PlanItem(item_id="apt-1", kind="drug", name="Ацетилсалициловая кислота", rationale="", citations=[])
    enox = PlanItem(item_id="ext-1", kind="drug", name="Эноксапарин натрия", rationale="", citations=[])
    about_metformin = StopOrder(item_id="apt-1", reason="ОПП - противопоказание к метформину",
                                citations=[Citation(chunk_id="154_4:3.1.3:t01", quote="метформин")])
    about_dose = StopOrder(item_id="ext-1", reason="рСКФ < 30, доза не скорректирована",
                           citations=[Citation(chunk_id="154_4:А3:tab04.r11", quote="1 мг/кг 1 раз в сутки")])
    assert not stop_refers_to(aspirin, about_metformin, kb)  # id аспирина, а причина - метформин
    assert stop_refers_to(enox, about_dose, kb)  # препарат назван в процитированной строке таблицы
