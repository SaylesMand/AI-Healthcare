from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from resident.drugs import ANTITHROMBOTIC, RAAS, has_class
from resident.schemas import Case, LabValue, PlanItem, TreatmentPlan
from resident.simulator import kinetics as k

TROPONIN_TIMES = [0, 1, 3, 24, 48, 72, 96]  # алгоритм 0/1(3) ч + ежедневно
DAILY_TIMES = [0, 24, 48, 72, 96]
NOISE_SD = {"troponin_hs": 0.05, "creatinine": 0.02, "hb": 1.5, "potassium": 0.08, "platelets": 4.0, "glucose": 0.3}


@dataclass
class ContrastExposure:
    item_id: str
    t_h: float
    contrast_ml: float
    egfr_before: float
    hydration: bool
    rise_fraction: float


@dataclass
class SimulationResult:
    labs: list[LabValue]
    exposures: list[ContrastExposure] = field(default_factory=list)

    def series(self, code: str) -> list[LabValue]:
        """Возвращает значения анализа, отсортированные по времени."""
        return sorted((v for v in self.labs if v.code == code), key=lambda v: v.t_h)

    def latest(self, code: str, t_h: float) -> LabValue | None:
        """Возвращает последнее значение анализа на момент t_h."""
        vals = [v for v in self.series(code) if v.t_h <= t_h]
        return vals[-1] if vals else None

    def panel(self, t_h: float) -> list[LabValue]:
        """Возвращает последние значения всех анализов на момент t_h."""
        codes = dict.fromkeys(v.code for v in self.labs)
        return [v for c in codes if (v := self.latest(c, t_h))]


def _active_drugs(items: list[PlanItem], t_h: float) -> list[PlanItem]:
    """Возвращает лечебные назначения, действующие в момент t_h."""
    return [i for i in items if i.active_at(t_h) and (
        i.kind in ("drug", "regimen") or (i.kind == "procedure" and not i.invasive and i.contrast_ml == 0))]


def executed_procedures(plan: TreatmentPlan, horizon_h: float) -> list[PlanItem]:
    """Возвращает процедуры, выполненные до горизонта и не отмененные к началу."""
    return [i for i in plan.items
            if i.kind == "procedure" and i.start_h <= horizon_h and i.active_at(i.start_h)]


def simulate(case: Case, plan: TreatmentPlan, horizon_h: float | None = None) -> SimulationResult:
    """Моделирует анализы пациента до горизонта с учетом процедур, назначений и событий сценария."""
    horizon = case.horizon_h if horizon_h is None else horizon_h
    p = case.patient
    rng = np.random.default_rng(case.seed)
    base = {v.code: v.value for v in p.baseline_labs}
    egfr0 = k.egfr_ckd_epi_2021(base["creatinine"], p.age, p.sex)

    exposures = []
    for proc in executed_procedures(plan, horizon):
        if proc.contrast_ml <= 0:
            continue
        hydration = any(has_class(i.name, "hydration") for i in _active_drugs(plan.items, proc.start_h))
        frac = k.cin_rise_fraction(proc.contrast_ml, egfr0, p.diabetes, p.age, hydration)
        exposures.append(ContrastExposure(proc.item_id, proc.start_h, proc.contrast_ml, egfr0, hydration, frac))
    contrast_events = [(e.t_h, e.rise_fraction) for e in exposures]
    bleeds = [(e.t_h, e.magnitude) for e in case.events if e.type == "gi_bleed"]
    reinfarctions = [(e.t_h, e.magnitude) for e in case.events if e.type == "reinfarction"]

    def noisy(code: str, value: float) -> float:
        sd = NOISE_SD.get(code, 0.0)
        rel = code in ("troponin_hs", "creatinine")
        return value * (1 + rng.normal(0, sd)) if rel else value + rng.normal(0, sd)

    labs: list[LabValue] = []
    for t in (t for t in TROPONIN_TIMES if t <= horizon):
        val = base["troponin_hs"] if t == 0 else noisy(
            "troponin_hs", k.troponin(t, p.symptom_onset_h, base["troponin_hs"], reinfarctions))
        labs.append(LabValue(code="troponin_hs", value=round(val), t_h=t))

    for t in (t for t in DAILY_TIMES if t <= horizon):
        if t == 0:
            labs += [LabValue(code=c, value=v, t_h=0) for c, v in base.items() if c != "troponin_hs"]
            if "egfr" not in base:
                labs.append(LabValue(code="egfr", value=round(egfr0, 1), t_h=0))
            continue
        drugs = _active_drugs(plan.items, t)
        cr = noisy("creatinine", k.creatinine(t, base["creatinine"], contrast_events))
        egfr = k.egfr_ckd_epi_2021(cr, p.age, p.sex)
        n_at = sum(has_class(d.name, ANTITHROMBOTIC) for d in drugs)
        raas = any(has_class(d.name, RAAS) for d in drugs)
        labs += [
            LabValue(code="creatinine", value=round(cr), t_h=t),
            LabValue(code="egfr", value=round(egfr, 1), t_h=t),
            LabValue(code="hb", value=round(noisy("hb", k.hemoglobin(t, base["hb"], n_at, bleeds))), t_h=t),
            LabValue(code="potassium", value=round(noisy("potassium", k.potassium(base["potassium"], egfr, egfr0, raas)), 1), t_h=t),
            LabValue(code="platelets", value=round(noisy("platelets", base["platelets"])), t_h=t),
        ]
        if "glucose" in base:
            labs.append(LabValue(code="glucose", value=round(noisy("glucose", base["glucose"]), 1), t_h=t))
    return SimulationResult(labs=labs, exposures=exposures)
