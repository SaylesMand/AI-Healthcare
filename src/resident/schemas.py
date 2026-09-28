from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --- База знаний -------------------------------------------------------------

class Chunk(BaseModel):
    """Единица базы знаний: тезис КР, абзац пояснений или строка таблицы."""

    chunk_id: str  # "154_4:3.2.3.2:t02"
    cr_id: str
    kind: Literal["thesis", "text", "table", "criterion"]
    section_num: str  # "3.2.3.2", "2.3", "А3", "criteria"
    section_path: list[str]  # заголовки от раздела до подраздела
    thesis_idx: int | None = None  # порядковый номер тезиса внутри подраздела
    locator: str = ""  # место внутри пункта: "тезис 2", "табл. 5, строка 3", "абз. 1"
    text: str  # сам тезис / абзац
    comment: str = ""  # «Комментарии» к тезису
    uur: str | None = None  # уровень убедительности рекомендаций: A/B/C
    udd: str | None = None  # уровень достоверности доказательств: 1-5
    eok: str | None = None  # класс/уровень ЕОК: "IA", "IIaB"
    drugs: list[str] = Field(default_factory=list)  # ЛС, помеченные ** (ЖНВЛП)
    refs: list[str] = Field(default_factory=list)  # номера источников литературы

    @property
    def full_text(self) -> str:
        """Возвращает текст тезиса вместе с комментарием."""
        return f"{self.text}\n{self.comment}".strip()

    @property
    def citation(self) -> str:
        """Возвращает ссылку на чанк вида [КР 154_4, п. 3.2.3.2, тезис 2; УУР A, УДД 2]."""
        parts = [f"КР {self.cr_id}", f"п. {self.section_num}"]
        if self.locator:
            parts.append(self.locator)
        ref = ", ".join(parts)
        if self.uur:
            ref += f"; УУР {self.uur}, УДД {self.udd}"
        return f"[{ref}]"


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    rank: int


# --- Пациент -------------------------------------------------------------------
# Время везде - часы от госпитализации (t_h); сутки N начинаются в t_h = 24*N.

LabCode = Literal["troponin_hs", "creatinine", "egfr", "hb", "platelets", "potassium", "glucose", "ldl"]
LAB_UNITS: dict[str, str] = {
    "troponin_hs": "нг/л", "creatinine": "мкмоль/л", "egfr": "мл/мин/1,73 м²", "hb": "г/л",
    "platelets": "×10⁹/л", "potassium": "ммоль/л", "glucose": "ммоль/л", "ldl": "ммоль/л",
}


class LabValue(BaseModel):
    code: LabCode
    value: float
    t_h: float = 0.0

    @property
    def unit(self) -> str:
        """Возвращает единицу измерения анализа."""
        return LAB_UNITS[self.code]


class Allergy(BaseModel):
    substance: str  # МНН или группа: "ацетилсалициловая кислота", "йодсодержащие контрастные"
    reaction: str = ""


class Patient(BaseModel):
    patient_id: str
    age: int
    sex: Literal["M", "F"]
    weight_kg: float
    diagnosis: str
    icd10: str
    anamnesis: str  # жалобы и анамнез в свободной форме
    comorbidities: list[str] = Field(default_factory=list)
    allergies: list[Allergy] = Field(default_factory=list)
    home_medications: list[str] = Field(default_factory=list)
    grace_score: int
    very_high_risk: bool = False  # критерии очень высокого риска (п. 3.1.1)
    high_bleeding_risk: bool = False  # шкала Г2
    gi_bleeding_risk: bool = False  # язва/ЖКК в анамнезе и др. (п. 3.2.3.1)
    potent_p2y12_contraindicated: bool = False  # противопоказания к тикагрелору/прасугрелу
    diabetes: bool = False
    symptom_onset_h: float = 4.0  # сколько часов от начала симптомов до госпитализации
    lvef: float | None = None  # ФВ ЛЖ по ЭхоКГ, %
    killip: int = 1  # класс ОСН по Killip при поступлении
    hr: int | None = None  # ЧСС при поступлении, уд/мин
    sbp: int | None = None  # систолическое АД при поступлении, мм рт. ст.
    heart_failure: bool = False  # сердечная недостаточность (ХСН или признаки СН)
    baseline_labs: list[LabValue]

    def lab(self, code: str) -> float | None:
        """Возвращает последнее значение анализа при поступлении."""
        vals = [v for v in self.baseline_labs if v.code == code]
        return vals[-1].value if vals else None


# --- План лечения -------------------------------------------------------------

ItemKind = Literal["lab", "instrumental", "drug", "procedure", "regimen"]


class Citation(BaseModel):
    chunk_id: str  # идентификатор чанка из переданного контекста
    quote: str  # дословная цитата из этого чанка


class PlanItemDraft(BaseModel):
    """Назначение в том виде, в каком его возвращает LLM."""

    kind: ItemKind
    applicability: str = ""  # какое условие тезиса КР выполнено у пациента (рассуждение до выбора)
    name: str  # МНН препарата / название исследования или процедуры
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    timing: str | None = None  # "в первые 24 ч", "ежедневно 3 дня"
    rationale: str
    citations: list[Citation]


class PlanItem(PlanItemDraft):
    item_id: str
    start_h: float = 0.0
    stop_h: float | None = None
    source: Literal["planner", "replanner", "external"] = "planner"
    ordered_by: str = ""  # для внешних назначений: «дежурный врач приёмного отделения»
    contrast_ml: float = 0.0  # для процедур с контрастом
    invasive: bool = False  # процедура из расписания, требующая ИДС
    verification: Literal["verified", "unsupported", "not_checked"] = "not_checked"
    verification_notes: list[str] = Field(default_factory=list)

    @property
    def active(self) -> bool:
        """Проверяет, что назначение не отменено."""
        return self.stop_h is None

    def active_at(self, t_h: float) -> bool:
        """Проверяет, действует ли назначение в момент t_h."""
        return self.start_h <= t_h and (self.stop_h is None or t_h < self.stop_h)


class PlanDraft(BaseModel):
    """Ответ LLM для одной клинической области плана."""

    items: list[PlanItemDraft]


class TreatmentPlan(BaseModel):
    patient_id: str
    items: list[PlanItem] = Field(default_factory=list)

    def active_items(self, t_h: float | None = None) -> list[PlanItem]:
        """Возвращает назначения, действующие сейчас или в момент t_h."""
        return [i for i in self.items if (i.active if t_h is None else i.active_at(t_h))]

    def get(self, item_id: str) -> PlanItem:
        """Возвращает назначение по id."""
        return next(i for i in self.items if i.item_id == item_id)


# --- Сценарий и трек ----------------------------------------------------------

class Procedure(BaseModel):
    """Процедура в расписании, выполненная до контрольной точки или запланированная после."""

    procedure_id: str
    t_h: float
    name: str  # "КГ + ЧКВ со стентированием"
    ordered_by: str = "рентгенэндоваскулярный хирург"
    invasive: bool = True
    contrast_ml: float = 0.0


class Consent(BaseModel):
    t_h: float
    procedure_id: str  # id процедуры в плане, на которую получено согласие
    procedure: str  # формулировка в бланке ИДС


class ScenarioEvent(BaseModel):
    t_h: float
    type: Literal["gi_bleed", "reinfarction"]
    magnitude: float = 1.0


class Case(BaseModel):
    case_id: str
    title: str
    pattern: Literal["baseline", "comorbid", "reactive"]
    patient: Patient
    seed: int = 0
    horizon_h: float = 72.0  # контрольная точка - 3-и сутки
    procedures: list[Procedure] = Field(default_factory=list)
    consents: list[Consent] = Field(default_factory=list)
    events: list[ScenarioEvent] = Field(default_factory=list)
    # «чужие» назначения (приёмное отделение, амбулаторный этап): часть из них - заложенные дефекты
    external_orders: list[PlanItem] = Field(default_factory=list)
    expected_findings: list[str] = Field(default_factory=list)  # rule_id, которые аудитор должен найти


class Deviation(BaseModel):
    code: str  # "aki", "egfr_below_20", "hb_drop", …
    t_h: float
    severity: Literal["warning", "critical"]
    message: str
    values: dict[str, float] = Field(default_factory=dict)
    source: str  # откуда порог: "КР 154_4, п. А3, табл. 5" / "KDIGO 2012"
    chunk_ids: list[str] = Field(default_factory=list)  # пункты КР, задающие порог
    item_ids: list[str] = Field(default_factory=list)  # назначения, которых касается отклонение


class StopOrder(BaseModel):
    item_id: str
    reason: str
    citations: list[Citation]


class PlanChangeDraft(BaseModel):
    """Ответ LLM при пересмотре плана."""

    stop: list[StopOrder]
    add: list[PlanItemDraft]
    rationale: str


class PlanChange(BaseModel):
    t_h: float
    deviations: list[Deviation]
    stopped: list[StopOrder]
    added: list[PlanItem]
    rationale: str


# --- Аудит --------------------------------------------------------------------

Severity = Literal["critical", "major", "minor"]
FindingCategory = Literal["clinical", "procedural", "interaction", "evidence", "monitoring"]


class Finding(BaseModel):
    rule_id: str  # "R1_renal_anticoagulant" или "LLM"
    severity: Severity
    category: FindingCategory
    item_ids: list[str] = Field(default_factory=list)
    message: str
    citations: list[Citation] = Field(default_factory=list)
    legal_ref: str | None = None  # для процедурных норм вне КР (323-ФЗ)
    source: Literal["rules", "llm"] = "rules"
    verified: bool = True  # для LLM - прошли ли ссылки проверку
    review: str | None = None  # для LLM - вывод проверяющей модели


class LLMFindingDraft(BaseModel):
    evidence: str  # факт из трека/анализов, на котором основана находка (id назначения, значение, время)
    severity: Severity
    category: FindingCategory
    item_ids: list[str]
    message: str
    citations: list[Citation]


class LLMAuditDraft(BaseModel):
    findings: list[LLMFindingDraft]


class FindingVerdict(BaseModel):
    """Вердикт проверяющей модели по одной находке LLM-аудитора: сначала обоснование, потом решение."""

    index: int  # номер находки в переданном списке
    reason: str  # что сказано в тезисе и что есть в треке
    condition_met: bool  # условия процитированного тезиса выполнены у этого пациента
    fact_correct: bool  # факт из трека указан верно


class VerdictDraft(BaseModel):
    verdicts: list[FindingVerdict]


class AuditReport(BaseModel):
    case_id: str
    findings: list[Finding]
    rejected: list[Finding] = Field(default_factory=list)  # находки LLM, отклоненные проверяющей моделью

    def by_severity(self) -> dict[str, int]:
        """Считает находки по степени тяжести."""
        out = {"critical": 0, "major": 0, "minor": 0}
        for f in self.findings:
            out[f.severity] += 1
        return out
