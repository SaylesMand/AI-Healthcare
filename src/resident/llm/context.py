from __future__ import annotations

from pathlib import Path

from resident.schemas import Chunk, LabValue, Patient, PlanItem
from resident.simulator.kinetics import egfr_ckd_epi_2021

PROMPTS_DIR = Path(__file__).parent / "prompts"

LAB_NAMES = {
    "troponin_hs": "Тропонин (вч)", "creatinine": "Креатинин", "egfr": "рСКФ (CKD-EPI)", "hb": "Гемоглобин",
    "platelets": "Тромбоциты", "potassium": "Калий", "glucose": "Глюкоза", "ldl": "ХС ЛНП",
}


def load_prompt(name: str) -> str:
    """Загружает текст промпта по имени файла."""
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def format_labs(labs: list[LabValue]) -> str:
    """Форматирует список анализов в строку."""
    return "; ".join(f"{LAB_NAMES[l.code]} {l.value:g} {l.unit}" for l in labs)


def format_patient(p: Patient, labs: list[LabValue] | None = None) -> str:
    """Форматирует данные пациента и факторы риска в текст для LLM."""
    sex = "мужчина" if p.sex == "M" else "женщина"
    allergies = "; ".join(f"{a.substance} ({a.reaction})" if a.reaction else a.substance for a in p.allergies)
    egfr = p.lab("egfr") or egfr_ckd_epi_2021(p.lab("creatinine"), p.age, p.sex)
    ckd = ("ХБП нет (рСКФ ≥ 60)" if egfr >= 60 else "рСКФ 45-59 (не умеренная/тяжёлая ХБП)" if egfr >= 45
           else "умеренная/тяжёлая ХБП (рСКФ 15-44)" if egfr >= 15 else "терминальная ХБП (рСКФ < 15)")
    yes_no = lambda v: "да" if v else "нет"
    flags = [
        f"GRACE {p.grace_score}",
        f"ФВ ЛЖ {p.lvef:g}%" if p.lvef is not None else "",
        f"Killip {'I II III IV'.split()[p.killip - 1]}",
        f"ЧСС {p.hr} уд/мин" if p.hr else "",
        f"САД {p.sbp} мм рт. ст." if p.sbp else "",
        f"сердечная недостаточность: {yes_no(p.heart_failure)}",
        f"рСКФ {egfr:.0f} - {ckd}",
        f"критерии очень высокого риска: {yes_no(p.very_high_risk)}",
        f"высокий риск кровотечений (Г2): {yes_no(p.high_bleeding_risk)}",
        f"факторы риска ЖКК (язва/ЖКК): {yes_no(p.gi_bleeding_risk)}",
        f"сахарный диабет: {yes_no(p.diabetes)}",
        f"пожилой возраст (≥ 75): {yes_no(p.age >= 75)}",
        f"противопоказания к тикагрелору/прасугрелу: {yes_no(p.potent_p2y12_contraindicated)}",
    ]
    lines = [
        f"ID: {p.patient_id}; {sex}, {p.age} лет, масса {p.weight_kg:g} кг",
        f"Диагноз: {p.diagnosis} ({p.icd10})",
        f"Анамнез: {p.anamnesis}",
        f"Сопутствующие заболевания: {'; '.join(p.comorbidities) or 'нет'}",
        f"Аллергии: {allergies or 'нет'}",
        f"Принимал до госпитализации: {'; '.join(p.home_medications) or 'нет'}",
        f"Риск: {'; '.join(f for f in flags if f)}",
        f"Анализы при поступлении: {format_labs(p.baseline_labs)}",
    ]
    if labs:
        lines.append(f"Текущие анализы: {format_labs(labs)}")
    return "\n".join(lines)


def format_chunk(c: Chunk) -> str:
    """Форматирует чанк КР в блок chunk с id и ссылкой."""
    body = [f"Раздел: {' / '.join(c.section_path)}", f"Текст: {c.text}"]
    if c.comment:
        body.append(f"Комментарий: {c.comment}")
    return f'<chunk id="{c.chunk_id}" ref="{c.citation}">\n' + "\n".join(body) + "\n</chunk>"


def format_chunks(chunks: list[Chunk]) -> str:
    """Форматирует список чанков для контекста LLM."""
    return "\n\n".join(format_chunk(c) for c in chunks)


def format_plan(items: list[PlanItem]) -> str:
    """Форматирует назначения плана с id, источником и статусом."""
    lines = []
    for i in items:
        dose = ", ".join(x for x in (i.dose, i.route, i.frequency, i.timing) if x)
        status = "действует" if i.active else f"отменено на {i.stop_h:g} ч"
        src = {"planner": "план", "replanner": "пересмотр", "external": "внешнее назначение"}[i.source]
        lines.append(f"- [{i.item_id}] ({i.kind}, {src}, с {i.start_h:g} ч, {status}) {i.name}"
                     + (f": {dose}" if dose else ""))
    return "\n".join(lines)
