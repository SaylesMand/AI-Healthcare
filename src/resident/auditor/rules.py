from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from resident.citations import normalize
from resident.drugs import P2Y12, PARENTERAL_ANTICOAGULANT, classes_of, has_class
from resident.kb.parse import load_chunks
from resident.mandatory import required_groups
from resident.monitor import PLATELET_MIN
from resident.schemas import Case, Chunk, Citation, Finding, PlanItem, TreatmentPlan
from resident.simulator.engine import SimulationResult

KR = "154_4"
LAW_CONSENT = "Федеральный закон № 323-ФЗ «Об основах охраны здоровья граждан в РФ», ст. 20 (ИДС)"


@lru_cache(maxsize=1)
def _kb() -> dict[str, Chunk]:
    """Загружает базу КР один раз и возвращает словарь чанков по id."""
    return {c.chunk_id: c for c in load_chunks()}


def cite(chunk_id: str, phrase: str) -> Citation:
    """Возвращает цитату из чанка КР, если фраза в нем есть, иначе бросает ValueError."""
    chunk = _kb()[f"{KR}:{chunk_id}"]
    if normalize(phrase) not in normalize(chunk.full_text):
        raise ValueError(f"цитата не найдена в {chunk.chunk_id}: {phrase}")
    return Citation(chunk_id=chunk.chunk_id, quote=phrase)


@dataclass
class AuditContext:
    case: Case
    plan: TreatmentPlan
    sim: SimulationResult
    t_end: float

    def egfr_at(self, t_h: float):
        """Возвращает последнюю рСКФ на момент t_h."""
        return self.sim.latest("egfr", t_h)

    def drugs(self) -> list[PlanItem]:
        """Возвращает лечебные назначения: ЛС, режимы и неинвазивные процедуры вроде гидратации."""
        return [i for i in self.plan.items
                if i.kind in ("drug", "regimen") or (i.kind == "procedure" and not i.invasive and i.contrast_ml == 0)]

    def scheduled_procedures(self) -> list[PlanItem]:
        """Возвращает инвазивные процедуры и процедуры с контрастом, не отмененные до начала."""
        return [i for i in self.plan.items
                if i.kind == "procedure" and (i.invasive or i.contrast_ml > 0)
                and (i.stop_h is None or i.stop_h > i.start_h)]

    def overlapping(self, a: PlanItem, b: PlanItem) -> bool:
        """Проверяет, пересекаются ли два назначения по времени действия."""
        a_end = a.stop_h if a.stop_h is not None else self.t_end + 1
        b_end = b.stop_h if b.stop_h is not None else self.t_end + 1
        return max(a.start_h, b.start_h) < min(a_end, b_end)


Rule = Callable[[AuditContext], list[Finding]]
RULES: list[Rule] = []


def rule(fn: Rule) -> Rule:
    """Регистрирует функцию как правило аудита."""
    RULES.append(fn)
    return fn


def _lab_times(ctx: AuditContext, item: PlanItem) -> list[float]:
    """Возвращает моменты измерения рСКФ, когда назначение действовало."""
    return [v.t_h for v in ctx.sim.series("egfr") if item.active_at(v.t_h)]


@rule
def r1_renal_anticoagulant(ctx: AuditContext) -> list[Finding]:
    """Находит антикоагулянты, продолженные при рСКФ ниже порога из КР."""
    out = []
    for item in ctx.drugs():
        for t in _lab_times(ctx, item):
            e = ctx.egfr_at(t).value
            if has_class(item.name, "fondaparinux") and e < 20:
                out.append(Finding(
                    rule_id="R1_renal_anticoagulant", severity="critical", category="clinical", item_ids=[item.item_id],
                    message=f"Фондапаринукс натрия продолжен при рСКФ {e:g} (< 20) на {t:g} ч",
                    citations=[cite("3.2.3.2:t06", "Фондапаринукс натрия противопоказан при рСКФ ниже 20")],
                ))
                break
            if has_class(item.name, "enoxaparin") and e < 30:
                twice = "2 раз" in normalize(f"{item.dose} {item.frequency}")
                if e < 15 or twice:
                    out.append(Finding(
                        rule_id="R1_renal_anticoagulant", severity="critical" if e < 15 else "major",
                        category="clinical", item_ids=[item.item_id],
                        message=(f"Эноксапарин натрия ({item.dose or ''} {item.frequency or ''}) при рСКФ {e:g} на {t:g} ч: "
                                 + ("не рекомендуется при рСКФ < 15" if e < 15 else "при рСКФ < 30 - 1 мг/кг 1 раз в сутки")),
                        citations=[cite("А3:tab04.r11", "у пациентов со скоростью клубочковой фильтрации < 30 мл/мин/1,73 м 2 подкожно 1 мг/кг 1 раз в сутки")]
                        if e >= 15 else [cite("3.2.3.2:t04", "Эноксапарин натрия не рекомендуется при рСКФ ниже 15")],
                    ))
                    break
    return out


@rule
def r2_contrast_renal(ctx: AuditContext) -> list[Finding]:
    """Проверяет процедуры с контрастом: свежую рСКФ, объем контраста и гидратацию при ХБП."""
    out = []
    for proc in ctx.scheduled_procedures():
        if proc.contrast_ml <= 0:
            continue
        before = [v for v in ctx.sim.series("egfr") if v.t_h <= min(proc.start_h, ctx.t_end)]
        last = before[-1] if before else None
        if last is None or proc.start_h - last.t_h > 48:
            out.append(Finding(
                rule_id="R2_contrast_renal", severity="major", category="procedural", item_ids=[proc.item_id],
                message=f"«{proc.name}» ({proc.start_h:g} ч): нет рСКФ за 48 ч до введения контраста",
                citations=[cite("3.1.3:t02", "выполнение КГ и ЧКВ рекомендуются после тщательной оценки соотношения риска и пользы, с учетом выраженности нарушения фильтрационной функции почек")],
            ))
            continue
        e, ratio = last.value, proc.contrast_ml / last.value
        if 15 <= e < 45 and ratio >= 3.7:
            out.append(Finding(
                rule_id="R2_contrast_renal", severity="critical", category="clinical", item_ids=[proc.item_id],
                message=(f"«{proc.name}» ({proc.start_h:g} ч): {proc.contrast_ml:g} мл контраста при рСКФ {e:g} "
                         f"(на {last.t_h:g} ч) - отношение {ratio:.1f} ≥ 3,7"),
                citations=[cite("3.1.3:t03", "рекомендуется ограничить объем вводимого контрастного вещества (соотношение объема контраста к рСКФ < 3,7)")],
            ))
        hydration = any(has_class(d.name, "hydration") and d.active_at(proc.start_h) for d in ctx.drugs())
        if 15 <= e < 45 and not hydration:
            citations = [cite("3.1.3:t03", "У пациентов с умеренной или тяжелой ХБП при инвазивной стратегии лечения ОКСбпST рекомендуется гидратация изотоническим раствором натрия хлорида")]
            if proc.contrast_ml > 100:
                citations.append(cite("3.1.3:t03", "Если ожидаемый объем контрастного вещества во время КГ/ЧКВ превышает 100 мл, рекомендуется гидратация с использованием изотонического раствора натрия хлорида"))
            out.append(Finding(
                rule_id="R2_contrast_renal", severity="major", category="clinical", item_ids=[proc.item_id],
                message=f"«{proc.name}»: рСКФ {e:g} (ХБП 15-44), контраст {proc.contrast_ml:g} мл без гидратации",
                citations=citations,
            ))
    return out


@rule
def r3_consent(ctx: AuditContext) -> list[Finding]:
    """Находит инвазивные процедуры без информированного согласия, полученного до начала."""
    out = []
    for proc in ctx.scheduled_procedures():
        if not proc.invasive:
            continue
        ok = any(c.procedure_id == proc.item_id and c.t_h <= proc.start_h for c in ctx.case.consents)
        if not ok:
            done = "выполнена" if proc.start_h <= ctx.t_end else "запланирована"
            out.append(Finding(
                rule_id="R3_consent", severity="critical", category="procedural", item_ids=[proc.item_id],
                message=f"«{proc.name}» ({done} на {proc.start_h:g} ч): нет информированного добровольного согласия",
                legal_ref=LAW_CONSENT,
            ))
    return out


@rule
def r4_gi_protection(ctx: AuditContext) -> list[Finding]:
    """Находит ДАТТ без ИПП при риске ЖКК и клопидогрел вместе с омепразолом или эзомепразолом."""
    out = []
    drugs = ctx.drugs()
    aspirin = [d for d in drugs if has_class(d.name, "aspirin")]
    p2y12 = [d for d in drugs if has_class(d.name, P2Y12)]
    ppis = [d for d in drugs if has_class(d.name, "ppi")]
    dapt = [(a, p) for a in aspirin for p in p2y12 if ctx.overlapping(a, p)]
    if dapt and ctx.case.patient.gi_bleeding_risk and not ppis:
        a, p = dapt[0]
        out.append(Finding(
            rule_id="R4_gi_protection", severity="major", category="clinical", item_ids=[a.item_id, p.item_id],
            message="ДАТТ у пациента с факторами риска ЖКК без ингибитора протонного насоса",
            citations=[cite("3.2.3.1:t14", "во время двойной антитромбоцитарной терапии для защиты слизистой оболочки желудка и снижения риска желудочно-кишечных кровотечений рекомендуется использовать ингибиторы протонного насоса")],
        ))
    for clop in (d for d in p2y12 if has_class(d.name, "clopidogrel")):
        for ppi in (d for d in ppis if has_class(d.name, "ppi_cyp2c19")):
            if ctx.overlapping(clop, ppi):
                out.append(Finding(
                    rule_id="R4_gi_protection", severity="minor", category="interaction", item_ids=[clop.item_id, ppi.item_id],
                    message=f"Клопидогрел + {ppi.name}: возможное ослабление антитромбоцитарного эффекта; предпочтительны пантопразол или рабепразол",
                    citations=[cite("3.2.3.1:t14", "Известно о возможном ослаблении антитромбоцитарного эффекта клопидогрела при его сочетании с омепразолом или эзомепразолом, но не с пантопразолом или рабепразолом")],
                ))
    return out


@rule
def r5_allergy(ctx: AuditContext) -> list[Finding]:
    """Находит назначения препаратов, на которые или на группу которых у пациента аллергия."""
    out = []
    for allergy in ctx.case.patient.allergies:
        a_classes = classes_of(allergy.substance)
        a_stem = normalize(allergy.substance)[:6]
        for item in ctx.plan.items:
            if (a_classes & classes_of(item.name)) or (len(a_stem) >= 5 and a_stem in normalize(item.name)):
                out.append(Finding(
                    rule_id="R5_allergy", severity="critical", category="clinical", item_ids=[item.item_id],
                    message=f"«{item.name}» при аллергии в анамнезе: {allergy.substance} ({allergy.reaction})",
                ))
    return out


@rule
def r6_nsaid(ctx: AuditContext) -> list[Finding]:
    """Находит назначения НПВП у пациента с ОКСбпST."""
    return [
        Finding(
            rule_id="R6_nsaid", severity="major", category="clinical", item_ids=[d.item_id],
            message=f"НПВП «{d.name}» у пациента с ОКСбпST"
                    + (" (дополнительно нефротоксичен на фоне ОПП)" if _aki(ctx) else ""),
            citations=[
                cite("3.3:t03", "не рекомендуется использовать нестероидные противовоспалительные и противоревматические препараты"),
                cite("3.3:t03", "Рекомендуется отменить нестероидные противовоспалительные и противоревматические средства препараты и/или не начинать их использование при госпитализации с ОКСбпST"),
            ],
        )
        for d in ctx.drugs() if has_class(d.name, "nsaid") and not has_class(d.name, "aspirin")
    ]


def _aki(ctx: AuditContext) -> bool:
    """Проверяет, есть ли у пациента ОПП по динамике креатинина."""
    cr = ctx.sim.series("creatinine")
    return bool(cr) and (cr[-1].value / cr[0].value >= 1.5 or cr[-1].value - cr[0].value >= 26.5)


@rule
def r7_unsupported(ctx: AuditContext) -> list[Finding]:
    """Находит назначения плана, у которых не подтвердилась ссылка на КР."""
    return [
        Finding(
            rule_id="R7_unsupported", severity="major", category="evidence", item_ids=[i.item_id],
            message=f"Необоснованное назначение «{i.name}»: ссылка на КР не прошла проверку ({'; '.join(i.verification_notes)[:200]})",
        )
        for i in ctx.plan.items if i.source in ("planner", "replanner") and i.verification == "unsupported"
    ]


@rule
def r8_monitoring(ctx: AuditContext) -> list[Finding]:
    """Находит отсутствие в плане контроля тропонина и креатинина после контраста."""
    out = []
    labs = [i for i in ctx.plan.items if i.kind == "lab" and i.source != "external"]
    text = " ".join(normalize(f"{i.name} {i.timing or ''} {i.frequency or ''}") for i in labs)
    if not any(k in text for k in ("тропонин",)):
        out.append(Finding(
            rule_id="R8_monitoring", severity="major", category="monitoring",
            message="В плане нет исследования динамики сердечного тропонина",
            citations=[cite("2.3:t01", "рекомендуется исследование динамики уровня сердечного тропонина Т или I")],
        ))
    contrast = [p for p in ctx.scheduled_procedures() if p.contrast_ml > 0]
    if contrast and not any(k in text for k in ("креатинин", "скф", "биохими")):
        out.append(Finding(
            rule_id="R8_monitoring", severity="minor", category="monitoring", item_ids=[p.item_id for p in contrast],
            message="Введение контраста без запланированного контроля креатинина/рСКФ",
            citations=[cite("2.3:t03", "рекомендуется проведение общего (клинического) анализа крови, анализа крови биохимического общетерапевтического")],
        ))
    return out


@rule
def r9_platelets(ctx: AuditContext) -> list[Finding]:
    """Находит антиагреганты при тромбоцитах ниже порога из КР."""
    out = []
    for v in ctx.sim.series("platelets"):
        for d in ctx.drugs():
            for cls, limit in PLATELET_MIN.items():
                if has_class(d.name, cls) and d.active_at(v.t_h) and v.value < limit:
                    out.append(Finding(
                        rule_id="R9_platelets", severity="critical", category="clinical", item_ids=[d.item_id],
                        message=f"{d.name} при тромбоцитах {v.value:g}×10⁹/л (< {limit}) на {v.t_h:g} ч",
                        citations=[cite("3.2.3.1:t15", "не рекомендуется применять ACК при уровне тромбоцитов менее 10х10 9 /л, клопидогрел - менее 30х10 9 /л, тикагрелор или прасугрел - менее 50х10 9 /л")],
                    ))
    return out


@rule
def r12_anticoagulant_overlap(ctx: AuditContext) -> list[Finding]:
    """Находит одновременное назначение НФГ и эноксапарина."""
    out = []
    ac = [d for d in ctx.drugs() if classes_of(d.name) & PARENTERAL_ANTICOAGULANT]
    for i, a in enumerate(ac):
        for b in ac[i + 1:]:
            pair = classes_of(a.name) | classes_of(b.name)
            if {"enoxaparin", "ufh"} <= pair and ctx.overlapping(a, b):
                out.append(Finding(
                    rule_id="R12_anticoagulant_overlap", severity="major", category="interaction",
                    item_ids=[a.item_id, b.item_id],
                    message=f"Одновременно/попеременно НФГ и эноксапарин: «{a.name}» и «{b.name}»",
                    citations=[cite("3.2.3.2:t04", "Переход с НФГ на эноксапарин натрия и с эноксапарина натрия на НФГ не рекомендуется из-за увеличения риска кровотечений")],
                ))
    return out


@rule
def r13_anticoagulant_after_pci(ctx: AuditContext) -> list[Finding]:
    """Находит парентеральные антикоагулянты, продолженные больше 24 ч или начатые после успешного ЧКВ."""
    pcis = [p for p in ctx.scheduled_procedures()
            if p.start_h <= ctx.t_end and ("чкв" in normalize(p.name) or "стент" in normalize(p.name))]
    if not pcis or any("фибрилляц" in normalize(c) or "протез" in normalize(c) for c in ctx.case.patient.comorbidities):
        return []
    t_pci = pcis[0].start_h
    out = []
    for d in ctx.drugs():
        if not classes_of(d.name) & PARENTERAL_ANTICOAGULANT or "профилакт" in normalize(d.rationale):
            continue
        if d.active_at(t_pci + 24):
            what = f"продолжается > 24 ч после успешного ЧКВ ({t_pci:g} ч)"
        elif t_pci < d.start_h <= ctx.t_end:  # в т.ч. замена одного антикоагулянта другим при пересмотре плана
            what = f"начат на {d.start_h:g} ч, после успешного ЧКВ ({t_pci:g} ч)"
        else:
            continue
        out.append(Finding(
            rule_id="R13_anticoagulant_after_pci", severity="major", category="clinical", item_ids=[d.item_id],
            message=f"«{d.name}» {what} без показаний к длительной антикоагуляции",
            citations=[cite("3.2.3.2:t02", "После успешного ЧКВ у пациентов с ОКСбпST рекомендуется прекратить парентеральное введение антикоагулянта из-за опасности увеличения риска кровотечений")],
        ))
    return out


@rule
def r14_p2y12_choice(ctx: AuditContext) -> list[Finding]:
    """Находит клопидогрел у пациента без показаний к нему вместо тикагрелора или прасугрела."""
    p = ctx.case.patient
    has_oac = any(has_class(d.name, "oac") for d in ctx.drugs())
    if p.potent_p2y12_contraindicated or p.high_bleeding_risk or p.age >= 70 or has_oac:
        return []
    return [
        Finding(
            rule_id="R14_p2y12_choice", severity="minor", category="evidence", item_ids=[d.item_id],
            message=f"«{d.name}» выбран без условий тезиса: у пациента нет противопоказаний к тикагрелору/прасугрелу, "
                    "показаний к ОАК, пожилого возраста и высокого риска кровотечений",
            citations=[
                cite("3.2.3.1:t10", "которые не могут получать тикагрелор или прасугрел из-за наличия противопоказаний, непереносимости или недоступности, или нуждаются в длительном лечении антикоагулянтами"),
                cite("3.2.3.1:t08", "рекомендуется в добавление к АСК сроком на 12 месяцев у пациентов с ОКСбпST вне зависимости от стратегии лечения"),
            ],
        )
        for d in ctx.drugs() if has_class(d.name, "clopidogrel") and d.source != "external"
    ]


@rule
def r15_hydration_not_indicated(ctx: AuditContext) -> list[Finding]:
    """Находит гидратацию у пациента без ХБП и сахарного диабета."""
    p = ctx.case.patient
    egfr0 = ctx.sim.series("egfr")[0].value if ctx.sim.series("egfr") else None
    if egfr0 is None or egfr0 < 45 or p.diabetes:
        return []
    return [
        Finding(
            rule_id="R15_hydration_not_indicated", severity="minor", category="evidence", item_ids=[d.item_id],
            message=f"«{d.name}» при рСКФ {egfr0:g} без СД: условие тезиса (умеренная/тяжёлая ХБП, рСКФ 15-44) не выполнено",
            citations=[cite("3.1.3:t03", "У пациентов с умеренной или тяжелой ХБП при инвазивной стратегии лечения ОКСбпST рекомендуется гидратация изотоническим раствором натрия хлорида")],
        )
        for d in ctx.drugs() if has_class(d.name, "hydration") and d.source != "external"
    ]


@rule
def r16_omissions(ctx: AuditContext) -> list[Finding]:
    """Находит отсутствие в плане групп ЛС, которые КР рекомендует этому пациенту (mandatory.required_groups)."""
    # засчитываются только обоснованные назначения плана (внешние и заблокированные - нет)
    given = set().union(*(classes_of(d.name) for d in ctx.drugs()
                          if d.source != "external" and d.verification == "verified"))
    return [
        Finding(rule_id="R16_omission", severity="major", category="clinical",
                message=f"Упущение: в плане нет обоснованного назначения «{g.label}»", citations=[cite(g.ref, g.quote)])
        for g in required_groups(ctx.case.patient) if not g.classes & given
    ]


def run_rules(ctx: AuditContext) -> list[Finding]:
    """Запускает все зарегистрированные правила и возвращает их находки."""
    findings = []
    for fn in RULES:
        findings += fn(ctx)
    return findings
