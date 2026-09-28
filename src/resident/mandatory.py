from __future__ import annotations

from dataclasses import dataclass

from resident.drugs import P2Y12, classes_of
from resident.schemas import Patient


@dataclass(frozen=True)
class RequiredGroup:
    domain: str  # область плана, в которой назначается группа
    classes: frozenset[str]  # классы ЛС из drugs.DRUG_CLASSES, любой из которых закрывает группу
    label: str
    ref: str  # пункт и тезис КР без префикса КР: "3.2.3.1:t01"
    quote: str  # дословная часть тезиса для ссылки аудитора


def required_groups(p: Patient) -> list[RequiredGroup]:
    """Возвращает группы ЛС, которые КР рекомендует этому пациенту при отсутствии противопоказаний."""
    acei_allergy = any("acei" in classes_of(a.substance) for a in p.allergies)
    groups = [
        RequiredGroup("apt", frozenset({"aspirin"}), "АСК", "3.2.3.1:t01",
                      "всем пациентам с ОКСбпST, не имеющим противопоказаний и вне зависимости от стратегии лечения"),
        RequiredGroup("apt", frozenset(P2Y12), "ингибитор P2Y12", "3.2.3.1:t02",
                      "в добавление к АСК рекомендуются ингибиторы"),
        RequiredGroup("base", frozenset({"statin"}), "ингибитор ГМГ-КоА-редуктазы", "3.2.8:t01",
                      "вне зависимости от исходного уровня холестерина (ХС) рекомендуется в период госпитализации "
                      "начать или продолжить"),
        RequiredGroup("base", frozenset({"acei", "arb"}), "АРА (валсартан) - иАПФ противопоказан", "3.2.7:t03",
                      "в случае непереносимости иАПФ для лечения сердечной недостаточности, снижения риска смерти и "
                      "прогрессирования сердечной недостаточности рекомендуется использовать АРА, предпочтительно - "
                      "валсартан")
        if acei_allergy else
        RequiredGroup("base", frozenset({"acei", "arb"}), "иАПФ", "3.2.7:t02",
                      "У всех пациентов с ОКСбпST рекомендуются рутинное применение иАПФ для снижения риска "
                      "сердечно-сосудистых осложнений, если к препаратам этой группы нет противопоказаний"),
    ]
    if p.lvef is not None and p.lvef <= 40:
        groups.append(RequiredGroup(
            "base", frozenset({"beta_blocker"}), "бета-адреноблокатор (ФВ ЛЖ ≤ 40%)", "3.2.5:t02",
            "длительный пероральный прием бета-адреноблокаторов рекомендуется у пациентов с ФВ ЛЖ ≤ 40% вне "
            "зависимости от наличия сердечной недостаточности"))
        # противопоказания из комментария к тезису: креатинин > 220 (м) / > 175 (ж) мкмоль/л, калий > 5 ммоль/л
        creat, k = p.lab("creatinine"), p.lab("potassium")
        renal_ok = creat is None or creat <= (220 if p.sex == "M" else 175)
        if (p.heart_failure or p.diabetes) and renal_ok and (k is None or k <= 5.0):
            groups.append(RequiredGroup(
                "base", frozenset({"mra"}), "антагонист альдостерона (эплеренон)", "3.2.7:t05",
                "к терапевтическим дозам иАПФ и бета-адреноблокаторов рекомендуется добавить антагонист "
                "альдостерона, предпочтительно эплеренон"))
    return groups
