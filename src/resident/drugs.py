from __future__ import annotations

import re

DRUG_CLASSES: dict[str, str] = {
    "aspirin": r"ацетилсалицил|\bаск\b|аспирин",
    "clopidogrel": r"клопидогрел",
    "ticagrelor": r"тикагрелор",
    "prasugrel": r"прасугрел",
    "ufh": r"нефракционир|\bнфг\b|^гепарин",
    "enoxaparin": r"эноксапарин",
    "fondaparinux": r"фондапаринукс",
    "bivalirudin": r"бивалирудин",
    "oac": r"ривароксабан|апиксабан|дабигатран|варфарин|витамина к",
    "ppi": r"празол",
    "ppi_cyp2c19": r"\bомепразол|эзомепразол",
    "nsaid": r"кеторолак|диклофенак|ибупрофен|нимесулид|мелоксикам|кетопрофен|напроксен|индометацин|нпвп|нестероидн",
    "acei": r"\w+прил\b|иапф|ингибитор\w* апф",
    "arb": r"\w+сартан",
    "hydration": r"гидратац|натрия хлорид|изотонич",
    "statin": r"статин|гмг-коа|аторвастатин|розувастатин|симвастатин",
    "beta_blocker": r"\w+олол\b|карведилол|бета-адреноблок",
    "mra": r"эплеренон|спиронолактон|антагонист\w* альдостерона|антагонист\w* минералокортик",
    "contrast": r"коронарограф|\bкг\b|чкв|стентир|контраст",
}
P2Y12 = {"clopidogrel", "ticagrelor", "prasugrel"}
ANTIPLATELET = P2Y12 | {"aspirin"}
PARENTERAL_ANTICOAGULANT = {"ufh", "enoxaparin", "fondaparinux", "bivalirudin"}
ANTITHROMBOTIC = ANTIPLATELET | PARENTERAL_ANTICOAGULANT | {"oac"}
RAAS = {"acei", "arb", "mra"}  # повышают калий
# группы, в которых одновременно допустим только один препарат
EXCLUSIVE_GROUPS = {"ингибитор P2Y12": P2Y12, "парентеральный антикоагулянт": PARENTERAL_ANTICOAGULANT}

_COMPILED = {k: re.compile(v, re.I) for k, v in DRUG_CLASSES.items()}


def classes_of(name: str) -> set[str]:
    """Возвращает фармгруппы, к которым относится название препарата."""
    text = name.lower().replace("ё", "е")
    return {k for k, rx in _COMPILED.items() if rx.search(text)}


def has_class(name: str, cls: str | set[str]) -> bool:
    """Проверяет, относится ли препарат к одной из указанных фармгрупп."""
    wanted = {cls} if isinstance(cls, str) else cls
    return bool(classes_of(name) & wanted)
