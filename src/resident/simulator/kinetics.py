from __future__ import annotations

import math


def egfr_ckd_epi_2021(creatinine_umol: float, age: int, sex: str) -> float:
    """Считает рСКФ по формуле CKD-EPI 2021 без учета расы."""
    scr = creatinine_umol / 88.4  # мкмоль/л -> мг/дл
    kappa, alpha = (0.7, -0.241) if sex == "F" else (0.9, -0.302)
    egfr = 142 * min(scr / kappa, 1) ** alpha * max(scr / kappa, 1) ** -1.200 * 0.9938 ** age
    return egfr * (1.012 if sex == "F" else 1.0)


def creatinine_for_egfr(egfr: float, age: int, sex: str) -> float:
    """Подбирает креатинин, дающий заданную рСКФ, методом бисекции."""
    lo, hi = 20.0, 1500.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if egfr_ckd_epi_2021(mid, age, sex) > egfr:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def gamma_pulse(x_h: float, peak_h: float, shape: float = 1.0) -> float:
    """Возвращает форму подъема и спада с максимумом 1 в момент peak_h."""
    if x_h <= 0:
        return 0.0
    r = x_h / peak_h
    return (r * math.exp(1 - r)) ** shape


# --- тропонин -----------------------------------------------------------------
TROPONIN_PEAK_H = 18.0  # пик вч-тропонина при ИМбпST ~12-24 ч от начала симптомов
TROPONIN_BASELINE = 4.0  # нг/л


def troponin(t_h: float, onset_h: float, t0_value: float, reinfarctions: list[tuple[float, float]]) -> float:
    """Считает тропонин в момент t_h по кинетике ИМ и эпизодам повторного ИМ."""
    g0 = gamma_pulse(onset_h, TROPONIN_PEAK_H)
    peak = (t0_value - TROPONIN_BASELINE) / g0 if g0 > 0 else t0_value
    value = TROPONIN_BASELINE + peak * gamma_pulse(t_h + onset_h, TROPONIN_PEAK_H)
    for t_event, magnitude in reinfarctions:
        value += 0.5 * peak * magnitude * gamma_pulse(t_h - t_event, TROPONIN_PEAK_H)
    return value


# --- контраст-индуцированное ОПП ------------------------------------------------
CIN_PEAK_H = 60.0  # пик креатинина через 48-72 ч после введения контраста
CIN_SHAPE = 2.0  # в первые 24 ч прирост небольшой


def cin_rise_fraction(contrast_ml: float, egfr: float, diabetes: bool, age: int, hydration: bool) -> float:
    """Считает пиковый прирост креатинина после контраста по объему, рСКФ, СД, возрасту и гидратации."""
    ratio = contrast_ml / max(egfr, 1.0)
    rise = 0.05 + 0.30 * max(0.0, ratio - 2.5) + (0.30 if diabetes else 0.0) + (0.15 if age >= 75 else 0.0)
    if hydration:
        rise *= 0.5  # гидратация снижает риск КИН примерно вдвое
    return min(rise, 1.5)


def creatinine(t_h: float, baseline: float, contrast_events: list[tuple[float, float]]) -> float:
    """Считает креатинин в момент t_h с учетом эпизодов введения контраста."""
    value = baseline
    for t_c, frac in contrast_events:
        value += baseline * frac * gamma_pulse(t_h - t_c, CIN_PEAK_H, CIN_SHAPE)
    return value


# --- гемоглобин, калий ----------------------------------------------------------

def hemoglobin(t_h: float, baseline: float, n_antithrombotics: int, bleeds: list[tuple[float, float]]) -> float:
    """Считает гемоглобин с учетом антитромботиков и кровотечений."""
    drift = -1.0 * (t_h / 24) * max(0, n_antithrombotics - 1)
    loss = 0.0
    for t_b, magnitude in bleeds:
        if t_h > t_b:
            loss += min(t_h - t_b, 24) / 24 * 25 * magnitude  # ~25 г/л за сутки при magnitude=1
    return baseline + drift - loss


def potassium(baseline: float, egfr_now: float, egfr_baseline: float, raas_blocker: bool) -> float:
    """Считает калий с учетом падения рСКФ и блокаторов РААС."""
    return baseline + 0.03 * max(0.0, egfr_baseline - egfr_now) + (0.2 if raas_blocker else 0.0)
