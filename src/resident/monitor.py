from __future__ import annotations

from resident.drugs import has_class
from resident.schemas import Case, Deviation, TreatmentPlan
from resident.simulator.engine import SimulationResult

KR = "154_4"
PLATELET_MIN = {  # КР п. 3.2.3.1, тезис 15
    "aspirin": 10, "clopidogrel": 30, "ticagrelor": 50, "prasugrel": 50,
}


def detect_deviations(case: Case, plan: TreatmentPlan, sim: SimulationResult, t_now: float) -> list[Deviation]:
    """Находит отклонения анализов на контрольной точке по порогам из КР и KDIGO."""
    out: list[Deviation] = []
    active = plan.active_items(t_now)
    cr = [v for v in sim.series("creatinine") if v.t_h <= t_now]
    egfr = sim.latest("egfr", t_now)

    # 1. ОПП по KDIGO 2012: +26,5 мкмоль/л за 48 ч или ≥1,5× от исходного
    if cr:
        base = cr[0].value
        rise48 = max((b.value - a.value for a in cr for b in cr if 0 < b.t_h - a.t_h <= 48), default=0)
        ratio = cr[-1].value / base
        if rise48 >= 26.5 or ratio >= 1.5:
            stage = 3 if ratio >= 3 else 2 if ratio >= 2 else 1
            recent_contrast = [e for e in sim.exposures if 0 < t_now - e.t_h <= 96]
            out.append(Deviation(
                code="aki", t_h=t_now, severity="critical" if stage >= 2 else "warning",
                message=(f"Острое повреждение почек, стадия {stage}: креатинин {base:g} → {cr[-1].value:g} мкмоль/л "
                         f"(×{ratio:.2f}), рСКФ {egfr.value:g}"
                         + (f"; {recent_contrast[0].contrast_ml:g} мл контраста на {recent_contrast[0].t_h:g} ч "
                            f"- вероятна контраст-индуцированная нефропатия" if recent_contrast else "")),
                values={"creatinine_base": base, "creatinine": cr[-1].value, "ratio": round(ratio, 2),
                        "egfr": egfr.value},
                source="KDIGO 2012 (критерии ОПП; вне КР) + КР 154_4 п. 3.1.3",
                chunk_ids=[f"{KR}:3.1.3:t01", f"{KR}:3.1.3:t03"],
            ))

    # 2. рСКФ и антикоагулянты (КР п. А3, табл. 4-5; п. 3.2.3.2)
    if egfr:
        e = egfr.value
        for item in active:
            if has_class(item.name, "fondaparinux") and e < 20:
                out.append(Deviation(
                    code="egfr_fondaparinux", t_h=t_now, severity="critical", item_ids=[item.item_id],
                    message=f"рСКФ {e:g} < 20: фондапаринукс натрия противопоказан ({item.item_id})",
                    values={"egfr": e}, source="КР 154_4 п. 3.2.3.2, тезис 6; п. А3, табл. 5",
                    chunk_ids=[f"{KR}:3.2.3.2:t06", f"{KR}:3.2.3.2:t07", f"{KR}:А3:tab05.r10"],
                ))
            if has_class(item.name, "enoxaparin") and e < 30:
                out.append(Deviation(
                    code="egfr_enoxaparin", t_h=t_now, severity="critical" if e < 15 else "warning",
                    item_ids=[item.item_id],
                    message=(f"рСКФ {e:g}: эноксапарин натрия "
                             + ("не рекомендуется (< 15)" if e < 15 else "требует коррекции - 1 мг/кг 1 раз в сутки (< 30)")),
                    values={"egfr": e}, source="КР 154_4 п. А3, табл. 4 (строка 11) и табл. 5",
                    chunk_ids=[f"{KR}:А3:tab04.r11", f"{KR}:А3:tab05.r09", f"{KR}:3.2.3.2:t07"],
                ))

        # 3. запланированные процедуры с контрастом при сниженной рСКФ (п. 3.1.3)
        for item in plan.items:
            if item.kind == "procedure" and item.contrast_ml > 0 and item.start_h > t_now and item.active:
                ratio = item.contrast_ml / e
                if e < 45 and ratio >= 3.7:
                    out.append(Deviation(
                        code="planned_contrast_risk", t_h=t_now, severity="critical", item_ids=[item.item_id],
                        message=(f"Запланировано «{item.name}» на {item.start_h:g} ч с {item.contrast_ml:g} мл контраста "
                                 f"при рСКФ {e:g}: отношение {ratio:.1f} ≥ 3,7"),
                        values={"egfr": e, "contrast_ml": item.contrast_ml, "ratio": round(ratio, 1)},
                        source="КР 154_4 п. 3.1.3, тезис 3",
                        chunk_ids=[f"{KR}:3.1.3:t02", f"{KR}:3.1.3:t03"],
                    ))

    # 4. Гемоглобин (п. 3.3, тезис 4)
    hb = sim.series("hb")
    if hb:
        hb0, hb_now = hb[0].value, sim.latest("hb", t_now).value
        if hb_now < 70 or hb0 - hb_now >= 20:
            out.append(Deviation(
                code="hb_drop", t_h=t_now, severity="critical" if hb_now < 80 else "warning",
                message=f"Снижение гемоглобина {hb0:g} → {hb_now:g} г/л - исключить кровотечение",
                values={"hb_base": hb0, "hb": hb_now}, source="КР 154_4 п. 3.3, тезис 4; п. 3.2.3.1",
                chunk_ids=[f"{KR}:3.3:t04", f"{KR}:3.2.3.1:t14", f"{KR}:3.2.3.1:t17"],
            ))

    # 5. Калий (контроль электролитов - п. 2.3, тезис 7)
    k = sim.latest("potassium", t_now)
    if k and k.value >= 5.5:
        out.append(Deviation(
            code="hyperkalemia", t_h=t_now, severity="critical" if k.value >= 6.0 else "warning",
            message=f"Гиперкалиемия {k.value:g} ммоль/л",
            values={"potassium": k.value}, source="КР 154_4 п. 2.3, тезис 7 (контроль электролитов)",
            chunk_ids=[f"{KR}:2.3:t07", f"{KR}:3.2.7:t04"],
        ))

    # 6. Тромбоциты и антиагреганты (п. 3.2.3.1, тезис 15)
    plt = sim.latest("platelets", t_now)
    if plt:
        for item in active:
            for cls, limit in PLATELET_MIN.items():
                if has_class(item.name, cls) and plt.value < limit:
                    out.append(Deviation(
                        code="thrombocytopenia", t_h=t_now, severity="critical", item_ids=[item.item_id],
                        message=f"Тромбоциты {plt.value:g}×10⁹/л < {limit}: {item.name} не рекомендуется",
                        values={"platelets": plt.value}, source="КР 154_4 п. 3.2.3.1, тезис 15",
                        chunk_ids=[f"{KR}:3.2.3.1:t15"],
                    ))

    # 7. Повторный подъём тропонина после снижения (п. 3.4.6)
    tn = [v for v in sim.series("troponin_hs") if v.t_h <= t_now]
    if len(tn) >= 3:
        nadir = min(tn[1:-1], key=lambda v: v.value)
        peak_before = max(v.value for v in tn if v.t_h < nadir.t_h) if nadir.t_h > tn[0].t_h else nadir.value
        if nadir.value < peak_before and tn[-1].value >= 1.2 * nadir.value:
            out.append(Deviation(
                code="reinfarction", t_h=t_now, severity="critical",
                message=f"Повторный подъём тропонина: {nadir.value:g} → {tn[-1].value:g} нг/л",
                values={"troponin_nadir": nadir.value, "troponin": tn[-1].value},
                source="КР 154_4 п. 3.4.6 (рецидив ИМ)",
                chunk_ids=[f"{KR}:3.4.6:t01", f"{KR}:3.4.6:t02"],
            ))
    return out
