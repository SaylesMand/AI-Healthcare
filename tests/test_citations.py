from resident.citations import dose_supported, quote_in_chunk, verify_item
from resident.schemas import Citation, PlanItemDraft


def _item(name, cid, quote, dose=None, kind="drug"):
    return PlanItemDraft(kind=kind, name=name, dose=dose, rationale="",
                         citations=[Citation(chunk_id=cid, quote=quote)])


def test_exact_and_normalized_quote(kb):
    c = kb["154_4:3.2.3.1:t08"]
    assert quote_in_chunk("Тикагрелор (нагрузочная доза 180 мг внутрь", c)
    assert quote_in_chunk("тикагрелор   (НАГРУЗОЧНАЯ доза 180 мг внутрь", c)  # регистр и пробелы
    assert not quote_in_chunk("тикагрелор 360 мг однократно", c)
    assert not quote_in_chunk("тикагрелор", c)  # слишком короткая цитата


def test_verify_item_ok(kb):
    allowed = {"154_4:3.2.3.1:t08": kb["154_4:3.2.3.1:t08"]}
    item = _item("Тикагрелор", "154_4:3.2.3.1:t08", "Тикагрелор (нагрузочная доза 180 мг внутрь", dose="180 мг")
    assert verify_item(item, allowed) == (True, [])


def test_verify_rejects_chunk_outside_context(kb):
    item = _item("Тикагрелор", "154_4:3.2.3.1:t08", "Тикагрелор (нагрузочная доза 180 мг внутрь")
    ok, notes = verify_item(item, allowed={})
    assert not ok and "не передавался" in notes[0]


def test_verify_rejects_invented_dose_and_wrong_drug(kb):
    allowed = {"154_4:3.2.3.1:t08": kb["154_4:3.2.3.1:t08"]}
    ok, notes = verify_item(_item("Тикагрелор", "154_4:3.2.3.1:t08",
                                  "Тикагрелор (нагрузочная доза 180 мг внутрь", dose="360 мг"), allowed)
    assert not ok and any("доза" in n for n in notes)
    ok, notes = verify_item(_item("Прасугрел", "154_4:3.2.3.1:t08",
                                  "Тикагрелор (нагрузочная доза 180 мг внутрь"), allowed)
    assert not ok and any("не упоминается" in n for n in notes)


def test_dose_numbers(kb):
    assert dose_supported("1 мг/кг 1 раз в сутки", [kb["154_4:А3:tab04.r11"]])
    assert not dose_supported("3,5 мг/кг", [kb["154_4:А3:tab04.r11"]])


def test_paraphrase_is_aligned_to_verbatim_sentence(kb):
    from resident.citations import check_citations

    cid = "154_4:2.3:t07"
    cit = Citation(chunk_id=cid, quote="У всех пациентов с ОКСбпST рекомендуется определение концентрации "
                                       "электролитов крови (минимально - калий, натрий)")
    ok, problems, info = check_citations([cit], {cid: kb[cid]})
    assert ok and not problems and "выровнена" in info[0]
    assert cit.quote in kb[cid].full_text  # в плане остаётся дословный текст КР


def test_invented_statement_is_not_aligned(kb):
    from resident.citations import check_citations

    cid = "154_4:2.5.1:t01"
    cit = Citation(chunk_id=cid, quote="Коронарография должна быть выполнена в первые 24 часа после "
                                       "госпитализации у пациентов с высоким риском неблагоприятного исхода.")
    ok, problems, _ = check_citations([cit], {cid: kb[cid]})
    assert not ok and problems


def test_drug_classes_recognize_guideline_drugs():
    from resident.drugs import DRUG_CLASSES, classes_of

    assert all(not any(ord(ch) < 32 for ch in rx) for rx in DRUG_CLASSES.values())  # нет управляющих символов
    for name in ("Бисопролол", "Метопролол", "Карведилол"):  # бета-адреноблокаторы из п. 3.2.5
        assert "beta_blocker" in classes_of(name)
    assert classes_of("Эплеренон") == {"mra"} and classes_of("Валсартан") == {"arb"}
