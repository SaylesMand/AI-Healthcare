from resident.kb.parse import parse_grade


def test_all_theses_have_grades(chunks):
    theses = [c for c in chunks if c.kind == "thesis"]
    assert len(theses) >= 200
    assert all(c.uur in {"A", "B", "C"} and c.udd in {"1", "2", "3", "4", "5"} for c in theses)


def test_grade_typos_in_source_are_normalized():
    assert parse_grade("ЕОК IB (УУР 5, УДД С)")[:2] == ("C", "5")  # перепутаны местами
    assert parse_grade("ЕОК IА (УРР С, УДД 5)")[:2] == ("C", "5")  # «УРР»
    assert parse_grade("Уровень убедительности рекомендаций С (уровень достоверности доказательств – 5)")[:2] == ("C", "5")


def test_chunk_ids_unique_and_citation_format(chunks, kb):
    assert len(kb) == len(chunks)
    c = kb["154_4:3.2.3.2:t06"]
    assert "фондапаринукс" in c.text.lower()
    assert "противопоказан при рСКФ ниже 20" in c.comment
    assert c.citation == "[КР 154_4, п. 3.2.3.2, тезис 6; УУР A, УДД 2]"


def test_dosing_table_split_by_rows(kb):
    row = kb["154_4:А3:tab05.r10"]
    assert row.text.splitlines()[-1].startswith("Фондапаринукс")
    assert "рСКФ" in row.text  # шапка таблицы включена в строку
    assert row.citation == "[КР 154_4, п. А3, табл. 5, строка 10]"


def test_bm25_section_filter(bm25):
    hits = bm25.search("фондапаринукс противопоказан при сниженной рСКФ", k=5, sections=["3.2.3.2"])
    assert hits and all(h.chunk.section_num.startswith("3.2.3.2") for h in hits)
    assert hits[0].chunk.chunk_id == "154_4:3.2.3.2:t06"


def test_bm25_abbreviation_expansion(bm25):
    ids = [h.chunk.chunk_id for h in bm25.search("ИПП при ДАТТ", k=5)]
    assert "154_4:3.2.3.1:t14" in ids
