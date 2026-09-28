from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

from resident.config import ABBR_PATH, CHUNKS_PATH, CR_ID, KB_DIR
from resident.kb.fetch import load_snapshot
from resident.schemas import Chunk

# Разделы, которые попадают в базу знаний, и их номер для ссылок.
SECTION_NUMS = {
    "doc_crat_info_1_1": "1.1", "doc_crat_info_1_2": "1.2", "doc_crat_info_1_3": "1.3",
    "doc_crat_info_1_4": "1.4", "doc_crat_info_1_5": "1.5", "doc_crat_info_1_6": "1.6",
    "doc_diag_2": "2", "doc_diag_2_1": "2.1", "doc_diag_2_2": "2.2", "doc_diag_2_3": "2.3",
    "doc_diag_2_4": "2.4", "doc_diag_2_5": "2.5",
    "doc_3": "3", "doc_4": "4", "doc_5": "5", "doc_6": "6",
    "doc_criteria": "Критерии", "doc_a3": "А3", "doc_b": "Б", "doc_g": "Г",
}

MAX_TEXT_CHARS = 1200
MAX_TABLE_CHARS = 2500
CONTINUATION_RE = re.compile(r"^\s*(Плано|Неотлож)", re.I)
_LAT = str.maketrans("АВСаbс", "ABCabc")  # кириллические буквы в оценках -> латиница

# В источнике встречаются опечатки: "УРР С", "(УУР 5, УДД С)" - значения перепутаны местами.
GRADE_RE = re.compile(
    r"(?:УУР|УРР|[Уу]ровень убедительности рекомендаций)\s*[–—-]?\s*([ABCАВС1-5])\b.{0,60}?"
    r"(?:УДД|[Уу]ровень дост\w*верности доказательств)\s*[–—-]?\s*([ABCАВС1-5])\b",
    re.S,
)
EOK_RE = re.compile(r"ЕОК\s*([^()]{1,12}?)\s*\(")
HEADING_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s*(.*)$")
DRUG_RE = re.compile(r"([А-ЯЁа-яё][А-ЯЁа-яё\-]+(?:\s+(?:натрия|калия|кальция|кислота|магния))?)\*\*")
REFS_RE = re.compile(r"\[(\d[\d,\s–\-]*)\]")
COMMENT_PREFIX_RE = re.compile(r"^\s*Комментари[йи]\s*:?\s*")


def clean(text: str) -> str:
    """Убирает метки ЖНВЛП и лишние пробелы из текста."""
    text = text.replace("\xa0", " ").replace("**", "")
    return re.sub(r"\s+", " ", text).strip()


def node_text(node: Tag) -> str:
    """Возвращает очищенный текст HTML-узла."""
    return clean(node.get_text(" "))


def extract_drugs(raw: str) -> list[str]:
    """Извлекает названия препаратов, помеченных в КР как ЖНВЛП."""
    seen: dict[str, None] = {}
    for m in DRUG_RE.finditer(raw.replace("\xa0", " ")):
        name = m.group(1)
        name = name if name.isupper() else name.lower()
        seen.setdefault(name, None)
    return list(seen)


def extract_refs(text: str) -> list[str]:
    """Извлекает номера ссылок на литературу из квадратных скобок."""
    refs: list[str] = []
    for m in REFS_RE.finditer(text):
        refs += [r.strip() for r in m.group(1).split(",") if r.strip()]
    return refs


def parse_grade(text: str) -> tuple[str | None, str | None, str | None]:
    """Извлекает УУР, УДД и класс ЕОК из строки оценок, исправляя опечатки источника."""
    m = GRADE_RE.search(text)
    if not m:
        return None, None, None
    uur, udd = m.group(1).translate(_LAT), m.group(2).translate(_LAT)
    if uur.isdigit() and not udd.isdigit():
        uur, udd = udd, uur
    eok = EOK_RE.search(text)
    eok_val = re.sub(r"\s+", "", eok.group(1)).translate(_LAT) if eok else None
    return uur, udd, eok_val


def is_grade_line(text: str) -> bool:
    """Проверяет, является ли абзац строкой с УУР и УДД."""
    return len(text) < 160 and GRADE_RE.search(text) is not None


def is_comment_para(node: Tag, text: str) -> bool:
    """Проверяет, является ли абзац комментарием к тезису."""
    if COMMENT_PREFIX_RE.match(text):
        return True
    if "marginl" in (node.get("class") or []):
        return True
    em_len = sum(len(e.get_text()) for e in node.find_all("em"))
    return em_len >= 0.5 * max(len(text), 1)


def table_rows(table: Tag) -> list[list[str]]:
    """Возвращает строки HTML-таблицы как списки текстов ячеек."""
    rows = []
    for tr in table.find_all("tr"):
        cells = [node_text(td) for td in tr.find_all(["td", "th"])]
        if any(cells):
            rows.append(cells)
    return rows


@dataclass
class _SectionParser:
    cr_id: str
    section_num: str
    section_title: str
    chunks: list[Chunk] = field(default_factory=list)
    headings: list[tuple[str | None, str]] = field(default_factory=list)
    counters: Counter = field(default_factory=Counter)
    thesis: dict | None = None
    text_buf: list[str] = field(default_factory=list)

    # --- состояние ---
    @property
    def has_theses(self) -> bool:
        """Проверяет, может ли раздел содержать тезисы с УУР и УДД."""
        return self.section_num[0].isdigit()

    @property
    def num(self) -> str:
        """Возвращает номер текущего подраздела."""
        nums = [n for n, _ in self.headings if n]
        return nums[-1] if nums else self.section_num

    @property
    def path(self) -> list[str]:
        """Возвращает путь заголовков от раздела до текущего подраздела."""
        return [self.section_title] + [t for _, t in self.headings]

    def push_heading(self, text: str) -> None:
        """Открывает новый подраздел и сохраняет незавершенные чанки."""
        self.flush()
        m = HEADING_NUM_RE.match(text)
        num = m.group(1) if m and m.group(1).count(".") >= 1 else None
        if num:
            # оставляем только заголовки-предки по нумерации
            self.headings = [(n, t) for n, t in self.headings if n and num.startswith(n + ".")]
        else:
            self.headings = [(n, t) for n, t in self.headings if n]
        self.headings.append((num, text))

    # --- сборка чанков ---
    def _next_id(self, prefix: str) -> tuple[str, int]:
        """Выдает следующий id чанка заданного типа в текущем подразделе."""
        key = (self.num, prefix)
        self.counters[key] += 1
        idx = self.counters[key]
        return f"{self.cr_id}:{self.num}:{prefix}{idx:02d}", idx

    def flush_text(self) -> None:
        """Сохраняет накопленные абзацы как текстовый чанк."""
        if not self.text_buf:
            return
        text = " ".join(self.text_buf)
        self.text_buf = []
        chunk_id, idx = self._next_id("p")
        self.chunks.append(Chunk(
            chunk_id=chunk_id, cr_id=self.cr_id, kind="text", section_num=self.num,
            section_path=self.path, locator=f"абз. {idx}", text=text, refs=extract_refs(text),
        ))

    def flush_thesis(self) -> None:
        """Сохраняет текущий тезис с оценками и комментарием как чанк."""
        if not self.thesis:
            return
        t = self.thesis
        self.thesis = None
        chunk_id, idx = self._next_id("t")
        comment = " ".join(t["comment"])
        self.chunks.append(Chunk(
            chunk_id=chunk_id, cr_id=self.cr_id, kind="thesis", section_num=self.num,
            section_path=self.path, thesis_idx=idx, locator=f"тезис {idx}", text=t["text"], comment=comment,
            uur=t["uur"], udd=t["udd"], eok=t["eok"], drugs=extract_drugs(t["raw"]),
            refs=extract_refs(t["text"] + " " + comment),
        ))

    def flush(self) -> None:
        """Сохраняет незавершенные тезис и абзацы."""
        self.flush_thesis()
        self.flush_text()

    def add_text(self, text: str) -> None:
        """Добавляет абзац в буфер и сохраняет буфер при превышении лимита длины."""
        if self.text_buf and sum(map(len, self.text_buf)) + len(text) > MAX_TEXT_CHARS:
            self.flush_text()
        self.text_buf.append(text)

    # --- обход ---
    def feed(self, node: Tag) -> None:
        """Разбирает очередной HTML-узел раздела."""
        text = node_text(node)
        if not text:
            return
        if node.name in ("h1", "h2", "h3", "h4"):
            self.push_heading(text)
        elif node.name == "ul" and self.has_theses and "рекоменд" in text.lower():
            self.flush()
            self.thesis = {"text": text, "raw": node.get_text(" "), "comment": [],
                           "uur": None, "udd": None, "eok": None}
        elif self.thesis and is_grade_line(text) and self.thesis["uur"] is None:
            self.thesis["uur"], self.thesis["udd"], self.thesis["eok"] = parse_grade(text)
        elif self.thesis and node.name in ("p", "ul", "ol") and (
            self.thesis["uur"] is None or is_comment_para(node, text) or node.name != "p"
        ):
            self.thesis["comment"].append(COMMENT_PREFIX_RE.sub("", text))
            self.thesis["raw"] += " " + node.get_text(" ")
        elif node.name == "table":
            # короткий абзац прямо перед таблицей - её подпись («Таблица 5. …»)
            caption = self.text_buf.pop() if self.text_buf and len(self.text_buf[-1]) < 250 else ""
            self.flush()
            self.add_table(table_rows(node), caption)
        else:
            self.flush_thesis()
            self.add_text(text)

    def add_table(self, rows: list[list[str]], caption: str) -> None:
        """Сохраняет таблицу одним чанком, а большую таблицу разбивает на чанки по строкам."""
        self.counters[(self.num, "tab")] += 1
        tab_no = self.counters[(self.num, "tab")]
        base_id = f"{self.cr_id}:{self.num}:tab{tab_no:02d}"
        body = "\n".join(" | ".join(r) for r in rows)
        if len(body) <= MAX_TABLE_CHARS or len(rows) < 3:
            text = f"{caption}\n{body}".strip()
            self.chunks.append(Chunk(
                chunk_id=base_id, cr_id=self.cr_id, kind="table", section_num=self.num,
                section_path=self.path, locator=f"табл. {tab_no}", text=text, refs=extract_refs(text),
            ))
            return

        header, group, records = " | ".join(rows[0]), "", []
        for row in rows[1:]:
            cells = [c for c in row if c]
            if len(set(cells)) == 1:  # строка-заголовок группы: «Антикоагулянты»
                group = cells[0]
            elif records and CONTINUATION_RE.match(row[0]):  # «Плановое назначение» к предыдущему ЛС
                records[-1][1].append(" | ".join(row))
            else:
                records.append((group, [" | ".join(row)]))
        for i, (grp, lines) in enumerate(records, 1):
            text = "\n".join(filter(None, [caption, header, grp, *lines]))
            self.chunks.append(Chunk(
                chunk_id=f"{base_id}.r{i:02d}", cr_id=self.cr_id, kind="table", section_num=self.num,
                section_path=self.path, locator=f"табл. {tab_no}, строка {i}", text=text,
                refs=extract_refs(text),
            ))


def parse_criteria(cr_id: str, html: str) -> list[Chunk]:
    """Разбирает таблицы критериев качества в чанки по одной строке."""
    soup = BeautifulSoup(html, "lxml")
    chunks, title, tbl_no = [], "Критерии оценки качества медицинской помощи", 0
    for node in soup.body.children:
        if not isinstance(node, Tag):
            continue
        if node.name == "p":
            title = node_text(node)
        elif node.name == "table":
            tbl_no += 1
            for row in table_rows(node)[1:]:  # пропускаем шапку
                if len(row) < 2 or not row[1]:
                    continue
                n = int(re.sub(r"\D", "", row[0]) or 0)
                chunks.append(Chunk(
                    chunk_id=f"{cr_id}:criteria{tbl_no}:r{n:02d}", cr_id=cr_id, kind="criterion",
                    section_num="Критерии", section_path=[title], thesis_idx=n,
                    locator=f"табл. {tbl_no}, критерий {n}", text=row[1],
                ))
    return chunks


def parse_abbreviations(html: str) -> dict[str, str]:
    """Разбирает список сокращений КР в словарь."""
    soup = BeautifulSoup(html, "lxml")
    abbr = {}
    for p in soup.find_all("p"):
        m = re.match(r"^\s*([^–—-]{1,20}?)\s*[–—-]\s*(.+)$", node_text(p))
        if m:
            abbr[m.group(1).strip()] = m.group(2).strip()
    return abbr


def parse_clinrec(clinrec: dict) -> tuple[list[Chunk], dict[str, str]]:
    """Разбирает все разделы КР в чанки и словарь сокращений."""
    cr_id = clinrec["id"]
    chunks: list[Chunk] = []
    abbr: dict[str, str] = {}
    for sec in clinrec["obj"]["sections"]:
        sid, html = sec["id"], sec["content"] or ""
        if sid == "doc_abbreviation":
            abbr = parse_abbreviations(html)
        if sid not in SECTION_NUMS or not html.strip():
            continue
        if sid == "doc_criteria":
            chunks += parse_criteria(cr_id, html)
            continue
        parser = _SectionParser(cr_id, SECTION_NUMS[sid], clean(sec["title"]))
        soup = BeautifulSoup(html, "lxml")
        for node in soup.body.children:
            if isinstance(node, Tag):
                parser.feed(node)
        parser.flush()
        chunks += parser.chunks
    return chunks, abbr


def load_chunks(path=CHUNKS_PATH) -> list[Chunk]:
    """Загружает чанки базы знаний из jsonl."""
    with open(path, encoding="utf-8") as f:
        return [Chunk.model_validate_json(line) for line in f if line.strip()]


def main() -> None:
    """Разбирает снимок КР и сохраняет чанки и сокращения в data/kb."""
    clinrec = load_snapshot(CR_ID)["clinrec"]
    chunks, abbr = parse_clinrec(clinrec)
    ids = Counter(c.chunk_id for c in chunks)
    dupes = [k for k, v in ids.items() if v > 1]
    assert not dupes, f"duplicate chunk ids: {dupes[:5]}"

    KB_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
    ABBR_PATH.write_text(json.dumps(abbr, ensure_ascii=False, indent=1), encoding="utf-8")

    by_kind = Counter(c.kind for c in chunks)
    no_grade = sum(1 for c in chunks if c.kind == "thesis" and not c.uur)
    per_section = defaultdict(int)
    for c in chunks:
        if c.kind == "thesis":
            per_section[c.section_num] += 1
    print(f"chunks: {len(chunks)} {dict(by_kind)}; theses without grade: {no_grade}; abbreviations: {len(abbr)}")
    print("theses per section:", dict(per_section))


if __name__ == "__main__":
    main()
