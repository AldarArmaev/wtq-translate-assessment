"""Deterministic table checks: T1 structure, T5 immutable values, preparation of cells for the LLM judge."""
from __future__ import annotations

import random
import re
from typing import Iterable, Optional

from loaders import Table

# ----------------------------------------------------------------------------- classification
URL_RE = re.compile(r"^(?:https?://|www\.)\S+$|^[\w.+-]+@[\w-]+\.[\w.-]+$", re.I)
CODE_RE = re.compile(r"^(?=.*\d)[A-Z0-9]+(?:[-_/.#:][A-Z0-9]+)*$")   # IDs like "KV62", "B-52"

_EN = {"january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
       "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
       "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12}
_RU = {"январь": 1, "января": 1, "янв": 1, "февраль": 2, "февраля": 2, "фев": 2, "февр": 2,
       "март": 3, "марта": 3, "мар": 3, "апрель": 4, "апреля": 4, "апр": 4, "май": 5, "мая": 5,
       "июнь": 6, "июня": 6, "июн": 6, "июль": 7, "июля": 7, "июл": 7, "август": 8, "августа": 8, "авг": 8,
       "сентябрь": 9, "сентября": 9, "сен": 9, "сент": 9, "октябрь": 10, "октября": 10, "окт": 10,
       "ноябрь": 11, "ноября": 11, "ноя": 11, "нояб": 11, "декабрь": 12, "декабря": 12, "дек": 12}
_IGNORE = {"года", "год", "г", "th", "st", "nd", "rd", "of", "the", "числа"}
_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—―−"})

IMMUTABLE_KINDS = {"url", "numeric", "symbol", "date", "code"}


def parse_date(s: str):
    """(years, month, days) for dates that contain a month name (en or ru); None otherwise."""
    month, years, days = None, [], []
    for t in re.findall(r"\d+|[^\W\d_]+", s.lower()):
        if t.isdigit():
            if len(t) == 4:
                years.append(int(t))
            elif len(t) <= 2 and 1 <= int(t) <= 31:
                days.append(int(t))
            else:
                return None
        elif t in _EN or t in _RU:
            m = _EN.get(t) or _RU.get(t)
            if month is not None and month != m:
                return None
            month = m
        elif t in _IGNORE:
            continue
        else:
            return None
    if month is None or not (years or days):
        return None
    return (tuple(years), month, tuple(days))


def classify_cell(text: str) -> str:
    """empty | url | numeric | symbol | date | code | text  (the first five except empty are 'immutable')."""
    if text == "":
        return "empty"
    if URL_RE.match(text):
        return "url"
    has_alpha = any(ch.isalpha() for ch in text)
    has_digit = any(ch.isdigit() for ch in text)
    if not has_alpha:
        return "numeric" if has_digit else "symbol"
    if parse_date(text) is not None:
        return "date"
    if CODE_RE.match(text):
        return "code"
    return "text"


def is_text_cell(cell: dict) -> bool:
    return cell["kind"] in ("text", "empty")


# ----------------------------------------------------------------------------- T5 immutable values
def _numbers(s: str) -> list[str]:
    s = re.sub(r"(?<=\d)[ \u00a0\u202f,.](?=\d{3}(?!\d))", "", s)   # thousands separators
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)                            # decimal comma
    return re.findall(r"\d+(?:\.\d+)?", s)


def check_immutable(src: str, tr: Optional[str], kind: str) -> dict:
    """ok = strict (exact string, or same parsed date); ok_lenient also tolerates number/dash formatting."""
    if tr is None:
        return {"kind": kind, "ok": 0, "ok_lenient": 0, "reason": "missing_cell"}
    exact = int(src == tr)
    if kind == "date":
        ok = int(parse_date(tr) == parse_date(src))
        return {"kind": kind, "ok": ok, "ok_lenient": ok, "reason": "" if ok else "date_mismatch"}
    lenient = exact
    if kind == "numeric":
        ns = _numbers(src)
        lenient = int(exact or (bool(ns) and not any(c.isalpha() for c in tr) and ns == _numbers(tr)))
    elif kind == "symbol":
        lenient = int(exact or src.translate(_DASHES).replace(" ", "") == tr.translate(_DASHES).replace(" ", ""))
    return {"kind": kind, "ok": exact, "ok_lenient": lenient, "reason": "" if exact else "value_changed"}


# ----------------------------------------------------------------------------- T1 structure
def compare_structure(src: Table, tr: Table) -> dict:
    same_rows = src.n_rows == tr.n_rows
    same_cols = src.n_cols == tr.n_cols
    same_row_lengths = same_rows and [len(r) for r in src.rows] == [len(r) for r in tr.rows]
    missing = sum(1 for r in range(src.n_rows) for c in range(len(src.rows[r])) if tr.get(r, c) is None)
    return {
        "structure_preserved": int(same_rows and same_cols and same_row_lengths),
        "source_shape": [src.n_rows, src.n_cols],
        "translated_shape": [tr.n_rows, tr.n_cols],
        "same_row_count": int(same_rows),
        "same_column_count": int(same_cols),
        "same_row_lengths": int(same_row_lengths),
        "source_cells_without_counterpart": missing,
    }


# ----------------------------------------------------------------------------- cells (T2-T5)
def _make_cell(row, col, source, translation, header) -> Optional[dict]:
    kind = classify_cell(source)
    cell = dict(row=row, column=col, source=source, translation=translation, kind=kind, column_header=header,
                correct=None, information_preserved=None, no_added_information=None,
                judged_by=None, note="", immutable=None, needs_llm=False)
    empty_tr = not translation
    if kind in IMMUTABLE_KINDS:                       # deterministic (T5)
        imm = check_immutable(source, translation, kind)
        cell.update(immutable=imm, correct=imm["ok"], judged_by="rule",
                    information_preserved=0 if empty_tr else (1 if imm["ok"] else None),
                    no_added_information=1 if (imm["ok"] or empty_tr) else None)
        return cell
    if kind == "empty":
        if empty_tr:
            return None                               # nothing to evaluate
        cell.update(correct=0, information_preserved=1, no_added_information=0, judged_by="rule",
                    note="text added for an empty source cell")
        return cell
    if empty_tr:                                      # text cell whose translation is missing/empty
        cell.update(correct=0, information_preserved=0, no_added_information=1, judged_by="rule",
                    note="translation missing or empty")
        return cell
    cell["needs_llm"] = True                          # semantic judgement needed
    return cell


def prepare_cells(src: Table, tr: Table) -> list[dict]:
    """Pair source and translated cells BY COORDINATES (header = row None, data rows 0..n-1)."""
    cells = []
    for col, text in enumerate(src.header):
        cells.append(_make_cell(None, col, text, tr.get(None, col), text))
    for r, row in enumerate(src.rows):
        for col, text in enumerate(row):
            header = src.header[col] if col < len(src.header) else ""
            cells.append(_make_cell(r, col, text, tr.get(r, col), header))
    return [c for c in cells if c is not None]


def select_cells(cells: list[dict], max_cells: int, seed_str: str, priority: set) -> list[dict]:
    """Cap the number of judged cells deterministically; headers and mention-linked cells always stay."""
    if max_cells <= 0 or len(cells) <= max_cells:
        return cells
    must = [i for i, c in enumerate(cells) if c["row"] is None or (c["row"], c["column"]) in priority]
    rest = [i for i in range(len(cells)) if i not in set(must)]
    room = max(0, max_cells - len(must))
    chosen = set(must) | set(random.Random(seed_str).sample(rest, min(room, len(rest))))
    return [c for i, c in enumerate(cells) if i in chosen]


def _mean(vals: Iterable) -> Optional[float]:
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def table_stats(cells: list[dict]) -> dict:
    text = [c for c in cells if is_text_cell(c)]
    imm = [c["immutable"] for c in cells if c.get("immutable")]
    return {
        "cell_translation_accuracy": _mean(c["correct"] for c in text),
        "information_preservation": _mean(c["information_preserved"] for c in text),
        "no_added_information": _mean(c["no_added_information"] for c in text),
        "immutable_accuracy": _mean(i["ok"] for i in imm),
        "immutable_accuracy_lenient": _mean(i["ok_lenient"] for i in imm),
        "n_text_cells": len(text),
        "n_immutable_cells": len(imm),
    }
