"""Table-mention evaluation: M1 (deterministic), M2/M3 inputs for the LLM judge, M4 (deterministic)."""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

from loaders import Table
from tables import IMMUTABLE_KINDS

DIAG_PREFIXES = ("question_", "back_table_", "back_")


# ----------------------------------------------------------------------------- text helpers
def norm_surface(s: str) -> str:
    """casefold, ё->е, punctuation -> space, collapse whitespace (used for surface comparison)."""
    s = unicodedata.normalize("NFKC", s).casefold().replace("ё", "е")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def contains_bounded(haystack: str, needle: str) -> bool:
    h, n = norm_surface(haystack), norm_surface(needle)
    return bool(n) and re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", h) is not None


# ----------------------------------------------------------------------------- trace access
def mention_coordinate(m: dict) -> Optional[tuple]:
    c = m.get("coordinate")
    if not c:
        cs = m.get("coordinates") or []
        c = cs[0] if cs else None
    if not c or c.get("column") is None:
        return None
    return (c.get("row"), int(c["column"]))


def linked_coordinates(mentions: list[dict]) -> set:
    """All table cells/headers that mentions point to (selected coordinate + all candidate coordinates)."""
    out = set()
    for m in mentions:
        c = mention_coordinate(m)
        if c:
            out.add(c)
        for cc in m.get("coordinates") or []:
            if cc.get("column") is not None:
                out.add((cc.get("row"), int(cc["column"])))
    return out


def collect_diagnostics(trace: dict) -> dict:
    """Existing pipeline metrics, kept ONLY as diagnostics (never used as quality score)."""
    q = trace.get("quality") or {}
    diag = {}
    for src in (trace, q):
        for k, v in src.items():
            if k.startswith(DIAG_PREFIXES) and isinstance(v, (int, float)) and not isinstance(v, bool):
                diag[k] = v
    for k in ("back_translated_question", "status", "mode", "domain"):
        if q.get(k) is not None:
            diag[k] = q[k]
    diag["n_pipeline_warnings"] = len(q.get("warnings") or [])
    diag["n_pipeline_errors"] = len(q.get("errors") or [])
    diag["n_ambiguous_mentions"] = len(q.get("ambiguous_mentions") or [])
    return diag


# ----------------------------------------------------------------------------- M1 preservation
def preservation_info(quality: dict, mentions: list[dict]) -> dict:
    original = list(quality.get("original_masks") or [m.get("mask") for m in mentions])
    if not original or "forward_masks" not in quality:
        return {"preservation": None, "original_masks": original, "forward_masks": None,
                "missing_forward_masks": [], "added_forward_masks": []}
    forward = list(quality["forward_masks"] or [])
    missing = sorted(set(quality.get("missing_forward_masks") or []) | (set(original) - set(forward)))
    added = sorted(set(quality.get("added_forward_masks") or []) | (set(forward) - set(original)))
    return {"preservation": int(not missing and not added), "original_masks": original,
            "forward_masks": forward, "missing_forward_masks": missing, "added_forward_masks": added}


# ----------------------------------------------------------------------------- M2-M4 items
def build_items(mentions: list[dict], pres: dict, src: Table, tr: Table,
                cells_by_coord: dict, translated_question: str) -> list[dict]:
    items = []
    forward = pres.get("forward_masks")
    for idx, m in enumerate(mentions):
        coord = mention_coordinate(m)
        src_cell = src.get(*coord) if coord else None
        tr_cell = tr.get(*coord) if coord else None
        variants = []
        for cc in m.get("coordinates") or []:
            if cc.get("column") is None:
                continue
            s, t = src.get(cc.get("row"), int(cc["column"])), tr.get(cc.get("row"), int(cc["column"]))
            if s is not None and src_cell is not None and s.casefold() == src_cell.casefold() \
                    and t is not None and t not in variants:
                variants.append(t)
        flags = []
        if src_cell is not None and m.get("table_source_text") is not None \
                and src_cell.casefold() != str(m["table_source_text"]).strip().casefold():
            flags.append("trace_table_source_text_differs_from_source_file")
        item = {
            "mask": m.get("mask"),
            "source_text": m.get("question_text"),
            "source_type": m.get("source_type"),
            "coordinate": {"row": coord[0], "column": coord[1]} if coord else None,
            "table_source_text": src_cell if src_cell is not None else m.get("table_source_text"),
            "table_translation_text": tr_cell,
            "translation_variants": variants,
            "translation_variants_consistent": int(len(variants) <= 1),
            "preserved": None if forward is None else int(m.get("mask") in forward),
            "translated_span": None, "correctness": None, "semantic_consistency": None,
            "surface_consistency": None, "cell_text_in_question": None,
            "judged_by": None, "note": "", "flags": flags,
            "_id": f"m{idx}", "_needs_llm": False, "_payload": None,
        }
        if coord is None or src_cell is None:                       # broken trace coordinate: not a translation error
            flags.append("source_coordinate_missing")
        elif tr_cell is None:                                       # table lost this cell/header
            item.update(correctness=0, judged_by="rule")
            flags.append("translated_cell_missing")
        else:
            cell = cells_by_coord.get(coord)
            if cell and cell["kind"] in IMMUTABLE_KINDS and cell["correct"] == 1 and contains_bounded(translated_question, tr_cell):
                item.update(correctness=1, semantic_consistency=1, translated_span=tr_cell, judged_by="rule")
            else:
                item["_needs_llm"] = True
                item["_payload"] = {
                    "id": item["_id"], "source_mention": m.get("question_text"),
                    "source_element_type": m.get("source_type"), "source_table_text": src_cell,
                    "translated_table_text": tr_cell,
                    "other_translated_table_texts": [v for v in variants if v != tr_cell],
                }
        items.append(item)
    return items


def resolve_item(item: dict, res: Optional[dict], translated_question: str, cells_by_coord: dict) -> dict:
    """Merge the LLM verdict, apply deterministic overrides, compute M4 surface consistency."""
    if item["_needs_llm"]:
        if res is None:
            item["flags"].append("llm_judgement_unavailable")
        else:
            item.update(correctness=res["correctness"], semantic_consistency=res["semantic_consistency"],
                        translated_span=res["span"] or None, note=res["note"], judged_by="llm")
    coord = item["coordinate"]
    tr_cell = item["table_translation_text"]
    cell = cells_by_coord.get((coord["row"], coord["column"])) if coord else None
    imm = cell.get("immutable") if cell else None
    if imm and imm["ok"] == 0 and imm["ok_lenient"] == 0 and item["correctness"] != 0:
        item["correctness"] = 0                                     # the chain is broken by a changed table value
        item["flags"].append("immutable_cell_value_changed")
    span = item["translated_span"]
    if span and span not in translated_question and norm_surface(span) not in norm_surface(translated_question):
        item["flags"].append("span_not_found_in_question")
        span = None
        item["translated_span"] = None
    if span and tr_cell:
        item["surface_consistency"] = int(norm_surface(span) == norm_surface(tr_cell))
    if tr_cell:
        item["cell_text_in_question"] = int(contains_bounded(translated_question, tr_cell))
    for k in ("_id", "_needs_llm", "_payload"):
        item.pop(k, None)
    return item
