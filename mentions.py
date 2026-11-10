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
def build_items(mentions, pres, src, tr, cells_by_coord, translated_question):
    items = []
    for idx, m in enumerate(mentions):
        translated = m.get('translated_text')
        item = {'mask': m.get('mask'), 'source_text': m.get('question_text'),
                'translated_text': translated, 'translated_span': translated,
                'preserved': None if pres.get('forward_masks') is None else int(m.get('mask') in pres['forward_masks']),
                'correctness': None, 'judged_by': None, 'note': '', 'flags': [],
                '_id': f'm{idx}', '_needs_llm': False, '_payload': None}
        if not item['source_text']:
            item['flags'].append('source_text_missing')
        elif translated is None:
            item['flags'].append('translated_text_missing')
        elif not translated.strip():
            item.update(correctness=0, judged_by='rule', note='empty translated_text')
        else:
            item['_needs_llm'] = True
            item['_payload'] = {'id': item['_id'], 'source_mention': item['source_text'], 'translated_text': translated}
        items.append(item)
    return items


def resolve_item(item, res, translated_question, cells_by_coord):
    if item['_needs_llm']:
        if res is None:
            item['flags'].append('llm_judgement_unavailable')
        else:
            item.update(correctness=res['correctness'], note=res['note'], judged_by='llm')
    for k in ('_id', '_needs_llm', '_payload'):
        item.pop(k, None)
    return item
