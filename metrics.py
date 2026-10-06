"""Aggregate metrics computed by Python only (never by the LLM).

Usage (re-aggregate an existing run):  python metrics.py output/evaluation.jsonl
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

from tables import is_text_cell


# ----------------------------------------------------------------------------- primitives
def wilson(k: int, n: int, z: float = 1.96) -> Optional[list]:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def rate(values: Iterable) -> dict:
    vals = [v for v in values if v is not None]
    n, k = len(vals), int(sum(vals))
    return {"rate": (k / n) if n else None, "k": k, "n": n, "ci95": wilson(k, n)}


def pooled_and_macro(records: list, getter: Callable[[dict], list]) -> dict:
    """pooled = over all items; macro = mean of per-example rates (each example weighs the same)."""
    flat, per_ex = [], []
    for r in records:
        vals = [v for v in getter(r) if v is not None]
        if vals:
            flat.extend(vals)
            per_ex.append(sum(vals) / len(vals))
    out = rate(flat)
    out["macro"] = {"rate": (sum(per_ex) / len(per_ex)) if per_ex else None, "n_examples": len(per_ex)}
    return out


def _sec(r: dict, name: str) -> dict:
    return r.get(name) or {}


def _cells(r: dict) -> list:
    return _sec(r, "table").get("cells") or []


def _items(r: dict) -> list:
    return _sec(r, "mentions").get("items") or []


# ----------------------------------------------------------------------------- summary
def summarize(records: list, meta: dict) -> dict:
    q = lambda key: rate(_sec(r, "question").get(key) for r in records)          # noqa: E731
    question = {
        "meaning_preservation": q("meaning_preserved"),
        "key_elements_correctness": q("key_elements_correct"),
        "logic_preservation": q("logic_preserved"),
        "russian_naturalness": q("russian_naturalness"),
        "critical_error_rate": q("critical_error"),
        "critical_error_breakdown": {
            "llm": q("critical_error_llm"),
            "from_mentions": q("critical_error_from_mentions"),
            "from_table_values": q("critical_error_from_table"),
        },
    }
    table = {
        "structure_preservation": rate(_sec(r, "table").get("structure_preserved") for r in records),
        "cell_correctness": pooled_and_macro(records, lambda r: [c["correct"] for c in _cells(r) if is_text_cell(c)]),
        "information_preservation": pooled_and_macro(
            records, lambda r: [c["information_preserved"] for c in _cells(r) if is_text_cell(c)]),
        "no_added_information": pooled_and_macro(
            records, lambda r: [c["no_added_information"] for c in _cells(r) if is_text_cell(c)]),
        "immutable_accuracy": pooled_and_macro(
            records, lambda r: [c["immutable"]["ok"] for c in _cells(r) if c.get("immutable")]),
        "immutable_accuracy_lenient": pooled_and_macro(
            records, lambda r: [c["immutable"]["ok_lenient"] for c in _cells(r) if c.get("immutable")]),
        "cells_judged_by_llm": sum(1 for r in records for c in _cells(r) if c.get("judged_by") == "llm"),
        "cells_llm_failed": sum(1 for r in records for c in _cells(r) if c.get("judged_by") == "llm_failed"),
    }
    mentions = {
        "preservation": pooled_and_macro(records, lambda r: [i["preserved"] for i in _items(r)]),
        "preservation_all_masks": rate(_sec(r, "mentions").get("preservation") for r in records),
        "correctness": pooled_and_macro(records, lambda r: [i["correctness"] for i in _items(r)]),
        "semantic_consistency": pooled_and_macro(records, lambda r: [i["semantic_consistency"] for i in _items(r)]),
        "surface_consistency": pooled_and_macro(records, lambda r: [i["surface_consistency"] for i in _items(r)]),
        "cell_text_in_question": pooled_and_macro(records, lambda r: [i["cell_text_in_question"] for i in _items(r)]),
        "n_mentions": sum(len(_items(r)) for r in records),
    }
    return {"meta": meta, "question": question, "table": table, "mentions": mentions,
            "diagnostics": _diagnostics(records)}


def _diagnostics(records: list) -> dict:
    """Pipeline metrics (cosine / BERTScore / back-translation) kept as diagnostics only."""
    keys = sorted({k for r in records for k, v in (r.get("diagnostics") or {}).items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)})
    out = {}
    for k in keys:
        vals = [r["diagnostics"][k] for r in records if isinstance((r.get("diagnostics") or {}).get(k), (int, float))]
        by_crit = {}
        for flag in (0, 1):
            sub = [r["diagnostics"][k] for r in records
                   if _sec(r, "question").get("critical_error") == flag
                   and isinstance((r.get("diagnostics") or {}).get(k), (int, float))]
            by_crit[f"mean_when_critical_error={flag}"] = statistics.fmean(sub) if sub else None
            by_crit[f"n_critical_error={flag}"] = len(sub)
        out[k] = {"n": len(vals), "mean": statistics.fmean(vals), "median": statistics.median(vals),
                  "min": min(vals), "max": max(vals), **by_crit}
    return out


# ----------------------------------------------------------------------------- text report
def _pct(x: Optional[float]) -> str:
    return "  n/a " if x is None else f"{100 * x:5.1f}%"


def format_report(s: dict) -> str:
    L: list[str] = []
    m = s["meta"]
    L.append(f"WTQ en->ru translation quality | rubric {m.get('rubric_version')} ({m.get('rubric_hash')}) | "
             f"{m.get('provider')}:{m.get('model')} | seed={m.get('seed')} | evaluated={m.get('n_evaluated')}"
             f"/{m.get('n_sampled')}")
    if not m.get("use_llm"):
        L.append("!! LLM disabled: only deterministic metrics are filled in")

    def single(label, d, indent="    "):
        ci = d["ci95"]
        ci_s = f"  95% CI [{100 * ci[0]:.1f}; {100 * ci[1]:.1f}]" if ci else ""
        L.append(f"{indent}{label:<34}{_pct(d['rate'])}{ci_s}  (n={d['n']})")

    def double(label, d):
        ci = d["ci95"]
        ci_s = f"  95% CI [{100 * ci[0]:.1f}; {100 * ci[1]:.1f}]" if ci else ""
        L.append(f"    {label:<34}{_pct(d['rate'])}{ci_s}  (n={d['n']})   per-example mean: {_pct(d['macro']['rate'])}")

    q, t, mm = s["question"], s["table"], s["mentions"]
    L += ["", "Question translation"]
    single("Meaning preservation:", q["meaning_preservation"])
    single("Key elements correctness:", q["key_elements_correctness"])
    single("Logic preservation:", q["logic_preservation"])
    single("Russian naturalness:", q["russian_naturalness"])
    single("Critical error rate:", q["critical_error_rate"])
    for name, d in q["critical_error_breakdown"].items():
        single(f"  - {name}:", d, "      ")
    L += ["", "Table translation"]
    single("Structure preservation:", t["structure_preservation"])
    double("Cell correctness:", t["cell_correctness"])
    double("Information preservation:", t["information_preservation"])
    double("No added information:", t["no_added_information"])
    double("Immutable accuracy:", t["immutable_accuracy"])
    double("  (lenient formatting):", t["immutable_accuracy_lenient"])
    L += ["", "Table mentions"]
    double("Preservation:", mm["preservation"])
    single("  all masks kept (example):", mm["preservation_all_masks"], "    ")
    double("Correctness:", mm["correctness"])
    double("Semantic consistency:", mm["semantic_consistency"])
    double("Surface consistency (diag.):", mm["surface_consistency"])
    double("Cell text in question (diag.):", mm["cell_text_in_question"])
    L += ["", "Pipeline diagnostics (NOT quality scores)"]
    for k, d in s["diagnostics"].items():
        L.append(f"    {k:<32} mean={d['mean']:.3f} median={d['median']:.3f} n={d['n']}")
    return "\n".join(L)


# ----------------------------------------------------------------------------- CLI
def read_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    recs = read_jsonl(Path(sys.argv[1]))
    summ = summarize(recs, {"n_evaluated": len(recs), "n_sampled": len(recs)})
    print(format_report(summ))
