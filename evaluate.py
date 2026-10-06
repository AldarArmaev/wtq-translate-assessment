#!/usr/bin/env python3
"""Baseline evaluation of the en->ru translation of WikiTableQuestions (questions, tables, table mentions).

Example:
    python evaluate.py --root .. --sample-size 200 --seed 42

Principles: everything that can be checked deterministically is checked by Python; the LLM only makes atomic
binary semantic verdicts; all percentages are computed by Python. `targetValue` is never used.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import llm_judge as J
import mentions as M
import metrics
from config import Config, RUBRIC_VERSION, check_rubric_lock, load_dotenv
from loaders import (find_trace_dir, load_dataset, load_table, load_trace, natural_key,
                     resolve_source_table, sample_ids)
from tables import compare_structure, prepare_cells, select_cells, table_stats

HERE = Path(__file__).resolve().parent


# ============================================================================= per-example evaluation
def _table_section(cfg: Config, client, ex_id: str, src_tab, tr_tab, domain, priority: set) -> dict:
    structure = compare_structure(src_tab, tr_tab)
    all_cells = prepare_cells(src_tab, tr_tab)
    cells = select_cells(all_cells, cfg.max_cells, f"{cfg.seed}:{ex_id}", priority)
    todo = [c for c in cells if c["needs_llm"]]
    if client and todo:
        uniq: dict = {}
        for c in todo:   # identical (column, source, translation) triples are judged once
            uniq.setdefault((c["column_header"], c["source"], c["translation"]), len(uniq))
        items = [{"i": i, "column": k[0], "source": k[1], "translation": k[2]} for k, i in uniq.items()]
        res = J.judge_cells(client, items, domain, cfg.cell_batch_size)
        for c in todo:
            r = res.get(uniq[(c["column_header"], c["source"], c["translation"])])
            if r:
                c.update(correct=r["correct"], information_preserved=r["information_preserved"],
                         no_added_information=r["no_added_information"], note=r["note"], judged_by="llm")
            else:
                c["judged_by"] = "llm_failed"
    for c in cells:
        c.pop("needs_llm", None)
    return {"structure_preserved": structure["structure_preserved"], "structure": structure,
            "cells_total": len(all_cells), "cells_evaluated": len(cells), "cells": cells, **table_stats(cells)}


def _mention_section(client, trace, src_q, tr_q, src_tab, tr_tab, cells_by_coord) -> dict:
    raw = trace.get("mentions") or []
    pres = M.preservation_info(trace.get("quality") or {}, raw)
    items = M.build_items(raw, pres, src_tab, tr_tab, cells_by_coord, tr_q)
    results: dict = {}
    todo = [it["_payload"] for it in items if it["_needs_llm"]]
    if client and todo:
        results = J.judge_mentions(client, src_q, tr_q, todo)
    items = [M.resolve_item(it, results.get(it["_id"]), tr_q, cells_by_coord) for it in items]
    return {"preservation": pres["preservation"], "original_masks": pres["original_masks"],
            "forward_masks": pres["forward_masks"], "missing_forward_masks": pres["missing_forward_masks"],
            "added_forward_masks": pres["added_forward_masks"], "items": items}


def evaluate_example(ex_id: str, ds: dict, cfg: Config, client) -> dict:
    ex = ds[ex_id]
    trace = load_trace(cfg.trace_dir / f"{ex_id}.json")
    quality = trace.get("quality") or {}
    domain = trace.get("domain") or quality.get("domain")
    src_tab = load_table(resolve_source_table(cfg.root, trace["source_context"]), cfg.tsv_quoting)
    tr_path = cfg.dataset_dir / ex.context
    if not tr_path.is_file():
        raise FileNotFoundError(f"translated table not found: {tr_path}")
    tr_tab = load_table(tr_path, cfg.tsv_quoting)
    src_q, tr_q = trace.get("source_question") or "", ex.utterance

    rec = {"id": ex_id, "rubric_version": RUBRIC_VERSION, "rubric_hash": J.RUBRIC_HASH,
           "source_context": trace["source_context"], "translated_context": ex.context,
           "question": None, "table": None, "mentions": None,
           "diagnostics": M.collect_diagnostics(trace), "warnings": [], "section_errors": {}}
    if (trace.get("translated_question") or "").strip() not in ("", tr_q.strip()):
        rec["warnings"].append("trace.translated_question differs from dataset utterance (dataset utterance evaluated)")

    raw_mentions = trace.get("mentions") or []

    # ---- table (T1-T5)
    cells_by_coord: dict = {}
    try:
        rec["table"] = _table_section(cfg, client, ex_id, src_tab, tr_tab, domain, M.linked_coordinates(raw_mentions))
        cells_by_coord = {(c["row"], c["column"]): c for c in rec["table"]["cells"]}
    except Exception as e:  # noqa: BLE001
        rec["section_errors"]["table"] = repr(e)

    # ---- table mentions (M1-M4)
    try:
        rec["mentions"] = _mention_section(client, trace, src_q, tr_q, src_tab, tr_tab, cells_by_coord)
    except Exception as e:  # noqa: BLE001
        rec["section_errors"]["mentions"] = repr(e)

    # ---- question (Q1-Q4 + critical error)
    q = {"source": src_q, "translation": tr_q, "meaning_preserved": None, "key_elements_correct": None,
         "logic_preserved": None, "russian_naturalness": None, "critical_error": None,
         "critical_error_llm": None, "critical_error_from_mentions": None, "critical_error_from_table": None,
         "issues": []}
    try:
        if client:
            r = J.judge_question(client, src_q, tr_q, domain)
            q.update({k: r[k] for k in ("meaning_preserved", "key_elements_correct", "logic_preserved",
                                        "russian_naturalness", "issues")})
            q["critical_error_llm"] = r["critical_error"]
            if r["logic_preserved"] == 0 and r["critical_error"] == 0:
                rec["warnings"].append("logic_preserved=0 but critical_error=0: review manually")
    except Exception as e:  # noqa: BLE001
        rec["section_errors"]["question"] = repr(e)

    # critical error = LLM verdict OR deterministic/mention-derived triggers (never the similarity metrics)
    items = (rec["mentions"] or {}).get("items") or []
    leak = "<TABLE_REF" in tr_q
    if rec["mentions"] is not None:
        q["critical_error_from_mentions"] = int(leak or rec["mentions"]["preservation"] == 0
                                                or any(i["correctness"] == 0 for i in items))
    linked = M.linked_coordinates(raw_mentions)
    if rec["table"] is not None:
        q["critical_error_from_table"] = int(any(
            c.get("immutable") and c["immutable"]["ok"] == 0 and c["immutable"]["ok_lenient"] == 0
            for c in rec["table"]["cells"] if (c["row"], c["column"]) in linked))
    if leak:
        rec["warnings"].append("unresolved <TABLE_REF_*> placeholder in the final question")
    triggers = [q["critical_error_llm"], q["critical_error_from_mentions"], q["critical_error_from_table"]]
    if any(t == 1 for t in triggers):
        q["critical_error"] = 1
    elif all(t == 0 for t in triggers):
        q["critical_error"] = 0
    rec["question"] = q
    return rec


def _safe_eval(ex_id, ds, cfg, client):
    try:
        return "ok", evaluate_example(ex_id, ds, cfg, client)
    except Exception as e:  # noqa: BLE001
        return "fail", {"id": ex_id, "stage": "load/evaluate", "error": repr(e),
                        "traceback": traceback.format_exc(limit=3)}


# ============================================================================= run
def _read_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(cfg: Config, client=None) -> dict:
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    eval_path, err_path = out / "evaluation.jsonl", out / "errors.jsonl"

    ds = load_dataset(cfg.dataset_dir / "training.tsv", cfg.tsv_quoting)
    if cfg.ids_file:
        ids = [l.strip() for l in cfg.ids_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        unknown = [i for i in ids if i not in ds]
        if unknown:
            raise SystemExit(f"ids not in training.tsv: {unknown[:5]}")
    else:
        ids = sample_ids(ds.keys(), cfg.sample_size, cfg.seed)
    (out / "sample_ids.json").write_text(json.dumps(
        {"seed": cfg.seed, "sample_size": cfg.sample_size, "n_dataset": len(ds), "ids": ids}, indent=1), encoding="utf-8")

    done: dict = {}
    if eval_path.is_file() and eval_path.stat().st_size:
        if not (cfg.resume or cfg.force):
            raise SystemExit(f"{eval_path} already exists: use --resume to continue or --force to overwrite.")
        if cfg.resume:
            for r in _read_jsonl(eval_path):
                if r.get("rubric_hash") != J.RUBRIC_HASH:
                    raise SystemExit("Existing results were produced with a different rubric; use another --out-dir.")
                if r["id"] in ids and not r.get("section_errors"):
                    done[r["id"]] = r
    pending = [i for i in ids if i not in done]
    print(f"sample: {len(ids)} examples ({len(done)} already done, {len(pending)} to evaluate)", file=sys.stderr)

    failures: list = []
    with open(eval_path, "w", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
        for r in done.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        futs = {pool.submit(_safe_eval, i, ds, cfg, client): i for i in pending}
        for n, fut in enumerate(as_completed(futs), 1):
            status, payload = fut.result()
            if status == "ok":
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                fh.flush()
            else:
                failures.append(payload)
            print(f"[{n}/{len(pending)}] {futs[fut]} {status}", file=sys.stderr)

    records = sorted(_read_jsonl(eval_path), key=lambda r: natural_key(r["id"]))
    _write_jsonl(eval_path, records)
    errors = failures + [{"id": r["id"], "stage": f"section:{k}", "error": v}
                         for r in records for k, v in (r.get("section_errors") or {}).items()]
    _write_jsonl(err_path, sorted(errors, key=lambda e: natural_key(e["id"])))

    meta = {"rubric_version": RUBRIC_VERSION, "rubric_hash": J.RUBRIC_HASH, "use_llm": bool(client),
            "provider": cfg.provider if client else None, "model": client.model if client else None,
            "temperature": cfg.temperature, "seed": cfg.seed, "sample_size_requested": cfg.sample_size,
            "n_sampled": len(ids), "n_evaluated": len(records), "n_failed_examples": len(failures),
            "n_with_section_errors": sum(1 for r in records if r.get("section_errors")),
            "max_cells_per_table": cfg.max_cells,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    summary = metrics.summarize(records, meta)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = metrics.format_report(summary)
    (out / "report.txt").write_text(report + "\n", encoding="utf-8")
    print("\n" + report)
    if errors:
        print(f"\n{len(errors)} error(s) written to {err_path}", file=sys.stderr)
    return summary


# ============================================================================= CLI
def parse_args(argv: Optional[list] = None) -> Config:
    load_dotenv(HERE / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=HERE.parent, help="wtq/ project root (default: parent of evaluation/)")
    ap.add_argument("--dataset-dir", type=Path, help="default: <root>/dataset")
    ap.add_argument("--trace-dir", type=Path, help="default: auto-detect <root>/*/traces/training")
    ap.add_argument("--out-dir", type=Path, default=HERE / "output")
    ap.add_argument("--sample-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ids-file", type=Path, help="evaluate exactly these ids (one per line) instead of sampling")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-cells", type=int, default=150, help="max cells judged per table (0 = no cap)")
    ap.add_argument("--cell-batch-size", type=int, default=40)
    ap.add_argument("--provider", choices=["anthropic", "openai"], default=os.environ.get("LLM_PROVIDER", "anthropic"))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", ""))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tsv-quoting", choices=["none", "minimal"], default="none")
    ap.add_argument("--no-llm", action="store_true", help="deterministic checks only (no API calls)")
    ap.add_argument("--resume", action="store_true", help="keep finished examples of an earlier run")
    ap.add_argument("--force", action="store_true", help="overwrite existing results")
    ap.add_argument("--aggregate-only", action="store_true", help="re-aggregate output/evaluation.jsonl and exit")
    a = ap.parse_args(argv)

    if a.aggregate_only:
        recs = _read_jsonl(a.out_dir / "evaluation.jsonl")
        summ = metrics.summarize(recs, {"n_evaluated": len(recs), "n_sampled": len(recs), "use_llm": True})
        print(metrics.format_report(summ))
        raise SystemExit(0)

    root = a.root.resolve()
    return Config(root=root, dataset_dir=(a.dataset_dir or root / "dataset").resolve(),
                  trace_dir=(a.trace_dir or find_trace_dir(root)).resolve(), out_dir=a.out_dir.resolve(),
                  sample_size=a.sample_size, seed=a.seed, workers=a.workers, max_cells=a.max_cells,
                  cell_batch_size=a.cell_batch_size, use_llm=not a.no_llm, provider=a.provider, model=a.model,
                  temperature=a.temperature, tsv_quoting=a.tsv_quoting, ids_file=a.ids_file,
                  resume=a.resume, force=a.force)


def main(argv: Optional[list] = None) -> None:
    cfg = parse_args(argv)
    check_rubric_lock(HERE / "rubric_lock.json", RUBRIC_VERSION, J.RUBRIC_HASH)
    client = J.LLMClient(cfg, cfg.out_dir / "cache") if cfg.use_llm else None
    if client is None:
        print("WARNING: --no-llm, semantic metrics will be empty", file=sys.stderr)
    run(cfg, client)


if __name__ == "__main__":
    main()
