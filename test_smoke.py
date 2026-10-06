"""End-to-end smoke test on synthetic data with a FAKE judge (no API calls).

Run:  python tests/test_smoke.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluate  # noqa: E402
import llm_judge as J  # noqa: E402
from config import Config  # noqa: E402
from loaders import sample_ids  # noqa: E402
from tables import check_immutable, classify_cell  # noqa: E402


class FakeClient:
    """Mimics LLMClient.complete_json with trivial heuristics (everything is 'fine' unless obviously broken)."""
    model = "fake"

    def complete_json(self, system, user, parse):
        if system == J.SYSTEM_QUESTION:
            obj = {"issues": [], "meaning_preserved": 1, "key_elements_correct": 1, "logic_preserved": 1,
                   "russian_naturalness": 1, "critical_error": 0}
        elif system == J.SYSTEM_CELLS:
            cells = json.loads(user)["cells"]
            obj = {"results": [{"i": c["i"], "correct": 1, "information_preserved": 1,
                                "no_added_information": 0 if "добавлено" in c["translation"] else 1, "note": ""}
                               for c in cells]}
        else:
            data = json.loads(user)
            res = []
            for m in data["mentions"]:
                t = m["translated_table_text"]
                span = t if t.casefold() in data["translated_question"].casefold() else ""
                res.append({"id": m["id"], "span": span, "correctness": int(bool(span)),
                            "semantic_consistency": int(bool(span)), "note": ""})
            obj = {"results": res}
        return parse(obj)


def build_dataset(root: Path):
    (root / "csv/1-csv").mkdir(parents=True)
    (root / "dataset/tables/training").mkdir(parents=True)
    tdir = root / "trace-abc/traces/training"
    tdir.mkdir(parents=True)

    (root / "csv/1-csv/1.csv").write_text(
        "Year\tLeague\tWinner\n2004\tUSL A-League\tSir Barton\n2005\tUSL A-League\tGallant Fox\n", encoding="utf-8")
    (root / "csv/1-csv/2.csv").write_text(
        'Place,Attendance,Date\n1st,"9,734","March 5, 2004"\n2nd,"8,000","May 2, 2005"\n', encoding="utf-8")
    (root / "dataset/tables/training/nt-0.tsv").write_text(
        "Год\tЛига\tПобедитель\n2004\tUSL А-лига\tСэр Бартон\n2005\tUSL А-лига\tГаллант Фокс\n", encoding="utf-8")
    # nt-1: a number was corrupted, a date localised correctly, one extra added-info cell
    (root / "dataset/tables/training/nt-1.tsv").write_text(
        "Место\tПосещаемость\tДата\n1-е добавлено\t9,743\t5 марта 2004 г.\n2-е\t8,000\t2 мая 2005\n", encoding="utf-8")

    (root / "dataset/training.tsv").write_text(
        "id\tutterance\tcontext\ttargetValue\n"
        "nt-0\tв каком последнем году эта команда была частью USL А-лига?\ttables/training/nt-0.tsv\t2004\n"
        "nt-1\tсколько человек посетило матч 9,743?\ttables/training/nt-1.tsv\t1\n"
        "nt-2\tвопрос без trace\ttables/training/nt-0.tsv\t0\n", encoding="utf-8")

    def mention(mask, text, stype, coords, src_text):
        return {"mask": mask, "question_text": text, "source_type": stype, "coordinates": coords,
                "table_source_text": src_text, "coordinate": coords[0]}

    quality_ok = {"original_masks": ["<TABLE_REF_0000>", "<TABLE_REF_0001>"],
                  "forward_masks": ["<TABLE_REF_0000>", "<TABLE_REF_0001>"],
                  "missing_forward_masks": [], "added_forward_masks": [], "warnings": [], "errors": [],
                  "status": "accepted", "domain": "sports", "question_cosine_similarity": 0.98,
                  "question_bert_f1": 0.55, "back_table_mean_similarity": 0.99}
    (tdir / "nt-0.json").write_text(json.dumps({
        "question_id": "nt-0", "source_context": "csv/1-csv/1.csv", "output_context": "tables/training/nt-0.tsv",
        "source_question": "what was the last year where this team was a part of the usl a-league?",
        "translated_question": "в каком последнем году эта команда была частью USL А-лига?",
        "mentions": [mention("<TABLE_REF_0000>", "year", "header", [{"row": None, "column": 0}], "Year"),
                     mention("<TABLE_REF_0001>", "usl a-league", "cell",
                             [{"row": 0, "column": 1}, {"row": 1, "column": 1}], "USL A-League")],
        "quality": quality_ok, "domain": "sports"}, ensure_ascii=False), encoding="utf-8")
    quality_bad = dict(quality_ok, original_masks=["<TABLE_REF_0000>"], forward_masks=[])
    quality_bad["missing_forward_masks"] = ["<TABLE_REF_0000>"]
    (tdir / "nt-1.json").write_text(json.dumps({
        "question_id": "nt-1", "source_context": "csv/1-csv/2.csv", "output_context": "tables/training/nt-1.tsv",
        "source_question": "how many people attended the 9,734 game?",
        "translated_question": "сколько человек посетило матч 9,743?",
        "mentions": [mention("<TABLE_REF_0000>", "9,734", "cell", [{"row": 0, "column": 1}], "9,734")],
        "quality": quality_bad, "domain": "sports"}, ensure_ascii=False), encoding="utf-8")


def main():
    # unit checks of the deterministic parts
    assert classify_cell("9,734") == "numeric" and classify_cell("March 5, 2004") == "date"
    assert classify_cell("KV62") == "code" and classify_cell("Johnny Loftus") == "text"
    assert check_immutable("9,734", "9,734", "numeric")["ok"] == 1
    assert check_immutable("9,734", "9,743", "numeric")["ok_lenient"] == 0
    assert check_immutable("9,734", "9 734", "numeric") == {"kind": "numeric", "ok": 0, "ok_lenient": 1, "reason": "value_changed"}
    assert check_immutable("March 5, 2004", "5 марта 2004 г.", "date")["ok"] == 1
    assert sample_ids([f"nt-{i}" for i in range(100)], 10, 42) == sample_ids([f"nt-{i}" for i in range(100)], 10, 42)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_dataset(root)
        cfg = Config(root=root, dataset_dir=root / "dataset", trace_dir=root / "trace-abc/traces/training",
                     out_dir=root / "out", sample_size=10, seed=42, workers=2, force=True)
        summary = evaluate.run(cfg, FakeClient())

        recs = {json.loads(l)["id"]: json.loads(l) for l in (root / "out/evaluation.jsonl").read_text("utf-8").splitlines()}
        errors = [json.loads(l) for l in (root / "out/errors.jsonl").read_text("utf-8").splitlines()]
        assert set(recs) == {"nt-0", "nt-1"} and [e["id"] for e in errors] == ["nt-2"], (set(recs), errors)

        r0, r1 = recs["nt-0"], recs["nt-1"]
        assert r0["table"]["structure_preserved"] == 1 and r0["mentions"]["preservation"] == 1
        assert [i["correctness"] for i in r0["mentions"]["items"]] == [1, 1]
        assert r0["mentions"]["items"][1]["surface_consistency"] == 1     # "USL А-лига" == table cell
        assert r0["question"]["critical_error"] == 0
        assert r1["mentions"]["preservation"] == 0
        assert r1["mentions"]["items"][0]["correctness"] == 0             # corrupted number breaks the chain
        assert r1["question"]["critical_error"] == 1 and r1["question"]["critical_error_from_table"] == 1
        assert r1["table"]["immutable_accuracy"] < 1.0
        assert any(c["no_added_information"] == 0 for c in r1["table"]["cells"])
        assert r0["diagnostics"]["question_cosine_similarity"] == 0.98
        assert "targetValue" not in json.dumps(r0)
        assert summary["question"]["critical_error_rate"]["n"] == 2
        assert (root / "out/summary.json").is_file() and (root / "out/sample_ids.json").is_file()

        # resume: nothing left to evaluate, results identical
        cfg.force, cfg.resume = False, True
        evaluate.run(cfg, FakeClient())
    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()
