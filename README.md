# WTQ en→ru translation quality baseline

```bash
cd wtq/evaluation
pip install -r requirements.txt
cp .env.example .env            # add the API key
python tests/test_smoke.py      # offline sanity check (fake judge)

python evaluate.py --sample-size 200 --seed 42        # baseline
python evaluate.py --sample-size 100 --seed 42 --workers 8
python evaluate.py --resume                            # continue an interrupted run
python evaluate.py --no-llm --force                    # deterministic checks only
python evaluate.py --aggregate-only                    # re-aggregate output/evaluation.jsonl
python evaluate.py --ids-file ids.txt --out-dir output_review --force   # re-run chosen ids
```

Outputs (`output/`): `evaluation.jsonl` (one record per example), `summary.json`, `report.txt`,
`errors.jsonl`, `sample_ids.json`, `cache/` (LLM answers keyed by model+rubric+prompt).

## What is deterministic and what is LLM
| Block | Deterministic (Python) | LLM (binary, temp 0) |
|---|---|---|
| Question | placeholder-leak check | Q1–Q4, critical_error |
| Table | T1 structure, T5 immutable values (numbers, years, dates, %, URLs, IDs, symbols), missing/empty cells | T2–T4 for text cells |
| Mentions | M1 preservation (trace masks), M4 surface, coordinate/translation-variant checks | M2 correctness, M3 semantic consistency, span extraction |

Final `critical_error` = LLM flag OR any mention with correctness=0 / lost mask / leaked `<TABLE_REF_*>`
OR a changed immutable value in a cell a mention points to. Breakdown is in `summary.json`.
Similarity/BERTScore/back-translation metrics from the traces are copied to `diagnostics` only.

## Rubric versioning
Prompts live in `llm_judge.py`; `RUBRIC_HASH` is their hash. `rubric_lock.json` maps `RUBRIC_VERSION → hash`;
editing a prompt without bumping `RUBRIC_VERSION` in `config.py` aborts the run.
