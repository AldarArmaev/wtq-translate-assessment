"""Run entity-only question experiments: English->Russian and English->back-English.

No tables or mention judgements are loaded or sent to the judge.
"""
import argparse
import datetime as dt
import json
import time
from pathlib import Path

import entities as E
import llm_judge as J
import metrics
from config import Config, RUBRIC_VERSION, check_rubric_lock, load_dotenv
from loaders import load_dataset, load_trace, sample_ids


class LoggedClient(J.LLMClient):
    """Record every API attempt, including invalid JSON attempts, without credentials."""
    def _call(self, system, user):
        started = time.perf_counter()
        status = 'response_received'
        try:
            return super()._call(system, user)
        except Exception as exc:
            status = type(exc).__name__
            raise
        finally:
            event = {'timestamp': dt.datetime.now(dt.timezone.utc).isoformat(),
                     'example_id': self.example_id, 'status': status,
                     'elapsed_seconds': time.perf_counter() - started}
            with (self.cfg.out_dir / 'api_attempts.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(event) + '\n')


def candidate(trace, russian_question, mode):
    if mode == 'forward':
        text, lang = russian_question, 'ru'
    elif mode == 'back':
        text = (trace.get('quality') or {}).get('back_translated_question')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('missing quality.back_translated_question; no fallback or new translation')
        lang = 'en'
    else:
        raise ValueError('unknown comparison mode')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('empty question candidate')
    return text, lang


def run_mode(extractor, args, ds, ids, mode):
    out = args.out_dir / mode
    if out.exists():
        raise FileExistsError(f'Use a new output directory: {out}')
    out.mkdir(parents=True)
    cfg = Config(root=Path.cwd(), dataset_dir=args.dataset_dir, trace_dir=args.trace_dir,
                 out_dir=out, model=args.model, provider='openai', workers=1, max_cells=0)
    client = LoggedClient(cfg, out / 'cache')
    started = time.perf_counter()
    records, errors = [], []
    for ex_id in ids:
        item_started = time.perf_counter()
        client.example_id = ex_id
        try:
            trace_path = args.trace_dir / f'{ex_id}.json'
            trace = load_trace(trace_path)
            src = trace.get('source_question')
            if not isinstance(src, str) or not src.strip():
                raise ValueError('source_question missing')
            target, lang = candidate(trace, ds[ex_id].utterance, mode)
            result = E.evaluate(extractor, client, src, target, [], target_lang=lang)
            if any(it['judgement_available'] is False for it in result['items']):
                raise RuntimeError('entity judgement unavailable')
            records.append({'id': ex_id, 'comparison': mode, 'trace_path': str(trace_path),
                            'entities': result, 'elapsed_seconds': time.perf_counter() - item_started})
            print(f'{mode}: {ex_id} ok ({records[-1]["elapsed_seconds"]:.2f}s)', flush=True)
        except Exception as exc:
            errors.append({'id': ex_id, 'error': repr(exc)})
            print(f'{mode}: {ex_id} failed: {exc}', flush=True)
    attempts = [json.loads(line) for line in (out / 'api_attempts.jsonl').read_text().splitlines()] \
        if (out / 'api_attempts.jsonl').exists() else []
    meta = {'comparison': mode, 'scope': 'question_entities_only', 'model': args.model,
            'rubric_version': RUBRIC_VERSION, 'rubric_hash': J.RUBRIC_HASH, 'seed': args.seed,
            'ids': ids, 'n_sampled': len(ids), 'n_evaluated': len(records), 'n_failed': len(errors),
            'temperature': 0, 'workers': 1, 'entity_setup_seconds_shared': extractor.setup_seconds,
            'elapsed_seconds': time.perf_counter() - started,
            'mean_example_seconds': sum(r['elapsed_seconds'] for r in records) / len(records) if records else None,
            'api_attempts': len(attempts), 'cache_initially_empty': True,
            'dataset_dir': str(args.dataset_dir), 'trace_dir': str(args.trace_dir),
            'back_translation_source': 'existing trace quality.back_translated_question' if mode == 'back' else None}
    summary = {'meta': meta, 'entities': metrics.summarize(records, meta)['entities']}
    for name, rows in [('evaluation', records), ('errors', errors)]:
        (out / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main():
    load_dotenv(Path(__file__).resolve().parent / '.env')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-dir', type=Path, required=True)
    ap.add_argument('--trace-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--entity-model-dir', type=Path, default=Path('.models'))
    ap.add_argument('--model', default='deepseek/deepseek-v4-flash-0731')
    ap.add_argument('--sample-size', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ids-file', type=Path, help='Explicit question IDs, one per line')
    ap.add_argument('--ner-backend', choices=['gliner', 'stanza-ontonotes'], default='gliner')
    ap.add_argument('--mode', choices=['both', 'forward', 'back'], default='both')
    args = ap.parse_args()
    if args.ner_backend == 'stanza-ontonotes' and args.mode != 'back':
        ap.error('Stanza English OntoNotes requires --mode back')
    if args.out_dir.exists():
        raise FileExistsError('Choose a fresh out-dir to keep both experiments comparable')
    check_rubric_lock(Path(__file__).resolve().parent / 'rubric_lock.json', RUBRIC_VERSION, J.RUBRIC_HASH)
    ds = load_dataset(args.dataset_dir / 'training.tsv')
    ids = (args.ids_file.read_text(encoding='utf-8-sig').split()
           if args.ids_file else sample_ids(ds.keys(), args.sample_size, args.seed))
    if not ids or len(ids) != len(set(ids)) or any(i not in ds for i in ids):
        raise ValueError('IDs must be nonempty, unique, and present in the dataset')
    # Fail before downloading models or spending API credits if inputs are missing.
    for ex_id in ids:
        trace = load_trace(args.trace_dir / f'{ex_id}.json')
        for mode in (('forward', 'back') if args.mode == 'both' else (args.mode,)):
            candidate(trace, ds[ex_id].utterance, mode)
    if args.ner_backend == 'stanza-ontonotes':
        from stanza_entities import StanzaEntityExtractor
        extractor = StanzaEntityExtractor(args.entity_model_dir)
    else:
        extractor = E.EntityExtractor(args.entity_model_dir)
    modes = ('forward', 'back') if args.mode == 'both' else (args.mode,)
    summaries = {mode: run_mode(extractor, args, ds, ids, mode) for mode in modes}
    (args.out_dir / 'comparison.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    if any(s['meta']['n_failed'] for s in summaries.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
