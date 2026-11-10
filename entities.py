"""Independent bilingual entity evaluation: Stanza tokenization + multilingual GLiNER."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import time
import threading

LABELS = ['person', 'organization', 'location', 'geopolitical entity', 'nationality',
          'facility', 'product', 'event', 'work of art', 'law', 'language',
          'date', 'time', 'money', 'percentage', 'quantity', 'ordinal', 'cardinal']
MODEL = 'urchade/gliner_multi-v2.1'


class EntityExtractor:
    def __init__(self, model_dir: Path):
        started = time.perf_counter()
        import stanza
        import torch
        from gliner import GLiNER
        torch.set_num_threads(4)
        self._lock = threading.Lock()
        self.nlp = {lang: stanza.Pipeline(lang, processors='tokenize',
                    dir=str(model_dir / 'stanza'), use_gpu=False, verbose=False)
                    for lang in ('en', 'ru')}
        local = model_dir / 'gliner-local'
        self.model = GLiNER.from_pretrained(str(local) if (local / 'model.safetensors').is_file()
                                            else MODEL, cache_dir=str(model_dir / 'huggingface'))
        self.setup_seconds = time.perf_counter() - started

    def extract(self, texts: list[str], lang: str) -> dict:
        with self._lock:
            return self._extract(texts, lang)

    def _extract(self, texts: list[str], lang: str) -> dict:
        # Sentence segmentation prevents long notes/cells being silently truncated by NER.
        unique = list(dict.fromkeys(t for t in texts if t))
        chunks, owners = [], []
        for text in unique:
            doc = self.nlp[lang](text)
            for sentence in doc.sentences:
                tokens = sentence.tokens
                # Bound each window well below GLiNER's token limit, retaining overlap.
                for start in range(0, len(tokens), 80):
                    window = tokens[start:start + 100]
                    if window:
                        offset = window[0].start_char
                        chunks.append(text[offset:window[-1].end_char])
                        owners.append((text, offset))
        result = {text: [] for text in unique}
        for start in range(0, len(chunks), 8):
            predictions = self.model.batch_predict_entities(chunks[start:start + 8], LABELS,
                                                            threshold=0.5)
            for (text, offset), found in zip(owners[start:start + 8], predictions):
                for ent in found:
                    entry = {'text': ent['text'], 'type': ent['label'],
                             'start': offset + ent['start'], 'end': offset + ent['end'],
                             'score': float(ent['score'])}
                    if not any(e['start'] == entry['start'] and e['end'] == entry['end']
                               and e['type'] == entry['type'] for e in result[text]):
                        result[text].append(entry)
        return result


def compare_types(source: list, translation: list) -> dict:
    """Count comparison is diagnostic; it does not establish entity identity."""
    a, b = Counter(e['type'] for e in source), Counter(e['type'] for e in translation)
    return {'source': dict(a), 'translation': dict(b), 'equal': int(a == b)}


def evaluate(extractor, client, src_q, tr_q, cells, target_lang='ru'):
    import llm_judge as J
    pairs = [('question', src_q, tr_q)] + [
        (f"cell:{c['row']}:{c['column']}", c['source'], c['translation'] or '') for c in cells]
    source = extractor.extract([s for _, s, _ in pairs], 'en')
    target = extractor.extract([t for _, _, t in pairs], target_lang)
    items, tasks, unique = [], [], {}
    for location, s, t in pairs:
        a, b = source.get(s, []), target.get(t, [])
        item = {'location': location, 'source_text': s, 'translation_text': t,
                'source_entities': a, 'translation_entities': b,
                'type_counts': compare_types(a, b), 'matches': [], 'added_target_ids': [],
                'judgement_available': None if a or b else True}
        if a or b:
            key = (s, t)
            if key not in unique:
                idx = len(tasks)
                unique[key] = idx
                tasks.append({'i': idx, 'source': s, 'translation': t,
                              'source_entities': [e['text'] for e in a],
                              'translation_entities': [e['text'] for e in b]})
            item['_task'] = unique[key]
        items.append(item)
    results = J.judge_entities(client, tasks) if client and tasks else {}
    for item in items:
        task = item.pop('_task', None)
        if task is None:
            continue
        verdict = results.get(task)
        item['judgement_available'] = verdict is not None
        if verdict is None:
            continue
        item['added_target_ids'] = verdict['added_target_ids']
        for match in verdict['matches']:
            dest = match['target_id']
            match = dict(match)
            match['type_preserved'] = int(dest is not None and
                item['source_entities'][match['source_id']]['type'] ==
                item['translation_entities'][dest]['type'])
            item['matches'].append(match)
    return {'model': getattr(extractor, 'model_name', MODEL),
            'labels': getattr(extractor, 'labels', LABELS),
            'threshold': getattr(extractor, 'threshold', 0.5),
            'tokenizer': getattr(extractor, 'tokenizer_name', 'stanza'), 'source_language': 'en',
            'target_language': target_lang, 'items': items}
