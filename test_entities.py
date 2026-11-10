"""Offline contract tests for entity alignment and pair-only mention evaluation."""
import unittest

import entities
import llm_judge as J
import mentions
from question_entity_experiments import candidate


class Extractor:
    def extract(self, texts, lang):
        return {text: [{'text': text, 'type': 'person' if lang == 'en' else 'organization'}]
                for text in texts if text}


class Judge:
    def complete_json(self, system, user, parse):
        import json
        tasks = json.loads(user)['cells']
        return parse({'results': [{'i': t['i'], 'matches': [
            {'source_id': 0, 'target_id': 0, 'meaning_preserved': 1, 'note': ''}],
            'added_target_ids': []} for t in tasks]})


class EntityTests(unittest.TestCase):
    def test_full_evaluation_keeps_russian_question_but_uses_back_for_entities(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import evaluate
        from config import Config
        from loaders import load_dataset
        from test_smoke import build_dataset, FakeClient
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dataset(root)
            trace_dir = root / 'trace-abc/traces/training'
            trace_path = trace_dir / 'nt-0.json'
            trace = json.loads(trace_path.read_text(encoding='utf-8'))
            trace['quality']['back_translated_question'] = 'English back question'
            trace_path.write_text(json.dumps(trace), encoding='utf-8')
            cfg = Config(root=root, dataset_dir=root/'dataset', trace_dir=trace_dir,
                         out_dir=root/'out', entity_backend='stanza-ontonotes',
                         entity_scope='question', entity_comparison='back')
            cfg._extractor = object()
            ds = load_dataset(root/'dataset/training.tsv')
            with patch('entities.evaluate', return_value={'items': []}) as mocked:
                rec = evaluate.evaluate_example('nt-0', ds, cfg, FakeClient())
            self.assertEqual(mocked.call_args.args[3], 'English back question')
            self.assertEqual(mocked.call_args.args[4], [])
            self.assertEqual(mocked.call_args.kwargs['target_lang'], 'en')
            self.assertEqual(rec['question']['translation'], ds['nt-0'].utterance)
            self.assertIsNotNone(rec['table'])
            self.assertFalse(rec['section_errors'])

    def test_stanza_backend_metadata_is_not_gliner(self):
        from stanza_entities import StanzaEntityExtractor
        class FakeStanza(Extractor):
            model_name = StanzaEntityExtractor.model_name
            labels = StanzaEntityExtractor.labels
            threshold = None
            tokenizer_name = 'stanza/en'
        result = entities.evaluate(FakeStanza(), Judge(), 'Allen', 'Allen', [], target_lang='en')
        self.assertEqual(result['model'], 'stanza/en/ontonotes_nocharlm')
        self.assertIsNone(result['threshold'])
        self.assertIn('PERSON', result['labels'])

    def test_stanza_rejects_russian_without_initializing_any_model(self):
        from stanza_entities import StanzaEntityExtractor
        extractor = StanzaEntityExtractor.__new__(StanzaEntityExtractor)
        with self.assertRaises(ValueError):
            extractor.extract(['Аллен'], 'ru')

    def test_back_uses_english_and_has_no_cell_tasks(self):
        text, lang = candidate({'quality': {'back_translated_question': 'Allen'}}, 'Аллен', 'back')
        result = entities.evaluate(Extractor(), Judge(), 'Allen', text, [], target_lang=lang)
        self.assertEqual(result['target_language'], 'en')
        self.assertEqual(len(result['items']), 1)
        self.assertEqual(result['items'][0]['location'], 'question')
        self.assertEqual(result['items'][0]['matches'][0]['type_preserved'], 1)

    def test_missing_back_does_not_fall_back_to_russian(self):
        with self.assertRaises(ValueError):
            candidate({}, 'Аллен', 'back')

    def test_forward_uses_russian_dataset_question(self):
        self.assertEqual(candidate({'translated_question': 'other'}, 'Аллен', 'forward'), ('Аллен', 'ru'))

    def test_semantics_does_not_override_detected_types(self):
        result = entities.evaluate(Extractor(), Judge(), 'Allen', 'Аллен', [])
        match = result['items'][0]['matches'][0]
        self.assertEqual(match['meaning_preserved'], 1)
        self.assertEqual(match['type_preserved'], 0)

    def test_duplicate_target_alignment_rejected(self):
        class BadJudge:
            def complete_json(self, system, user, parse):
                return parse({'results': [{'i': 0, 'matches': [
                    {'source_id': i, 'target_id': 0, 'meaning_preserved': 1}
                    for i in range(2)], 'added_target_ids': []}]})
        with self.assertRaises(ValueError):
            J.judge_entities(BadJudge(), [{'i': 0, 'source_entities': ['A', 'B'],
                                          'translation_entities': ['А']}])

    def test_mentions_ignore_table_and_use_trace_translation(self):
        item = mentions.build_items([{'question_text': 'year', 'translated_text': 'году'}],
                                    {}, None, None, {}, '')[0]
        self.assertEqual(set(item['_payload']), {'id', 'source_mention', 'translated_text'})
        resolved = mentions.resolve_item(item, {'correctness': 1, 'note': ''}, '', {})
        self.assertEqual(resolved['correctness'], 1)

    def test_missing_translation_is_unknown(self):
        item = mentions.build_items([{'question_text': 'year'}], {}, None, None, {}, '')[0]
        self.assertIsNone(item['correctness'])
        self.assertIn('translated_text_missing', item['flags'])


if __name__ == '__main__':
    unittest.main()
