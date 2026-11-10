"""English Stanza OntoNotes NER only. Does not import or initialize GLiNER."""
import threading
import time

LABELS = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT',
          'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY',
          'QUANTITY', 'ORDINAL', 'CARDINAL']


class StanzaEntityExtractor:
    model_name = 'stanza/en/ontonotes_nocharlm'
    labels = LABELS
    threshold = None
    tokenizer_name = 'stanza/en'

    def __init__(self, model_dir):
        started = time.perf_counter()
        import torch
        import stanza
        torch.set_num_threads(4)
        self._lock = threading.Lock()
        self.nlp = stanza.Pipeline('en', processors={'tokenize': 'combined',
                                  'ner': 'ontonotes_nocharlm'}, package=None,
                                  dir=str(model_dir / 'stanza'), use_gpu=False,
                                  download_method=stanza.DownloadMethod.REUSE_RESOURCES)
        self.setup_seconds = time.perf_counter() - started

    def extract(self, texts, lang):
        if lang != 'en':
            raise ValueError('English OntoNotes model requires English on both sides')
        with self._lock:
            return {text: [{'text': e.text, 'type': e.type, 'start': e.start_char,
                            'end': e.end_char} for e in self.nlp(text).ents]
                    for text in dict.fromkeys(texts) if text}
