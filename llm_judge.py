"""LLM-as-a-judge: atomic binary verdicts, strict JSON, temperature 0, on-disk cache.

The LLM only judges semantic aspects. It never produces scores or percentages;
all aggregation is done by Python (metrics.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from config import Config, DEFAULT_MODELS


class JudgeError(RuntimeError):
    pass


# ============================================================================= RUBRIC (versioned!)
# Any edit below changes RUBRIC_HASH -> bump RUBRIC_VERSION in config.py (enforced by rubric_lock.json).

SYSTEM_QUESTION = """You are an expert English-Russian translation reviewer. You evaluate the Russian translation of a question from the WikiTableQuestions (WTQ) dataset. WTQ questions are asked about Wikipedia tables and are answered by reasoning over the table (lookup, comparison, counting, sums, averages, superlatives, ordering, differences, etc.).

You judge ONLY the translation against the English source. Do NOT try to answer the question and do NOT judge whether it is answerable.

Give four independent binary verdicts (1 = OK, 0 = problem) and one critical-error flag.

1. meaning_preserved
   1 = the Russian question asks about the same thing as the English one: same object of the question, action, relations, conditions, scope, time reference and all essential information.
   0 = something essential is distorted, missing or invented.

2. key_elements_correct
   1 = all key elements are translated correctly: person names, organisations, league/competition names, geographic names, dates, numbers, units, domain-specific terms, other entities. If the question contains no such elements, answer 1.
   0 = at least one substantive error in such an element (wrong entity, wrong number or date, wrong unit, mistranslated term, a name garbled beyond recognition).
   Names may be transliterated, translated with an established Russian equivalent, or left in Latin script; any of these is fine when the entity stays clearly the same. Normal Russian inflection of names and numerals is expected and is not an error.

3. logic_preserved
   1 = the logic is fully preserved, including: before/after, more/less, highest/lowest, first/last, at least/at most, negation, conditions, and/or, count, sum, average, max, min, difference, ranking and any other operation or relation that affects the answer.
   0 = any of these has been changed, weakened, lost or added.
   Example: "What team scored the most goals?" -> "Какая команда забила меньше всего голов?" is logic_preserved = 0.

4. russian_naturalness
   1 = natural, grammatical Russian that a native speaker would accept as a normal question: correct grammar and agreement, syntax, word order, lexis; no calques; easy to read.
   0 = a noticeable problem (ungrammatical, clearly unnatural word order, calque, wrong agreement, awkward or misleading wording).
   Do NOT penalise capitalisation or the presence/absence of a final question mark when it simply follows the style of the source.

5. critical_error
   1 = there is an error that could plausibly change the meaning or the correct answer, e.g. first<->last, more<->less, highest<->lowest, before<->after, a wrong aggregation, a wrong number, a wrong date, a wrong entity, an essential condition lost, or a table-related value distorted. Pure style problems are never critical.
   0 = otherwise.
   If logic_preserved = 0 because of a change that affects the answer, critical_error must be 1.

Judge every criterion independently: do not lower one criterion because another one failed, and do not give 0 for a harmless paraphrase.

Return ONLY a JSON object (no markdown, no commentary) in exactly this form:
{"issues": ["<very short description of each real problem; empty list if none>"], "meaning_preserved": 0 or 1, "key_elements_correct": 0 or 1, "logic_preserved": 0 or 1, "russian_naturalness": 0 or 1, "critical_error": 0 or 1}"""

QUESTION_USER = """Domain of the table: {domain}

English source question:
<<<
{src}
>>>

Russian translation:
<<<
{tr}
>>>"""

SYSTEM_CELLS = """You are an expert English-Russian translation reviewer. You evaluate the Russian translation of individual cells of a table from the WikiTableQuestions dataset (the English table was translated cell by cell into Russian).

You receive a JSON object with the table domain and a list of cells. Each cell has: "i" (id), "column" (the English header of its column, for context only), "source" (English cell text) and "translation" (Russian cell text). Judge every cell independently.

For each cell give three binary verdicts (1 = OK, 0 = problem):

- correct: the Russian text is a faithful translation of the English cell: it conveys the same meaning and nothing in it is mistranslated or distorted. Person, team and place names may be transliterated, translated with an established Russian equivalent, or kept in Latin script - all fine if clearly the same entity. Standard abbreviations may be kept or localised. Do not penalise stylistic variation.
- information_preserved: every piece of information of the English cell is still present in the Russian cell (no dropped list items, qualifiers, notes, parenthesised parts, units, etc.).
- no_added_information: the Russian cell contains nothing that is absent from the English cell (no added words that carry extra facts, qualifiers, divisions, units or explanations). Grammatical necessities of Russian (case endings, "г." after a year, etc.) are not added information.

Examples: English "1st, Western" -> Russian "1-е место": information_preserved = 0 ("Western" is lost). English "1st" -> Russian "1-е место в Западном дивизионе": no_added_information = 0.

Judge strictly the cell against its source; ignore any question the table may be used for.

Return ONLY a JSON object (no markdown):
{"results": [{"i": <id>, "correct": 0 or 1, "information_preserved": 0 or 1, "no_added_information": 0 or 1, "note": "<short reason if any verdict is 0, otherwise empty string>"}, ...]}
Include exactly one result per input cell."""

SYSTEM_MENTIONS = """You are an expert English-Russian translation reviewer. In the WikiTableQuestions dataset a question mentions elements of a table (column headers or cell values). Both the question and the table were translated from English to Russian. You check whether each table mention survived the translation correctly.

Input is a JSON object with:
- "source_question" and "translated_question";
- "mentions": a list; each mention has "id", "source_mention" (the wording in the English question), "source_element_type" ("header" = column header, "cell" = cell value), "source_table_text" (the English header/cell the mention refers to), "translated_table_text" (its translation in the Russian table) and optionally "other_translated_table_texts" (translations of other cells holding the same English value).

For every mention produce:

1. span: copy, character by character, the shortest contiguous fragment of "translated_question" that renders "source_mention" in Russian (keep the inflected form used in the question). Include only words that translate words of source_mention; do not include extra neighbouring words. If the translated question has no counterpart of the mention, use "".

2. correctness (chain check): source_mention -> source_table_text -> translated_table_text -> span.
   1 = the span refers to the same table element (same entity / value / column) as the English mention, AND translated_table_text is a correct translation of source_table_text.
   0 = the span is missing, or refers to a different entity/value/column, or translated_table_text is not a correct translation of source_table_text.

3. semantic_consistency: compare ONLY the span with translated_table_text (the Russian side).
   1 = they mean the same thing, even if the form differs: inflection/case, word order, capitalisation, transliteration variant, abbreviation vs full form, a harmless generic word added or dropped. Example: table "Премьер-лига" and span "в английской Премьер-лиге" are consistent.
   0 = they denote different entities/values/columns or different meanings, or there is no span.

Return ONLY a JSON object (no markdown):
{"results": [{"id": "<mention id>", "span": "<fragment or empty string>", "correctness": 0 or 1, "semantic_consistency": 0 or 1, "note": "<short reason if any verdict is 0, otherwise empty string>"}, ...]}
Include exactly one result per input mention."""

RUBRIC_HASH = hashlib.sha256(
    "\x00".join([SYSTEM_QUESTION, QUESTION_USER, SYSTEM_CELLS, SYSTEM_MENTIONS]).encode("utf-8")
).hexdigest()[:12]


# ============================================================================= parsing / validation
def parse_json_object(text: str) -> dict:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < i:
        raise ValueError("no JSON object found in the answer")
    obj = json.loads(t[i:j + 1])
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value is not an object")
    return obj


def as_bit(v: Any, name: str) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)) and v in (0, 1):
        return int(v)
    if isinstance(v, str) and v.strip() in ("0", "1"):
        return int(v.strip())
    raise ValueError(f"'{name}' must be 0 or 1, got {v!r}")


def _parse_question(obj: dict) -> dict:
    out = {k: as_bit(obj.get(k), k) for k in
           ("meaning_preserved", "key_elements_correct", "logic_preserved", "russian_naturalness", "critical_error")}
    issues = obj.get("issues") or []
    out["issues"] = [str(x) for x in issues] if isinstance(issues, list) else [str(issues)]
    return out


def _parse_results(obj: dict, ids: list, id_key: str, fields: tuple) -> dict:
    results = obj.get("results")
    if not isinstance(results, list):
        raise ValueError("'results' must be a list")
    got = {}
    for r in results:
        if not isinstance(r, dict) or id_key not in r:
            raise ValueError(f"every result needs '{id_key}'")
        rid = r[id_key]
        rid = int(rid) if isinstance(ids[0], int) else str(rid)
        got[rid] = r
    if set(got) != set(ids):
        raise ValueError(f"expected results for ids {ids}, got {sorted(got, key=str)}")
    out = {}
    for rid, r in got.items():
        item = {f: as_bit(r.get(f), f) for f in fields}
        item["note"] = str(r.get("note") or "")
        if "span" in r or id_key == "id":
            item["span"] = str(r.get("span") or "").strip()
        out[rid] = item
    return out


def _parse_cells(obj: dict, ids: list) -> dict:
    return _parse_results(obj, ids, "i", ("correct", "information_preserved", "no_added_information"))


def _parse_mentions(obj: dict, ids: list) -> dict:
    return _parse_results(obj, ids, "id", ("correctness", "semantic_consistency"))


# ============================================================================= client
class LLMClient:
    def __init__(self, cfg: Config, cache_dir: Path):
        self.cfg = cfg
        self.provider = cfg.provider
        self.model = cfg.model or os.environ.get("LLM_MODEL") or DEFAULT_MODELS.get(cfg.provider, "")
        if not self.model:
            raise SystemExit(f"No model for provider '{cfg.provider}': set LLM_MODEL in .env or pass --model.")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._skip: set[str] = set()
        if cfg.provider == "anthropic":
            import anthropic
            self._sdk = anthropic.Anthropic(max_retries=0, timeout=cfg.timeout)      # key from ANTHROPIC_API_KEY
        elif cfg.provider == "openai":
            import openai
            self._sdk = openai.OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None,
                                      max_retries=0, timeout=cfg.timeout)            # key from OPENAI_API_KEY
        else:
            raise SystemExit(f"Unknown provider: {cfg.provider}")

    # ---- cache
    def _key(self, system: str, user: str) -> str:
        blob = json.dumps([self.provider, self.model, self.cfg.temperature, RUBRIC_HASH, system, user],
                          ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[dict]:
        p = self.cache_dir / f"{key}.json"
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                return None
        return None

    def _cache_put(self, key: str, obj: dict) -> None:
        tmp = self.cache_dir / f"{key}.{threading.get_ident()}.tmp"
        tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.cache_dir / f"{key}.json")

    # ---- raw call
    def _call(self, system: str, user: str) -> str:
        if self.provider == "anthropic":
            kw = {} if "temperature" in self._skip else {"temperature": self.cfg.temperature}
            resp = self._sdk.messages.create(model=self.model, max_tokens=8192, system=system,
                                             messages=[{"role": "user", "content": user}], **kw)
            if resp.stop_reason == "max_tokens":
                raise ValueError("model output was truncated")
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        kw = {}
        if "temperature" not in self._skip:
            kw["temperature"] = self.cfg.temperature
        if "response_format" not in self._skip:
            kw["response_format"] = {"type": "json_object"}
        resp = self._sdk.chat.completions.create(
            model=self.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
        if resp.choices[0].finish_reason == "length":
            raise ValueError("model output was truncated")
        return resp.choices[0].message.content or ""

    def _should_retry(self, e: Exception, attempt: int) -> bool:
        status = getattr(e, "status_code", None)
        msg = str(e).lower()
        for param in ("temperature", "response_format"):          # parameter not supported by this model/endpoint
            if status == 400 and param in msg and param not in self._skip:
                self._skip.add(param)
                return True
        transient = status in (408, 409, 429, 500, 502, 503, 504, 529) or \
            type(e).__name__ in {"APIConnectionError", "APITimeoutError"}
        if transient and attempt < self.cfg.max_retries:
            time.sleep(min(60.0, 2 ** attempt + random.random()))
            return True
        return False

    # ---- public
    def complete_json(self, system: str, user: str, parse: Callable[[dict], Any]) -> Any:
        key = self._key(system, user)
        cached = self._cache_get(key)
        if cached is not None:
            try:
                return parse(cached)
            except ValueError:
                pass
        feedback, last = "", None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                text = self._call(system, user + feedback)
            except ValueError as e:                                # truncation
                last, feedback = e, f"\n\nYour previous answer was unusable ({e}). Keep notes very short."
                continue
            except Exception as e:  # noqa: BLE001 - SDK exception types differ per provider
                if self._should_retry(e, attempt):
                    continue
                raise JudgeError(f"API error: {e}") from e
            try:
                obj = parse_json_object(text)
                result = parse(obj)
            except ValueError as e:                                # includes JSONDecodeError
                last, feedback = e, f"\n\nYour previous answer was invalid ({e}). Return ONLY the JSON object in the required format."
                continue
            self._cache_put(key, obj)
            return result
        raise JudgeError(f"no valid JSON after {self.cfg.max_retries + 1} attempts: {last}")


# ============================================================================= task-level helpers
def judge_question(client, source: str, translation: str, domain: Optional[str]) -> dict:
    user = QUESTION_USER.format(domain=domain or "unknown", src=source, tr=translation)
    return client.complete_json(SYSTEM_QUESTION, user, _parse_question)


def _judge_batches(client, system: str, items: list, id_key: str, parser, extra: dict, batch_size: int) -> dict:
    """Run batched judgements; on persistent failure bisect the batch so one bad item cannot kill the rest."""
    out: dict = {}

    def run(batch: list) -> None:
        ids = [b[id_key] for b in batch]
        user = json.dumps({**extra, ("cells" if id_key == "i" else "mentions"): batch}, ensure_ascii=False)
        try:
            out.update(client.complete_json(system, user, lambda o: parser(o, ids)))
        except JudgeError:
            if len(batch) == 1:
                out[batch[0][id_key]] = None
            else:
                mid = len(batch) // 2
                run(batch[:mid])
                run(batch[mid:])

    for s in range(0, len(items), batch_size):
        run(items[s:s + batch_size])
    return out


def judge_cells(client, items: list, domain: Optional[str], batch_size: int = 40) -> dict:
    """items: [{"i": int, "column": str, "source": str, "translation": str}] -> {i: verdict | None}"""
    return _judge_batches(client, SYSTEM_CELLS, items, "i", _parse_cells,
                          {"domain": domain or "unknown"}, batch_size)


def judge_mentions(client, source_question: str, translated_question: str, items: list, batch_size: int = 25) -> dict:
    """items: mention payloads with string ids -> {id: verdict | None}"""
    return _judge_batches(client, SYSTEM_MENTIONS, items, "id", _parse_mentions,
                          {"source_question": source_question, "translated_question": translated_question},
                          batch_size)
