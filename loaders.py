"""Loading of the translated dataset, traces and tables + reproducible sampling."""
from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

csv.field_size_limit(2**31 - 1)


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _quoting(name: str) -> int:
    return csv.QUOTE_NONE if name == "none" else csv.QUOTE_MINIMAL


# --------------------------------------------------------------------------- dataset
@dataclass(frozen=True)
class Example:
    id: str
    utterance: str   # RUSSIAN translated question (the thing under evaluation)
    context: str     # path of the translated table, relative to dataset/


def load_dataset(path: Path, quoting: str = "none") -> dict[str, Example]:
    out: dict[str, Example] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=_quoting(quoting))
        for row in reader:
            ex_id = (row.get("id") or "").strip()
            if not ex_id:
                continue
            # NOTE: `targetValue` is deliberately never read: answers are not part of this evaluation.
            out[ex_id] = Example(ex_id, row.get("utterance") or "", (row.get("context") or "").strip())
    return out


def sample_ids(all_ids: Iterable[str], k: int, seed: int) -> list[str]:
    """Reproducible sample: ids are put in a canonical (natural) order before sampling."""
    ids = sorted(all_ids, key=natural_key)
    if k <= 0 or k >= len(ids):
        return ids
    rng = random.Random(seed)
    return sorted(rng.sample(ids, k), key=natural_key)


# --------------------------------------------------------------------------- traces
def find_trace_dir(root: Path) -> Path:
    cands = sorted(root.glob("*/traces/training"))
    if len(cands) == 1:
        return cands[0]
    raise SystemExit(
        f"Cannot auto-detect trace dir under {root} (found {len(cands)} candidates); pass --trace-dir."
    )


def load_trace(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- tables
def clean_cell(s: str) -> str:
    """Normalise whitespace/escapes identically for source and translated cells."""
    s = s.replace("\\n", " ").replace("\n", " ").replace("\r", " ").replace("\\p", "|").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Table:
    header: list[str]
    rows: list[list[str]]

    @property
    def n_rows(self) -> int:       # data rows (header excluded)
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return len(self.header)

    def get(self, row: Optional[int], col: int) -> Optional[str]:
        """row=None -> header cell; row=i -> data row i (0-based). None if the coordinate does not exist."""
        if col is None or col < 0:
            return None
        try:
            if row is None:
                return self.header[col]
            if row < 0:
                return None
            return self.rows[row][col]
        except IndexError:
            return None


def load_table(path: Path, tsv_quoting: str = "none") -> Table:
    """Read a .tsv (tab) or .csv (delimiter detected: tab if the first line has a tab, else comma)."""
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        first = f.readline()
        f.seek(0)
        if path.suffix.lower() == ".tsv" or "\t" in first:
            reader = csv.reader(f, delimiter="\t", quoting=_quoting(tsv_quoting))
        else:
            reader = csv.reader(f)
        grid = [[clean_cell(c) for c in row] for row in reader if row]
    if not grid:
        return Table([], [])
    return Table(grid[0], grid[1:])


def resolve_source_table(root: Path, source_context: str) -> Path:
    p = root / source_context
    alt = p.with_suffix(".tsv")
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"TSV table not found: {p}")
