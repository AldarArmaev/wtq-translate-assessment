"""Configuration, rubric versioning and .env loading for the WTQ translation evaluation."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Bump this on ANY change to the prompts / rubric text in llm_judge.py.
# rubric_lock.json stores the prompt hash per version and refuses to run if the prompts
# changed without a version bump ("do not change metrics retroactively").
RUBRIC_VERSION = "v1.0"

DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5"}


def load_dotenv(path: Path) -> None:
    """Tiny .env parser (KEY=VALUE lines); existing environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Config:
    root: Path
    dataset_dir: Path
    trace_dir: Path
    out_dir: Path
    sample_size: int = 200
    seed: int = 42
    workers: int = 4
    max_cells: int = 150          # max cells judged per table (0 = no cap)
    cell_batch_size: int = 40     # cells per LLM call
    use_llm: bool = True
    provider: str = "anthropic"   # anthropic | openai (any OpenAI-compatible endpoint)
    model: str = ""
    temperature: float = 0.0
    max_retries: int = 4
    timeout: float = 120.0
    tsv_quoting: str = "none"     # none | minimal  (how dataset/table TSVs are quoted)
    ids_file: Optional[Path] = None
    resume: bool = False
    force: bool = False


def check_rubric_lock(lock_path: Path, version: str, digest: str) -> None:
    """Register the rubric hash for `version`, or fail if the prompts changed under the same version."""
    lock = {}
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    known = lock.get(version)
    if known is None:
        lock[version] = digest
        lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif known != digest:
        raise SystemExit(
            f"Rubric prompts changed but RUBRIC_VERSION is still '{version}' "
            f"(locked hash {known}, current {digest}).\n"
            f"Bump RUBRIC_VERSION in config.py to record a new rubric version."
        )
