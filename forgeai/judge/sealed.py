"""The TEST side of a baseline or an iteration: written once, read only by the report.

The optimizer (P7) never sees the test split (spec §6.1, §6.6). Its results are therefore written under a
``sealed/`` directory, by this module, and read back by this module only. No module on the optimizer's import path
may import it (``tests/test_sealed.py`` follows the import graph to make sure).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forgeai.judge.models import CaseRun, CaseScore
from forgeai.judge.store import SEALED_DIR, write_split


def sealed_dir(out: Path) -> Path:
    return out / "05_baseline" / SEALED_DIR


def write_test(out: Path, runs: list[CaseRun], scores: list[CaseScore], card: dict[str, Any]) -> Path:
    directory = sealed_dir(out)
    write_split(directory, "test_", runs, scores, card)
    return directory


def read_test_scorecard(out: Path) -> dict[str, Any]:
    return json.loads((sealed_dir(out) / "test_scorecard.json").read_text(encoding="utf-8"))


def read_test_scores(out: Path) -> list[CaseScore]:
    path = sealed_dir(out) / "test_case_scores.jsonl"
    return [CaseScore.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
