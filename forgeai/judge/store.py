"""Reading and writing the DEV side of a baseline or an iteration: what the optimizer is allowed to see.

The test split lives in ``sealed.py``, in a different directory. Nothing in this module reads or names it; the
functions here refuse a path inside a ``sealed`` directory (``tests/test_sealed.py`` enforces both).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forgeai.evals.build import load_eval_set
from forgeai.evals.models import EvalCase
from forgeai.judge.models import CaseRun, CaseScore

DEV_DIR = "dev"
SEALED_DIR = "sealed"


def _guard(path: Path) -> Path:
    if SEALED_DIR in path.parts:
        raise PermissionError(f"{path} is sealed: the test split is not readable from this code path")
    return path


def write_split(directory: Path, prefix: str, runs: list[CaseRun], scores: list[CaseScore], card: dict[str, Any]) -> None:
    """Write one split's agent answers, per-case verdicts, scorecard and layer-1/layer-2 disagreement log."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{prefix}answers.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in runs), encoding="utf-8")
    (directory / f"{prefix}case_scores.jsonl").write_text("".join(s.model_dump_json() + "\n" for s in scores), encoding="utf-8")
    (directory / f"{prefix}scorecard.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    rows = [{"case_id": s.case_id, **d} for s in scores for d in s.disagreements]
    (directory / f"{prefix}judge_disagreements.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def dev_dir(out: Path) -> Path:
    return out / "05_baseline" / DEV_DIR


def load_dev_cases(run_dir: Path) -> list[EvalCase]:
    """The dev cases of an eval set; the test cases are dropped before anything else can look at them."""
    return [c for c in load_eval_set(_guard(run_dir) / "03_evals" / "eval_set.jsonl") if c.split == "dev"]


def read_dev_scorecard(out: Path) -> dict[str, Any]:
    return json.loads((_guard(dev_dir(out)) / "scorecard.json").read_text(encoding="utf-8"))


def read_dev_scores(out: Path) -> list[CaseScore]:
    path = _guard(dev_dir(out)) / "case_scores.jsonl"
    return [CaseScore.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
