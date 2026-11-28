"""The ruler, frozen: hash of the judge configuration, the judge prompt versions and the scorer's source code.

Scores of different iterations are only comparable if they were produced by the same judge. The fingerprint goes into
the run manifest next to the eval-set hash (``forgeai freeze``); the baseline and the optimizer loop check both and
stop if either changed, including between two cycles of one run.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from forgeai.config import ForgeConfig
from forgeai.manifest import git_sha, prompt_versions

PACKAGE = Path(__file__).resolve().parents[1]
# Every module whose behaviour changes a verdict: the two layers, claim handling, matching, units, the contract check.
SCORER_FILES = (
    "judge/layer1.py", "judge/layer2.py", "judge/claims.py", "judge/normalize.py", "judge/score.py", "judge/metrics.py",
    "judge/nli.py", "judge/models.py", "evals/matching.py", "evals/units.py", "textmatch.py", "runner/contract.py",
)
MANIFEST = "manifest.json"


class JudgeNotFrozenError(RuntimeError):
    """The run manifest records no judge fingerprint."""


class JudgeChangedError(RuntimeError):
    """The judge differs from the one that was frozen: scores would not be comparable."""


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()


def judge_fingerprint(config: ForgeConfig) -> dict[str, Any]:
    """Everything that decides a verdict, and one hash over all of it."""
    settings = {
        "judge_model": config.judge_model.model_dump(exclude={"digest"}),
        "judge": config.judge.model_dump(),
        "fact_fuzzy_score": config.thresholds.fact_fuzzy_score,
        "max_output_tokens": config.max_output_tokens,
        "think": config.think,
    }
    prompts = {k: v for k, v in prompt_versions().items() if k.startswith("judge/")}
    code = {name: _file_hash(PACKAGE / name) for name in SCORER_FILES}
    body = {"config": settings, "prompt_versions": prompts, "scorer_code": code}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
    return {**body, "sha256": digest}


def freeze_judge(run_dir: Path, config: ForgeConfig) -> dict[str, Any]:
    path = run_dir / MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    manifest["judge"] = {**judge_fingerprint(config), "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_sha": git_sha()}
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest["judge"]


def _differences(frozen: dict[str, Any], now: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for part in ("config", "prompt_versions", "scorer_code"):
        before, after = frozen.get(part, {}), now[part]
        for key in sorted(set(before) | set(after)):
            if before.get(key) != after.get(key):
                found.append(f"{part}.{key}")
    return found


def assert_judge_frozen(run_dir: Path, config: ForgeConfig) -> str:
    """Raise unless the judge configuration, prompt versions and scorer code match the frozen fingerprint."""
    path = run_dir / MANIFEST
    entry = json.loads(path.read_text(encoding="utf-8")).get("judge") if path.is_file() else None
    if not entry:
        raise JudgeNotFrozenError(f"{run_dir}: the manifest records no judge fingerprint; run `forgeai freeze` first")
    now = judge_fingerprint(config)
    if now["sha256"] != entry["sha256"]:
        raise JudgeChangedError(f"{run_dir}: the judge changed after it was frozen ({', '.join(_differences(entry, now)) or 'hash differs'}); "
                                "scores before and after would not be comparable")
    return now["sha256"]
