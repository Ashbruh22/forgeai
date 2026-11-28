"""Freezing an eval set: its hash goes into the run manifest, and everything that scores against it checks the hash.

An optimizer that is allowed to regenerate or edit the eval set it is scored on can pass any bar. After the
set is frozen, a changed byte makes ``assert_eval_set_frozen`` raise, and the iteration loop refuses to start.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from forgeai.config import ForgeConfig
from forgeai.manifest import git_sha, write_manifest

MANIFEST = "manifest.json"
EVAL_SET = Path("03_evals") / "eval_set.jsonl"


class EvalSetNotFrozenError(RuntimeError):
    """The run manifest has no eval-set hash: nothing says which eval set the scores belong to."""


class EvalSetChangedError(RuntimeError):
    """The eval set on disk is not the one that was frozen."""


def eval_set_hash(run_dir: Path) -> str:
    """sha256 of the eval set with line endings normalised, so a checkout that converts LF to CRLF (git on Windows)
    does not change the hash of an unchanged set."""
    return hashlib.sha256((run_dir / EVAL_SET).read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()


def _case_count(run_dir: Path) -> int:
    return sum(1 for line in (run_dir / EVAL_SET).read_text(encoding="utf-8").splitlines() if line.strip())


def freeze_eval_set(run_dir: Path, config: ForgeConfig, digests: dict[str, str | None] | None = None) -> dict[str, Any]:
    """Record the hash of ``run_dir``'s eval set in its manifest (creating the manifest when the run has none)."""
    path = run_dir / MANIFEST
    if not path.is_file():
        write_manifest(path, config, digests)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["eval_set"] = {"path": EVAL_SET.as_posix(), "sha256": eval_set_hash(run_dir), "cases": _case_count(run_dir),
                            "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_sha": git_sha()}
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest["eval_set"]


def assert_eval_set_frozen(run_dir: Path) -> str:
    """Raise unless the eval set on disk hashes to what the manifest froze. Returns the hash."""
    path = run_dir / MANIFEST
    entry = json.loads(path.read_text(encoding="utf-8")).get("eval_set") if path.is_file() else None
    if not entry:
        raise EvalSetNotFrozenError(f"{run_dir}: the manifest records no eval-set hash; run `forgeai freeze` first")
    now = eval_set_hash(run_dir)
    if now != entry["sha256"]:
        raise EvalSetChangedError(f"{run_dir}: eval_set.jsonl changed after it was frozen "
                                  f"(frozen {entry['sha256'][:12]}, now {now[:12]}); refusing to score against it")
    return now
