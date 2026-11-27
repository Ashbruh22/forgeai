"""Run manifest: everything needed to reproduce a run (models, digests, prompts, git SHA)."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import forgeai
from forgeai.config import ForgeConfig
from forgeai.llm import fetch_model_digests
from forgeai.prompts import PROMPTS_DIR, load_prompt


def prompt_versions() -> dict[str, int]:
    """Version of every prompt file, keyed by prompt id (e.g. ``ingest/summary``)."""
    return {
        path.relative_to(PROMPTS_DIR).with_suffix("").as_posix(): load_prompt(
            path.relative_to(PROMPTS_DIR).with_suffix("").as_posix()
        ).version
        for path in sorted(PROMPTS_DIR.rglob("*.md"))
    }


def git_sha(repo: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo or Path(__file__).parents[1],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def build_manifest(config: ForgeConfig, digests: dict[str, str | None] | None = None) -> dict[str, Any]:
    """Assemble the manifest. Pass ``digests`` to avoid a live Ollama query (tests)."""
    names = [config.agent_model.name, config.judge_model.name]
    digests = digests if digests is not None else fetch_model_digests(config.ollama_host, names)
    return {
        "forgeai_version": forgeai.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "python": platform.python_version(),
        "models": {
            "agent": {**config.agent_model.model_dump(), "digest": digests.get(config.agent_model.name)},
            "judge": {**config.judge_model.model_dump(), "digest": digests.get(config.judge_model.name)},
        },
        "llm": {
            "think": config.think, "num_ctx": config.num_ctx,
            "num_ctx_per_role": {"agent": config.num_ctx_for(config.agent_model.name), "judge": config.num_ctx_for(config.judge_model.name)},
            "max_output_tokens": config.max_output_tokens, "timeout_s": config.ollama_timeout_s,
            "ollama_max_loaded_models": os.environ.get("OLLAMA_MAX_LOADED_MODELS"),
            "ollama_env_wanted": config.ollama_env,
            "ollama_env_process": {k: os.environ.get(k) for k in config.ollama_env},
        },
        "embedding": {"name": config.embedding_model, "revision": config.embedding_revision},
        "thresholds": config.thresholds.model_dump(),
        "limits": config.limits.model_dump(),
        "prompt_versions": prompt_versions(),
    }


def write_manifest(path: Path, config: ForgeConfig, digests: dict[str, str | None] | None = None) -> dict[str, Any]:
    manifest = build_manifest(config, digests)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
