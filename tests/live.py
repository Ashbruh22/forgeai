"""Helpers for tests that need a running Ollama with the configured models.

Such tests are marked ``live`` and are skipped LOUDLY (with the reason) when Ollama or a
model is missing. They are never silently passed.
"""

from __future__ import annotations

import httpx
import pytest

from forgeai.config import load_config


def require_models(*names: str) -> None:
    config = load_config()
    try:
        tags = httpx.get(f"{config.ollama_host}/api/tags", timeout=3).json()
    except httpx.HTTPError as exc:
        pytest.skip(f"LIVE TEST SKIPPED: Ollama is not reachable at {config.ollama_host} ({exc})")
    installed = {m["name"] for m in tags.get("models", [])}
    missing = [n for n in names if n not in installed]
    if missing:
        pytest.skip(f"LIVE TEST SKIPPED: models not pulled: {missing}")
