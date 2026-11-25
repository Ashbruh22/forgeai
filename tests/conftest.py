"""Shared test helpers.

``FakeTransport`` exists only for unit tests of the LLM wrapper's failure handling.
It is never used by demo runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from forgeai.config import ForgeConfig, ModelSpec
from forgeai.llm import ChatResult, LLMClient


class FakeTransport:
    """Replays scripted replies and records every request it receives."""

    def __init__(self, replies: list[str | ChatResult]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        options: dict[str, Any],
    ) -> ChatResult:
        self.calls.append(
            {"model": model, "messages": [dict(m) for m in messages], "options": options}
        )
        reply = self._replies.pop(0)
        return reply if isinstance(reply, ChatResult) else ChatResult(content=reply)


@pytest.fixture
def config(tmp_path: Path) -> ForgeConfig:
    return ForgeConfig(
        agent_model=ModelSpec(name="qwen3:8b"),
        judge_model=ModelSpec(name="gemma3:4b", temperature=0.0),
        cache_dir=tmp_path / "cache",
    )


@pytest.fixture
def make_client(config: ForgeConfig, tmp_path: Path):
    def _make(replies: list[str | ChatResult]) -> tuple[LLMClient, FakeTransport]:
        transport = FakeTransport(replies)
        client = LLMClient(config, transport=transport, trace_path=tmp_path / "trace.jsonl")
        return client, transport

    return _make
