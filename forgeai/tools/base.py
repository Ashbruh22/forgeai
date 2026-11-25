"""Tool interface, execution context and registry metadata."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import httpx


class ToolError(Exception):
    """A tool call failed in a way the model should be told about (bad args, no data)."""


@dataclass
class ToolContext:
    """Everything a tool may need from the outside world, injectable for tests."""

    corpus_dir: Path | None = None
    today: Callable[[], date] = date.today
    http: httpx.Client = field(default_factory=lambda: httpx.Client(timeout=15.0))


class Tool(Protocol):
    name: str

    def run(self, args: dict[str, Any]) -> str:
        """Execute with model-supplied args; raise ``ToolError`` with a helpful message."""
        ...


@dataclass(frozen=True)
class ToolSpec:
    """Registry entry. ``capability`` is the tag requirements use to ask for a tool."""

    name: str
    capability: str
    factory: Callable[[ToolContext], Tool]
