"""Tool registry (spec §5.3): exactly four tools, each tagged with a capability."""

from __future__ import annotations

from forgeai.prompts import load_prompt
from forgeai.tools.base import Tool, ToolContext, ToolError, ToolSpec
from forgeai.tools.calculator import Calculator
from forgeai.tools.csv_lookup import CsvLookup
from forgeai.tools.simple import CurrentDate, Weather

REGISTRY: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec("csv_lookup", "structured_lookup", CsvLookup),
        ToolSpec("calculator", "arithmetic", lambda ctx: Calculator()),
        ToolSpec("current_date", "temporal", CurrentDate),
        ToolSpec("weather", "live_weather", Weather),
    )
}


def build_tools(names: list[str], context: ToolContext) -> dict[str, Tool]:
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ToolError(f"unknown tools {unknown}; registry has {sorted(REGISTRY)}")
    return {n: REGISTRY[n].factory(context) for n in names}


def tool_description(name: str) -> str:
    """The model-facing description of a tool, from its versioned prompt file."""
    return load_prompt(f"tools/{name}").user


__all__ = ["REGISTRY", "Tool", "ToolContext", "ToolError", "build_tools", "tool_description"]
