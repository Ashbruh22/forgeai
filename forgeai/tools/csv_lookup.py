"""csv_lookup: filter and project rows of a CSV file from the agent's corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from forgeai.tools.base import ToolContext, ToolError

MAX_ROWS = 20
_OPS = {"=", "==", "!=", "<", "<=", ">", ">=", "contains"}


def _load(corpus_dir: Path, file: str) -> pd.DataFrame:
    path = (corpus_dir / file).resolve()
    if corpus_dir.resolve() not in path.parents:
        raise ToolError("file must be inside the agent's document folder")
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise ToolError(f"no such CSV file: {file}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _mask(frame: pd.DataFrame, column: str, condition: Any) -> pd.Series:
    if column not in frame.columns:
        raise ToolError(f"unknown column {column!r}; columns are {list(frame.columns)}")
    op, value = "=", condition
    if isinstance(condition, dict):
        op, value = str(condition.get("op", "=")), condition.get("value")
    if op not in _OPS:
        raise ToolError(f"unsupported operator {op!r}; use one of {sorted(_OPS)}")
    series = frame[column].str.strip()
    if op in ("<", "<=", ">", ">="):
        numeric = pd.to_numeric(series, errors="coerce")
        try:
            target = float(value)
        except (TypeError, ValueError) as exc:
            raise ToolError(f"{op} needs a numeric value, got {value!r}") from exc
        return {"<": numeric < target, "<=": numeric <= target, ">": numeric > target, ">=": numeric >= target}[op]
    text = str(value).strip().lower()
    lowered = series.str.lower()
    if op == "contains":
        return lowered.str.contains(text, regex=False)
    equal = lowered == text
    return equal if op in ("=", "==") else ~equal


class CsvLookup:
    name = "csv_lookup"

    def __init__(self, context: ToolContext) -> None:
        self._context = context

    def run(self, args: dict[str, Any]) -> str:
        if self._context.corpus_dir is None:
            raise ToolError("no document folder is attached to this agent")
        file = args.get("file")
        if not isinstance(file, str):
            raise ToolError("csv_lookup needs a string argument 'file'")
        frame = _load(self._context.corpus_dir, file)
        filters = args.get("filters") or {}
        if not isinstance(filters, dict):
            raise ToolError("'filters' must be an object of column -> value")
        mask = pd.Series(True, index=frame.index)
        for column, condition in filters.items():
            mask &= _mask(frame, column, condition)
        matched = frame[mask]
        columns = args.get("columns") or list(frame.columns)
        unknown = [c for c in columns if c not in frame.columns]
        if unknown:
            raise ToolError(f"unknown columns {unknown}; columns are {list(frame.columns)}")
        lines = [
            " | ".join(f"{c}: {row[c]}" for c in columns) for _, row in matched.head(MAX_ROWS).iterrows()
        ]
        header = f"{len(matched)} matching rows"
        if len(matched) > MAX_ROWS:
            header += f" (showing first {MAX_ROWS})"
        return header + ("\n" + "\n".join(lines) if lines else "")
