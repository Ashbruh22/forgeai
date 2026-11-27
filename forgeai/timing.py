"""Wall-clock time per stage of a build, from the stage's own timer and its slice of ``trace.jsonl``.

``stage_timer`` appends one entry per stage to ``<run>/timing.json``: wall seconds, the number of
model calls, how many came from the cache, the summed latency of the calls that really ran, and the
time spent loading models. A rerun with a warm cache is therefore visibly cheaper than a cold build,
and the numbers say which is which.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from forgeai.resources import MemoryMonitor


def _trace_slice(trace: Path, offset: int) -> list[dict[str, Any]]:
    if not trace.is_file():
        return []
    with trace.open("rb") as handle:
        handle.seek(offset)
        lines = handle.read().decode("utf-8", errors="replace").splitlines()
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def summarize_calls(rows: list[dict[str, Any]]) -> dict[str, Any]:
    live = [r for r in rows if not r.get("cache_hit")]
    return {
        "llm_calls": len(rows), "cache_hits": len(rows) - len(live),
        "live_llm_seconds": round(sum(r.get("latency_ms", 0.0) for r in live) / 1000, 1),
        "model_load_seconds": round(sum(r.get("load_ms", 0.0) for r in live) / 1000, 1),
        "model_swaps": sum(bool(r.get("swap")) for r in rows),
        "fallbacks": sum(bool(r.get("fallback_used")) for r in rows),
        "by_model": {m: sum(1 for r in live if r.get("model") == m) for m in sorted({r.get("model") for r in live if r.get("model")})},
    }


@contextmanager
def stage_timer(run_dir: Path, stage: str, warn_below_free_pct: float = 15.0) -> Iterator[None]:
    trace = run_dir / "trace.jsonl"
    offset = trace.stat().st_size if trace.is_file() else 0
    started = time.perf_counter()
    monitor = MemoryMonitor(warn_below_free_pct=warn_below_free_pct)
    monitor.__enter__()
    try:
        yield
    finally:
        monitor.__exit__(None, None, None)
        entry = {"wall_seconds": round(time.perf_counter() - started, 1), **summarize_calls(_trace_slice(trace, offset)),
                 "ram": monitor.summary(), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with trace.open("a", encoding="utf-8") as handle:  # peak RAM per stage is also part of the trace
            handle.write(json.dumps({"event": "stage_resources", "stage": stage, **entry["ram"]}) + chr(10))
        path = run_dir / "timing.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data[stage] = entry
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
