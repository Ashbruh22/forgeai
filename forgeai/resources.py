"""Memory safety for builds that run on a laptop: RAM sampling, GPU residency of the loaded models, Ollama settings.

* ``MemoryMonitor`` samples system RAM once a second in a background thread and keeps the peak used, the
  lowest free percentage and this process's peak working set. It logs a warning (once per stage) when free
  RAM drops below the configured percentage.
* ``model_residency`` asks Ollama which models are loaded and how much of each sits in GPU memory. A model
  that is not 100% on the GPU runs partly on the CPU, which is slow and memory hungry; callers report it and
  propose a fix instead of letting the run continue silently.
* ``verify_ollama_env`` reads the settings the Ollama SERVER actually started with from its log, so a setting
  that was exported but never reached the server is caught.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("forgeai.resources")


def system_memory() -> tuple[float, float]:
    """``(total_gb, free_gb)`` of physical RAM."""
    if sys.platform == "win32":
        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("tpage", ctypes.c_ulonglong), ("apage", ctypes.c_ulonglong),
                        ("tvirt", ctypes.c_ulonglong), ("avirt", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]

        status = Status()
        status.length = ctypes.sizeof(Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
        return status.total / 2**30, status.avail / 2**30
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        values[key] = int(rest.split()[0]) / 2**20
    return values["MemTotal"], values["MemAvailable"]


def process_peak_gb() -> float:
    """Peak working set of this process so far, in GB (0 where it cannot be read)."""
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong), ("peak", ctypes.c_size_t),
                        ("ws", ctypes.c_size_t), ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
                        ("d", ctypes.c_size_t), ("pagefile", ctypes.c_size_t), ("peakpagefile", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):  # type: ignore[attr-defined]
            return counters.peak / 2**30
        return 0.0
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    except ImportError:
        return 0.0


@dataclass
class MemoryMonitor:
    """Samples RAM while a stage runs. Use as a context manager; ``summary()`` is JSON-ready."""

    warn_below_free_pct: float = 15.0
    interval_s: float = 1.0
    total_gb: float = 0.0
    min_free_gb: float = field(default=1e9)
    warnings: int = 0
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def _sample(self) -> None:
        total, free = system_memory()
        self.total_gb = total
        self.min_free_gb = min(self.min_free_gb, free)
        if free / total * 100 < self.warn_below_free_pct:
            if self.warnings == 0:
                log.warning("free RAM is %.1f GB (%.0f%% of %.1f GB), below the %.0f%% limit",
                            free, free / total * 100, total, self.warn_below_free_pct)
            self.warnings += 1

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self) -> MemoryMonitor:
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._sample()

    def summary(self) -> dict[str, Any]:
        used = self.total_gb - self.min_free_gb
        return {
            "system_ram_total_gb": round(self.total_gb, 1),
            "peak_system_ram_used_gb": round(used, 1),
            "min_free_ram_gb": round(self.min_free_gb, 1),
            "min_free_ram_pct": round(self.min_free_gb / self.total_gb * 100, 1) if self.total_gb else None,
            "low_memory_warnings": self.warnings,
            "forgeai_process_peak_gb": round(process_peak_gb(), 2),
        }


def model_residency(host: str) -> list[dict[str, Any]]:
    """Loaded models with their GPU share (``size_vram / size``) and context length, from Ollama's /api/ps."""
    models = httpx.get(f"{host}/api/ps", timeout=10).json().get("models", [])
    return [{"name": m["name"], "size_gb": round(m["size"] / 2**30, 2), "gpu_pct": round(100 * m.get("size_vram", 0) / max(m["size"], 1)),
             "context": m.get("context_length")} for m in models]


def load_and_check(host: str, name: str, num_ctx: int) -> dict[str, Any]:
    """Load ``name`` with the context it will really use, then report where it sits (GPU/CPU split)."""
    from forgeai.llm import warm_model

    warm_model(host, name, num_ctx)
    loaded = next((m for m in model_residency(host) if m["name"] == name), None)
    if loaded is None:
        return {"name": name, "gpu_pct": None, "note": "not listed by ollama ps after loading"}
    loaded["fully_on_gpu"] = loaded["gpu_pct"] == 100
    return loaded


def verify_ollama_env(expected: dict[str, str]) -> dict[str, Any]:
    """Compare the wanted settings with those on the server's most recent start-up line in its log."""
    log_path = Path(os.environ.get("LOCALAPPDATA", "")) / "Ollama" / "server.log"
    seen: dict[str, str] = {}
    if log_path.is_file():
        for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
            if "server config" in line and "OLLAMA_NUM_PARALLEL" in line:
                for key in expected:
                    marker = f"{key}:"
                    if marker in line:
                        token = line.split(marker, 1)[1].split()[0].strip("]")
                        seen[key] = "" if token.endswith(":") else token  # an unset value is followed by the next key
                break
    norm = {"1": "true", "true": "true"}
    ok = {k: (seen.get(k) == v or (k == "OLLAMA_FLASH_ATTENTION" and norm.get(seen.get(k, ""), "") == norm.get(v, "x")))
          for k, v in expected.items()}
    return {"expected": expected, "server_log": seen, "took_effect": ok, "log_file": str(log_path), "log_found": log_path.is_file()}
