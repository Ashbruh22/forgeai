"""A lock file per run folder, so two builds can never write the same run (P5.1 hygiene).

``run_lock`` creates ``<run>/.forgeai.lock`` atomically and removes it on exit. A second build of
the same folder raises ``RunLockedError`` while the first is alive. A lock left behind by a process
that no longer exists (a crash, a killed terminal) is taken over, and the takeover is reported.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK_NAME = ".forgeai.lock"


class RunLockedError(Exception):
    """Another live process is building this run folder."""


def pid_alive(pid: int) -> bool:
    """Is a process with this id running? (``os.kill(pid, 0)`` would TERMINATE it on Windows, so ctypes is used there.)"""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_lock(run_dir: Path) -> dict | None:
    path = run_dir / LOCK_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@contextmanager
def run_lock(run_dir: Path, stage: str) -> Iterator[None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / LOCK_NAME
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = read_lock(run_dir) or {}
            if pid_alive(int(holder.get("pid", 0))):
                raise RunLockedError(
                    f"{run_dir} is being built by process {holder.get('pid')} (stage {holder.get('stage')!r}, "
                    f"since {holder.get('started')}). Wait for it or use another run folder.") from None
            path.unlink(missing_ok=True)  # stale: its process is gone
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "stage": stage, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, handle)
        break
    else:  # pragma: no cover - two attempts both lost a race
        raise RunLockedError(f"could not lock {run_dir}")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
