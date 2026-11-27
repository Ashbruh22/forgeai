"""Run lock and stage timing (P5.1 hygiene)."""

from __future__ import annotations

import json
import os

import pytest

from forgeai.runlock import LOCK_NAME, RunLockedError, pid_alive, run_lock
from forgeai.timing import stage_timer, summarize_calls


def test_a_second_build_of_the_same_run_folder_is_refused_while_the_first_is_alive(tmp_path) -> None:
    with run_lock(tmp_path, "evals"):
        assert (tmp_path / LOCK_NAME).is_file()
        with pytest.raises(RunLockedError, match="being built by process"):
            with run_lock(tmp_path, "intake"):
                pytest.fail("the second build must not start")
    assert not (tmp_path / LOCK_NAME).exists()  # released on exit


def test_the_lock_is_released_when_the_build_fails(tmp_path) -> None:
    with pytest.raises(RuntimeError):
        with run_lock(tmp_path, "evals"):
            raise RuntimeError("boom")
    with run_lock(tmp_path, "evals"):  # can be taken again
        pass


def test_a_lock_left_by_a_dead_process_is_taken_over(tmp_path) -> None:
    (tmp_path / LOCK_NAME).write_text(json.dumps({"pid": 2_000_000_000, "stage": "evals", "started": "x"}), encoding="utf-8")
    with run_lock(tmp_path, "evals"):
        assert json.loads((tmp_path / LOCK_NAME).read_text())["pid"] == os.getpid()


def test_pid_alive_knows_this_process_and_a_missing_one() -> None:
    assert pid_alive(os.getpid()) and not pid_alive(2_000_000_000) and not pid_alive(0)


def test_stage_timer_records_wall_time_and_only_its_own_slice_of_the_trace(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({"model": "m", "latency_ms": 99999, "cache_hit": False}) + "\n", encoding="utf-8")  # an earlier stage
    with stage_timer(tmp_path, "evals"):
        with trace.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"model": "qwen3:8b", "latency_ms": 2000, "cache_hit": False, "load_ms": 500, "swap": True}) + "\n")
            handle.write(json.dumps({"model": "qwen3:8b", "latency_ms": 0.3, "cache_hit": True}) + "\n")
    entry = json.loads((tmp_path / "timing.json").read_text())["evals"]
    assert entry["llm_calls"] == 2 and entry["cache_hits"] == 1 and entry["live_llm_seconds"] == 2.0
    assert entry["model_load_seconds"] == 0.5 and entry["model_swaps"] == 1 and entry["by_model"] == {"qwen3:8b": 1}
    with stage_timer(tmp_path, "audit"):
        pass
    assert set(json.loads((tmp_path / "timing.json").read_text())) == {"evals", "audit"}  # stages accumulate


def test_summaries_count_fallbacks() -> None:
    assert summarize_calls([{"fallback_used": True, "latency_ms": 1.0}])["fallbacks"] == 1
