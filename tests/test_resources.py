"""Memory safety: RAM monitor, Ollama settings check, timeouts and degenerate generations (P5.1)."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from forgeai.config import ForgeConfig, ModelSpec
from forgeai.llm import LLMClient, LLMTimeoutError, is_degenerate
from forgeai.manifest import build_manifest
from forgeai.resources import MemoryMonitor, system_memory, verify_ollama_env
from tests.conftest import FakeTransport  # noqa: F401  (fixtures below come from conftest)


class Out(BaseModel):
    answer: str = ""


def test_memory_monitor_reports_peak_and_low_memory_warnings() -> None:
    total, free = system_memory()
    assert total > 0 and 0 < free <= total
    with MemoryMonitor(warn_below_free_pct=101.0, interval_s=0.05) as monitor:  # always "below": must warn
        pass
    summary = monitor.summary()
    assert summary["low_memory_warnings"] >= 1 and summary["system_ram_total_gb"] > 0
    assert summary["peak_system_ram_used_gb"] >= 0 and summary["forgeai_process_peak_gb"] >= 0


def test_per_role_context_and_timeouts_are_configured_and_recorded(tmp_path) -> None:
    config = ForgeConfig(agent_model=ModelSpec(name="qwen3:8b", num_ctx=6144), judge_model=ModelSpec(name="llama3.1:8b", num_ctx=4096),
                         ollama_env={"OLLAMA_NUM_PARALLEL": "1"}, cache_dir=tmp_path)
    assert (config.num_ctx_for("qwen3:8b"), config.num_ctx_for("llama3.1:8b"), config.num_ctx_for("other")) == (6144, 4096, 8192)
    manifest = build_manifest(config, digests={})
    assert manifest["llm"]["num_ctx_per_role"] == {"agent": 6144, "judge": 4096}
    assert manifest["llm"]["timeout_s"] == 180.0 and manifest["llm"]["ollama_env_wanted"] == {"OLLAMA_NUM_PARALLEL": "1"}


def test_the_repo_config_sizes_context_above_the_largest_measured_prompt() -> None:
    from forgeai.config import load_config

    config = load_config()
    assert config.ollama_timeout_s <= 180 and config.ollama_env["OLLAMA_NUM_PARALLEL"] == "1"
    assert config.num_ctx_for(config.agent_model.name) >= 2672 + config.max_output_tokens  # measured prompt + output cap
    assert config.num_ctx_for(config.judge_model.name) >= 699 + config.max_output_tokens


def test_degenerate_output_is_detected_but_ordinary_json_is_not() -> None:
    assert is_degenerate('{"answer": "', done_reason="length")
    assert is_degenerate('{"answer": "' + "ab" * 400 + '"}')
    assert not is_degenerate(json.dumps({"answer": "a normal reply " * 5}))
    assert not is_degenerate(json.dumps({"items": [{"slot": f"s{i}", "fact_id": f"F{i}"} for i in range(12)]}))


def test_a_timeout_is_retried_once_then_the_candidate_falls_back_and_the_stage_continues(make_client, tmp_path) -> None:
    class Timeouts:
        calls = 0

        def chat(self, **kwargs):
            Timeouts.calls += 1
            raise LLMTimeoutError("timed out")

    client = LLMClient(make_client([])[0].config, transport=Timeouts(), trace_path=tmp_path / "t.jsonl")
    result = client.call_structured("r", "_selftest", {"sentence": "x"}, Out, "qwen3:8b", 0.0, fallback=Out(answer="skipped"))
    assert result.answer == "skipped" and Timeouts.calls == 2  # one retry, then skip: not three full timeouts
    record = json.loads((tmp_path / "t.jsonl").read_text().splitlines()[-1])
    assert record["fallback_used"] is True and record["error"].startswith("timeout")


def test_a_degenerate_reply_is_retried_with_a_shortened_history(make_client) -> None:
    client, transport = make_client(['{"answer": "' + "x" * 600, '{"answer": "fine"}'])
    assert client.call_structured("r", "_selftest", {"sentence": "x"}, Out, "qwen3:8b", 0.0).answer == "fine"
    repair = transport.calls[1]["messages"][-2]
    assert len(repair["content"]) <= 300  # the runaway output is not fed back whole


def test_ollama_settings_are_compared_with_the_servers_own_startup_line() -> None:
    report = verify_ollama_env({"OLLAMA_NUM_PARALLEL": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"})
    assert set(report["took_effect"]) == {"OLLAMA_NUM_PARALLEL", "OLLAMA_KV_CACHE_TYPE"}
    assert all(isinstance(v, bool) for v in report["took_effect"].values())
    pytest.importorskip("forgeai")


def test_a_server_error_such_as_a_cuda_oom_is_transient_not_fatal(make_client, tmp_path, monkeypatch) -> None:
    from forgeai.llm import LLMTransientError
    import forgeai.llm as llm

    monkeypatch.setattr(llm, "RECOVERY_PAUSE_S", 0.0)

    class Flaky:
        calls = 0

        def chat(self, **kwargs):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise LLMTransientError("Ollama server error 500: CUDA error: out of memory")
            from forgeai.llm import ChatResult

            return ChatResult(content='{"answer": "ok"}')

    client = LLMClient(make_client([])[0].config, transport=Flaky(), trace_path=tmp_path / "t.jsonl")
    assert client.call_structured("r", "_selftest", {"sentence": "x"}, Out, "qwen3:8b", 0.0).answer == "ok"
    assert Flaky.calls == 2
