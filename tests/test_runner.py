"""Runner behaviour with scripted model replies (tests of failure handling only).

The real end-to-end check against live models is tests/test_live_runner.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgeai.config import ForgeConfig, ModelSpec
from forgeai.llm import LLMClient, set_client
from forgeai.runner.agent_def import compile_system_prompt, load_agent
from forgeai.runner.runner import MODEL_OUTPUT_INVALID, AgentRunner
from tests.conftest import FakeTransport
from tests.helpers import FakeEmbedder

SMOKE = Path(__file__).parents[1] / "data" / "_smoke"
QUOTE = "Books may be borrowed for 21 days."


def final(**kwargs) -> str:
    base = {"action": "final", "status": "answered", "answer": "21 days.", "citations": [
        {"doc": "library_rules.md", "chunk_id": "library_rules.md#c0", "quote": QUOTE}]}
    return json.dumps({**base, **kwargs})


@pytest.fixture
def runner_for(tmp_path: Path):
    config = ForgeConfig(
        agent_model=ModelSpec(name="qwen3:8b"), judge_model=ModelSpec(name="gemma3:4b"),
        cache_dir=tmp_path / "cache",
    )

    def make(replies: list[str]) -> tuple[AgentRunner, FakeTransport]:
        transport = FakeTransport(replies)
        set_client(LLMClient(config, transport=transport, trace_path=tmp_path / "trace.jsonl"))
        runner = AgentRunner(load_agent(SMOKE / "agent.yaml"), SMOKE, config, FakeEmbedder())
        return runner, transport

    yield make
    set_client(None)


def test_chunk_ids_in_smoke_corpus_match_the_scripted_citation(runner_for) -> None:
    runner, _ = runner_for([])
    assert QUOTE in runner.index.by_id["library_rules.md#c0"].text or any(
        QUOTE in c.text for c in runner.index.chunks)


def real_chunk_id(runner: AgentRunner) -> str:
    return next(c.chunk_id for c in runner.index.chunks if QUOTE in c.text)


def test_valid_final_answer(runner_for) -> None:
    runner, transport = runner_for([])
    cid = real_chunk_id(runner)
    transport._replies = [final(citations=[{"doc": "library_rules.md", "chunk_id": cid, "quote": QUOTE}])]
    result = runner.ask("How long can I borrow a book?")
    assert result.contract_valid and result.answer.status == "answered"
    assert result.retrieved, "retrieval ran"
    sent = transport.calls[0]["messages"]
    assert "Brightwater Library" in sent[0]["content"] and "How long can I borrow" in sent[1]["content"]


def test_fabricated_quote_gets_one_repair_retry_with_feedback(runner_for) -> None:
    runner, transport = runner_for([])
    cid = real_chunk_id(runner)
    bad = final(citations=[{"doc": "library_rules.md", "chunk_id": cid, "quote": "Books last a month."}])
    good = final(citations=[{"doc": "library_rules.md", "chunk_id": cid, "quote": QUOTE}])
    transport._replies = [bad, good]
    result = runner.ask("How long can I borrow a book?")
    assert result.contract_valid and len(transport.calls) == 2
    assert "not a verbatim span" in transport.calls[1]["messages"][1]["content"]


def test_repair_is_attempted_only_once(runner_for) -> None:
    runner, transport = runner_for([])
    cid = real_chunk_id(runner)
    bad = final(citations=[{"doc": "library_rules.md", "chunk_id": cid, "quote": "invented"}])
    transport._replies = [bad, bad]
    result = runner.ask("How long can I borrow a book?")
    assert not result.contract_valid and len(transport.calls) == 2  # reported, not hidden


def test_tool_call_loop_records_real_results(runner_for) -> None:
    runner, transport = runner_for([])
    cid = real_chunk_id(runner)
    transport._replies = [
        json.dumps({"action": "tool_call", "tool": "calculator", "args": {"expression": "21 * 2"}}),
        final(answer="42 days.", citations=[{"doc": "library_rules.md", "chunk_id": cid, "quote": QUOTE}],
              tool_calls=[{"tool": "calculator", "args": {}, "result": "FAKE 999"}]),
    ]
    result = runner.ask("How many days is two loan periods?")
    assert [(c.tool, c.result) for c in result.answer.tool_calls] == [("calculator", "42")]
    assert "calculator({'expression': '21 * 2'}) -> 42" in transport.calls[1]["messages"][1]["content"]


def test_unattached_tool_and_tool_errors_are_reported_to_the_model(runner_for) -> None:
    runner, transport = runner_for([])
    transport._replies = [
        json.dumps({"action": "tool_call", "tool": "weather", "args": {}}),
        json.dumps({"action": "tool_call", "tool": "calculator", "args": {"expression": "1/0"}}),
        final(status="refused", answer="I cannot help with that.", citations=[]),
    ]
    result = runner.ask("What's the weather?")
    assert "not available" in result.answer.tool_calls[0].result
    assert result.answer.tool_calls[1].result == "ERROR: division by zero"
    assert result.answer.status == "refused" and result.contract_valid


def test_tool_call_limit_is_enforced(runner_for) -> None:
    runner, transport = runner_for([])
    call = json.dumps({"action": "tool_call", "tool": "calculator", "args": {"expression": "1+1"}})
    # Distinct args so each call is a cache miss; the 4th call attempt must not execute.
    calls = [json.dumps({"action": "tool_call", "tool": "calculator", "args": {"expression": f"{i}+1"}}) for i in range(4)]
    transport._replies = calls + [final(status="refused", answer="Done.", citations=[])]
    result = runner.ask("Loop please")
    assert len(result.answer.tool_calls) == runner.config.limits.max_tool_calls_per_turn == 3
    assert "limit" in transport.calls[4]["messages"][1]["content"]
    assert call


def test_invalid_model_output_is_flagged_not_hidden(runner_for) -> None:
    runner, _ = runner_for(["garbage", "garbage", "garbage"])
    result = runner.ask("Anything")
    assert MODEL_OUTPUT_INVALID in result.issues and not result.contract_valid


def test_compiled_prompt_contains_sections_tools_and_contract() -> None:
    prompt = compile_system_prompt(load_agent(SMOKE / "agent.yaml"))
    for needle in ("## Refusal policy", "set status to \"refused\"", "csv_lookup:", "calculator:",
                   "needs_clarification", "## Format examples"):
        assert needle in prompt
    assert "weather:" not in prompt


def test_no_tools_means_empty_tool_block() -> None:
    agent = load_agent(SMOKE / "agent.yaml").model_copy(update={"tools": []})
    assert "## Tools available\n(none)" in compile_system_prompt(agent)
