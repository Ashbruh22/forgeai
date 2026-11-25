import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from forgeai.llm import ChatResult, LLMError, StructuredCallError, get_client, set_client
from forgeai.prompts import PromptError, load_prompt


class Place(BaseModel):
    city: str
    country: str


class WithDefaults(BaseModel):
    city: str = "UNKNOWN"
    country: str = "UNKNOWN"


GOOD = json.dumps({"city": "Chennai", "country": "India"})
VARS = {"sentence": "The port of Chennai is in India."}


def call(client, schema=Place, model="qwen3:8b", **kwargs):
    return client.call_structured(
        "test", "_selftest", VARS, schema, model, 0.1, seed=7, **kwargs
    )


def trace_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_malformed_json_is_repaired_with_error_feedback(make_client, tmp_path) -> None:
    client, transport = make_client(['{"city": "Chen', GOOD])
    result = call(client)
    assert result == Place(city="Chennai", country="India")
    assert len(transport.calls) == 2
    # The retry carries the bad reply and the validation error back to the model.
    retry_messages = transport.calls[1]["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": '{"city": "Chen'}
    assert "could not be accepted" in retry_messages[-1]["content"]
    line = trace_lines(tmp_path / "trace.jsonl")[0]
    assert line["retries"] == 1 and line["fallback_used"] is False


def test_schema_violation_is_repaired(make_client) -> None:
    client, transport = make_client([json.dumps({"city": 5}), GOOD])
    assert call(client).country == "India"
    assert "country" in transport.calls[1]["messages"][-1]["content"]


def test_fallback_after_retries_exhausted(make_client, tmp_path) -> None:
    client, transport = make_client(["nope", "still nope", "never json"])
    result = call(client, schema=WithDefaults)
    assert result == WithDefaults()
    assert len(transport.calls) == 3  # 1 try + 2 retries
    line = trace_lines(tmp_path / "trace.jsonl")[0]
    assert line["fallback_used"] is True and line["retries"] == 2


def test_explicit_fallback_is_used(make_client) -> None:
    client, _ = make_client(["x", "y", "z"])
    fallback = Place(city="n/a", country="n/a")
    assert call(client, fallback=fallback) == fallback


def test_no_fallback_available_raises(make_client) -> None:
    client, _ = make_client(["x", "y", "z"])
    with pytest.raises(StructuredCallError):
        call(client)  # Place has required fields, so no default can be built


def test_fallback_results_are_not_cached(make_client) -> None:
    client, transport = make_client(["x", "y", "z", GOOD])
    call(client, schema=WithDefaults)
    assert call(client).city == "Chennai"  # reached the model again
    assert len(transport.calls) == 4


def test_cache_hit_skips_the_model(make_client, tmp_path) -> None:
    client, transport = make_client([GOOD])
    first = call(client)
    second = call(client)
    assert first == second
    assert len(transport.calls) == 1
    lines = trace_lines(tmp_path / "trace.jsonl")
    assert [line["cache_hit"] for line in lines] == [False, True]


def test_cache_key_depends_on_model_and_variables(make_client) -> None:
    client, transport = make_client([GOOD, GOOD, GOOD])
    call(client)
    call(client, model="gemma3:4b")
    client.call_structured("test", "_selftest", {"sentence": "other"}, Place, "qwen3:8b", 0.1, seed=7)
    assert len(transport.calls) == 3


def test_swap_and_load_time_are_traced(make_client, tmp_path) -> None:
    replies = [
        ChatResult(GOOD, load_ms=4200.0),
        ChatResult(GOOD, load_ms=0.0),
        ChatResult(GOOD, load_ms=3100.0),
    ]
    client, _ = make_client(replies)
    client.call_structured("a", "_selftest", {"sentence": "1"}, Place, "qwen3:8b", 0.1)
    client.call_structured("a", "_selftest", {"sentence": "2"}, Place, "qwen3:8b", 0.1)
    client.call_structured("j", "_selftest", {"sentence": "3"}, Place, "gemma3:4b", 0.0)
    lines = trace_lines(tmp_path / "trace.jsonl")
    assert [line["swap"] for line in lines] == [False, False, True]
    assert lines[0]["load_ms"] == 4200.0 and lines[2]["load_ms"] == 3100.0
    assert client.swap_count == 1


def test_trace_records_required_fields(make_client, tmp_path) -> None:
    client, _ = make_client([ChatResult(GOOD, tokens_in=11, tokens_out=7)])
    call(client)
    line = trace_lines(tmp_path / "trace.jsonl")[0]
    assert line["role"] == "test"
    assert line["prompt_id"] == "_selftest" and line["prompt_version"] == 1
    assert (line["tokens_in"], line["tokens_out"]) == (11, 7)
    assert "latency_ms" in line and "cache_hit" in line and "retries" in line


def test_seed_and_temperature_reach_the_model(make_client) -> None:
    client, transport = make_client([GOOD])
    call(client)
    options = transport.calls[0]["options"]
    assert options["seed"] == 7 and options["temperature"] == 0.1


def test_module_level_client_must_be_configured() -> None:
    set_client(None)
    with pytest.raises(LLMError):
        get_client()


def test_prompt_missing_variable_is_an_error() -> None:
    with pytest.raises(PromptError, match="sentence"):
        load_prompt("_selftest").render({})


def test_prompt_versions_are_parsed() -> None:
    template = load_prompt("_selftest")
    assert template.version == 1 and template.system and "{{sentence}}" in template.user


def test_no_direct_ollama_calls_outside_llm_module() -> None:
    """Rule 2: only forgeai/llm.py may talk to Ollama."""
    root = Path(__file__).parents[1] / "forgeai"
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if path.name != "llm.py"
        and any(token in path.read_text(encoding="utf-8") for token in ("/api/chat", "/api/generate", "import ollama"))
    ]
    assert offenders == []
