"""Live P0 check: the real wrapper against real models, no fakes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from forgeai.config import load_config
from forgeai.llm import LLMClient
from tests.live import require_models

pytestmark = pytest.mark.live


class Place(BaseModel):
    city: str
    country: str


def test_real_structured_call_on_agent_and_judge_models(tmp_path: Path) -> None:
    config = load_config().model_copy(update={"cache_dir": tmp_path / "cache"})
    require_models(config.agent_model.name, config.judge_model.name)
    trace = tmp_path / "trace.jsonl"
    client = LLMClient(config, trace_path=trace)
    variables = {"sentence": "The vessel docked in Rotterdam, in the Netherlands, on Monday."}
    for spec in (config.agent_model, config.judge_model):
        place = client.call_structured(
            "live-test", "_selftest", variables, Place, spec.name, spec.temperature, seed=spec.seed
        )
        assert place.city.lower() == "rotterdam"
        assert "netherlands" in place.country.lower()
    lines = [json.loads(x) for x in trace.read_text(encoding="utf-8").splitlines()]
    assert [x["swap"] for x in lines] == [False, True]  # the model swap is logged
    assert all(x["tokens_out"] > 0 for x in lines)
