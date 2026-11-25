"""The tool-selection suite: 8 mini-briefs, real models, exact expected tool sets (P4 exit test).

Run with:  pytest -m live tests/tool_selection
Briefs 4 to 6 are keyword traps: they prove selection is not keyword matching.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from forgeai.config import load_config
from forgeai.generator.tool_selector import select_tools
from forgeai.llm import LLMClient, set_client
from tests.live import require_models

pytestmark = pytest.mark.live
BRIEFS = yaml.safe_load((Path(__file__).parent / "briefs.yaml").read_text(encoding="utf-8"))


def test_the_suite_has_eight_briefs_including_three_keyword_traps() -> None:
    assert len(BRIEFS) == 8
    assert sum("TRAP" in b["id"] for b in BRIEFS) >= 3
    assert any("weather" in b["expected"] for b in BRIEFS)


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory):
    cfg = load_config().model_copy(update={"cache_dir": Path(__file__).parents[2] / ".cache"})
    require_models(cfg.agent_model.name, cfg.judge_model.name)
    set_client(LLMClient(cfg, trace_path=tmp_path_factory.mktemp("trace") / "trace.jsonl"))
    yield cfg
    set_client(None)


@pytest.fixture(scope="module")
def results(config):
    return {b["id"]: select_tools(b["brief"], b["topics"], b["has_table"], config) for b in BRIEFS}


@pytest.mark.parametrize("case", BRIEFS, ids=[b["id"] for b in BRIEFS])
def test_selector_picks_exactly_the_needed_tools(case, results) -> None:
    selection = results[case["id"]]
    attached = sorted(d.tool for d in selection.attached)
    why = [(d.tool, d.attached, d.reason) for d in selection.decisions]
    assert attached == sorted(case["expected"]), why
    for decision in selection.attached:  # every attached tool is justified by a requirement and a quote
        assert decision.requirement_id in case["topics"] and decision.necessity_quote


def test_suite_score(results) -> None:
    passed = sum(sorted(d.tool for d in results[b["id"]].attached) == sorted(b["expected"]) for b in BRIEFS)
    print(f"tool-selection suite: {passed}/{len(BRIEFS)}")
    assert passed == len(BRIEFS)
