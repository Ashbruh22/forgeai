"""P4 exit tests (live): generate agent.v0.yaml for both demos and the library fixture, load each in
the generic chat runner and ask real questions. Real models, no mocks.

The tool-selection suite (8 mini-briefs) is tests/tool_selection/test_suite.py.
Run with:  pytest -m live tests/test_live_p4.py -s   (-s prints the answers)
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from forgeai.config import load_config
from forgeai.embeddings import SentenceTransformerEmbedder
from forgeai.generator.generate import generate_agent
from forgeai.intake.orchestrate import run_intake_and_guard
from forgeai.intake.schema import Requirements
from forgeai.llm import LLMClient, set_client
from forgeai.runner.agent_def import load_agent
from forgeai.runner.runner import AgentRunner
from tests.live import require_models

pytestmark = pytest.mark.live
ROOT = Path(__file__).parents[1]
DATA = ROOT / "data"
CASES = {
    "hr": (DATA / "hr_shore", "docs", "brief.txt"),
    "procurement": (DATA / "spares_procurement", "docs", "brief.txt"),
    "library": (DATA / "_smoke", "docs", "brief_gap.txt"),
}
FACTS = {  # a fact each in-scope / snapshot answer must contain (from the committed documents)
    "hr": {"in-scope": "24"},
    "procurement": {"in-scope": "420", "snapshot": "Singapore"},
    "library": {"in-scope": "21", "snapshot": "F-12"},
}


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory):
    config = load_config().model_copy(update={"cache_dir": ROOT / ".cache"})
    require_models(config.agent_model.name, config.judge_model.name)
    embedder = SentenceTransformerEmbedder(config.embedding_model, config.embedding_revision)
    out = {}
    for name, (base, docs, brief) in CASES.items():
        run_dir = tmp_path_factory.mktemp(name)
        set_client(LLMClient(config, trace_path=run_dir / "trace.jsonl"))
        req = run_intake_and_guard((base / brief).read_text(encoding="utf-8"), base / docs, run_dir, config, embedder)
        agent = generate_agent(run_dir, base / docs, config)
        runner = AgentRunner(load_agent(run_dir / "02_agent" / "agent.v0.yaml"), run_dir / "02_agent", config, embedder)
        out[name] = (req, agent, runner, base)
    yield out
    set_client(None)


def test_demo_a_gets_no_tools_and_demo_b_the_three_it_needs(built) -> None:
    assert built["hr"][1].tools == []
    assert sorted(built["procurement"][1].tools) == ["calculator", "csv_lookup", "current_date"]
    assert "weather" not in built["procurement"][1].tools


@pytest.mark.parametrize("name", list(CASES))
def test_refusal_policy_names_every_must_refuse_topic_and_gap_with_its_redirect(built, name) -> None:
    req: Requirements = built[name][0]
    policy = built[name][1].prompt_sections.refusal_policy
    for topic in req.must_refuse_topics:
        assert topic.topic in policy and topic.redirect in policy
    for gap in (g for g in req.gaps if g.recommended_action != "warn"):
        assert (gap.redirect in policy), gap.id


@pytest.mark.parametrize("name", list(CASES))
def test_three_smoke_questions_per_agent(built, name, capsys) -> None:
    """One in-scope, one gap, one out-of-scope question (plus a snapshot question where one exists)."""
    _, agent, runner, base = built[name]
    questions = yaml.safe_load((base / "smoke_questions.yaml").read_text(encoding="utf-8"))
    kinds = {q["kind"] for q in questions}
    assert {"in-scope", "gap", "out-of-scope"} <= kinds
    for item in questions:
        result = runner.ask(item["question"])
        answer = result.answer
        print(f"\n[{name} / {item['kind']}] {item['question']}\n   -> {answer.status}: {answer.answer}")
        if item["kind"] == "out-of-scope":
            assert answer.status == "refused", answer
        elif item["kind"] == "gap":
            # Either refuse, or give only a labelled snapshot answer. Never a bare live claim.
            assert answer.status == "refused" or "as of" in answer.answer.lower(), answer
            assert not re.match(r"^\s*(yes|no)\b", answer.answer, re.IGNORECASE), answer
        else:
            assert answer.status == "answered", answer
            assert FACTS[name][item["kind"]].lower() in (answer.answer + str(answer.tool_calls)).lower(), answer
