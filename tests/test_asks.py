"""Specific asks of a brief, and the refusal check no longer matching the generic purpose sentence (P5.1 correction 4)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from forgeai.intake.asks import is_task, specific_asks
from forgeai.intake.consistency import find_conflicts
from forgeai.intake.schema import Requirements, Topic
from forgeai.llm import LLMClient, set_client
from forgeai.runner.retrieval import load_corpus_chunks
from tests.helpers import FakeEmbedder, SchemaRouter
from tests.test_consistency import req_with
from tests.test_intake import config  # noqa: F401  (fixture)

ROOT = Path(__file__).parents[1]
HR_BRIEF = ("I need an assistant that answers HR policy questions for our shore staff: leave, travel claims, conduct rules. "
            "Also help people with salary slip questions.")
PROCUREMENT_BRIEF = ("Our technical superintendents need help finding ship spares, comparing suppliers, estimating order cost with "
                     "discounts, and knowing when parts will arrive. It should also tell them what's in stock right now at each port.")


def test_the_generic_purpose_sentence_is_not_an_ask() -> None:
    asks = specific_asks(HR_BRIEF)
    assert asks == ["leave", "travel claims", "conduct rules", "salary slip questions"]
    assert not any("assistant" in a or "HR policy" in a for a in asks)


def test_asks_are_reduced_to_the_thing_asked_for() -> None:
    assert specific_asks(PROCUREMENT_BRIEF) == [
        "finding ship spares", "comparing suppliers", "estimating order cost with discounts",
        "knowing when parts will arrive", "tell them what's in stock right now at each port"]
    assert specific_asks("Help superintendents estimate order cost with discounts.") == ["estimate order cost with discounts"]


def test_a_task_is_specific_but_a_category_label_is_not() -> None:
    assert is_task("estimating order cost with discounts") and is_task("tell them what's in stock right now")
    assert not is_task("travel claims") and not is_task("conduct rules") and not is_task("Travel expense reimbursement procedures")


def judge_says_contradiction(quote: str) -> SchemaRouter:
    return SchemaRouter({"ConflictCheck": lambda u: {"reason": "r", "quote": quote, "contradicts": True}})


def conflicts(config, tmp_path, router, items, topics, brief, chunks=None):
    set_client(LLMClient(config, transport=router, trace_path=tmp_path / "t.jsonl"))
    return find_conflicts(items, req_with([], topics), brief, chunks or [], FakeEmbedder(), config)


def test_demo_a_keeps_personal_data_and_legal_advice_as_refusals(config, tmp_path) -> None:
    """The regression: a judge that finds 'personal data' part of the generic sentence, or of a broad category, must not win."""
    router = judge_says_contradiction("answers HR policy questions")  # a quote from the generic sentence
    out = conflicts(config, tmp_path, router, [("must_refuse", "personal data requests"), ("must_refuse", "legal advice")],
                    ["Annual leave entitlements", "Travel expense reimbursement procedures", "Code of conduct rules"], HR_BRIEF)
    assert out == []
    assert not any("assistant that answers" in user for _, user in router.calls)  # the generic sentence was never offered to the judge


def test_a_category_label_such_as_travel_claims_never_reaches_the_judge(config, tmp_path) -> None:
    router = judge_says_contradiction("Travel claims")
    out = conflicts(config, tmp_path, router, [("must_refuse", "personal data requests")],
                    ["Travel expense reimbursement procedures"], "Help staff with leave and travel claims.")
    assert out == [] and router.count("ConflictCheck") == 0


def test_a_described_task_still_reaches_the_judge(config, tmp_path) -> None:
    router = judge_says_contradiction("estimating order cost with discounts")
    out = conflicts(config, tmp_path, router, [("must_refuse", "outlay figures")], [], "Our team needs help estimating order cost with discounts.")
    assert [(r.item, r.conflict_with) for r in out] == [("outlay figures", "brief")]


def test_demo_b_still_rejects_pricing_details(config, tmp_path) -> None:
    chunks, _ = load_corpus_chunks(ROOT / "data" / "spares_procurement" / "docs", 350, 60)
    out = conflicts(config, tmp_path, SchemaRouter({"ConflictCheck": lambda u: {"reason": "", "quote": "", "contradicts": False}}),
                    [("must_refuse", "pricing details")], ["Order cost estimates with discounts"], PROCUREMENT_BRIEF, chunks)
    assert [(r.item, r.conflict_with) for r in out] == [("pricing details", "corpus")]


@pytest.mark.parametrize("path", sorted((ROOT / "runs").glob("p5_checkpoint/hr_shore/00_intake/requirements.yaml")) or [None],
                         ids=lambda p: "committed Demo A" if p else "no committed Demo A yet")
def test_the_committed_demo_a_requirements_keep_personal_data_and_legal_advice(path) -> None:
    """Checks the real artifact once it exists: both stay as refusals (scope out or must-refuse)."""
    if path is None:
        pytest.skip("Demo A has not been rebuilt into runs/p5_checkpoint yet")
    req = Requirements.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    refused = " | ".join([*(i.lower() for i in req.scope.out), *(t.topic.lower() for t in req.must_refuse_topics)])
    assert "personal data" in refused and "legal advice" in refused, refused
    assert Topic  # keep the import honest
