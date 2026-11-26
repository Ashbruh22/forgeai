"""No topic is both asked for and refused (the P5 blocker), including for committed requirements."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from forgeai.intake.consistency import (
    ConsistencyError, assert_consistent, brief_asks, contradictions, find_conflicts, table_vocabulary,
)
from forgeai.intake.schema import RefuseTopic, Requirements, Scope, Topic
from forgeai.runner.retrieval import load_corpus_chunks
from tests.helpers import FakeEmbedder, SchemaRouter
from tests.test_intake import config  # noqa: F401  (fixture)

ROOT = Path(__file__).parents[1]
BRIEF = "Help superintendents estimate order cost with discounts. It should also answer what is on the shelf."


def req_with(refuse: list[str], answer: list[str]) -> Requirements:
    return Requirements(
        agent_name="a", purpose="p", audience="x", tone="t", scope=Scope(),
        must_answer_topics=[Topic(id=f"T{i}", topic=t) for i, t in enumerate(answer, 1)],
        must_refuse_topics=[RefuseTopic(id=f"R{i}", topic=t, reason="r", redirect="d") for i, t in enumerate(refuse, 1)])


@pytest.mark.parametrize("path", sorted((ROOT / "runs").glob("**/requirements.yaml")) or [None],
                         ids=lambda p: str(p.relative_to(ROOT)) if p else "none")
def test_invariant_no_topic_is_both_must_answer_and_must_refuse(path) -> None:
    """Runs on EVERY committed requirements.yaml."""
    if path is None:
        pytest.skip("no committed requirements.yaml yet")
    assert_consistent(Requirements.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))))


def test_the_invariant_catches_the_same_topic_and_allows_a_neighbour() -> None:
    assert contradictions(req_with(["fines"], ["overdue fines"]))
    assert not contradictions(req_with(["library staff fines waived"], ["fines"]))  # a narrower exclusion
    assert contradictions(req_with(["pricing"], ["pricing and discounts"]))
    assert not contradictions(req_with(["waiving fines"], ["overdue fines"]))  # different things
    with pytest.raises(ConsistencyError):
        assert_consistent(req_with(["fines"], ["overdue fines"]))


def test_table_columns_and_cells_are_corpus_evidence() -> None:
    chunks, _ = load_corpus_chunks(ROOT / "data" / "spares_procurement" / "docs", 350, 60)
    vocab = table_vocabulary(chunks)
    assert "pric" in vocab and "unit_price_usd" in vocab["pric"]  # "pricing" is covered by a column
    assert "lead" in vocab and "stock" in vocab


def judge(contradicts: bool, quote: str):
    return SchemaRouter({"ConflictCheck": lambda u: {"reason": "r", "quote": quote, "contradicts": contradicts}})


def run_conflicts(config, tmp_path, router, items, topics, brief=BRIEF, chunks=None):
    from forgeai.llm import LLMClient, set_client

    set_client(LLMClient(config, transport=router, trace_path=tmp_path / "t.jsonl"))
    req = req_with([], topics)
    return find_conflicts(items, req, brief, chunks or [], FakeEmbedder(), config)


def test_pricing_is_rejected_because_a_column_holds_prices(config, tmp_path) -> None:
    """The Demo B regression: 'pricing details' survived because nothing treated unit_price_usd as evidence."""
    chunks, _ = load_corpus_chunks(ROOT / "data" / "spares_procurement" / "docs", 350, 60)
    out = run_conflicts(config, tmp_path, judge(False, ""), [("must_refuse", "pricing details")],
                        ["Order cost estimates with discounts"], chunks=chunks)
    assert [(r.item, r.conflict_with) for r in out] == [("pricing details", "corpus")]
    assert "unit_price_usd" in out[0].evidence[0]


def test_the_judge_with_a_verified_quote_rejects_a_refusal_that_blocks_an_ask(config, tmp_path) -> None:
    out = run_conflicts(config, tmp_path, judge(True, "Order cost estimates"),
                        [("must_refuse", "outlay figures")], ["Order cost estimates with discounts"], brief="x")
    assert [(r.item, r.conflict_with) for r in out] == [("outlay figures", "brief")]
    assert out[0].evidence == ["Order cost estimates"]


def test_a_judge_yes_without_a_real_quote_does_not_reject(config, tmp_path) -> None:
    out = run_conflicts(config, tmp_path, judge(True, "words the brief never says"),
                        [("must_refuse", "legal advice")], ["Order cost estimates with discounts"])
    assert out == []


def test_a_judge_no_keeps_the_refusal(config, tmp_path) -> None:
    out = run_conflicts(config, tmp_path, judge(False, ""), [("must_refuse", "legal advice")],
                        ["Order cost estimates with discounts"])
    assert out == []


def test_brief_asks_cover_topics_and_clauses() -> None:
    asks = brief_asks(BRIEF, ["Order cost estimates"])
    assert "Order cost estimates" in asks and any("what is on the shelf" in a for a in asks)
