"""Requirement coverage matrix (P5.1 correction 1), answer-bearing facts (correction 5), dev coverage of the split."""

from __future__ import annotations

import json

from forgeai.evals.build import (
    EvalSetResult, _pick_for_coverage, assign_split, prune_facts, restates_the_question, write_eval_set,
)
from forgeai.evals.coverage import ask_capabilities, build_items, build_matrix
from forgeai.evals.generators import Candidate
from forgeai.evals.models import EvalCase, KeyFact
from forgeai.ingest.spans import SeedSpan
from forgeai.intake.schema import Gap, Requirements, Scope, Topic
from tests.helpers import FakeEmbedder
from tests.test_evals import config, install  # noqa: F401  (fixture and helper)

BRIEF = ("Our team needs help finding parts, estimating order cost with discounts, and knowing when parts will arrive. "
         "It should also tell them what is in stock right now at each port.")


def req() -> Requirements:
    return Requirements(
        agent_name="a", purpose="p", audience="x", tone="t", scope=Scope(),
        capabilities_needed=["document_qa", "structured_lookup", "arithmetic", "temporal"],
        must_answer_topics=[
            Topic(id="T1", topic="Stock levels at ports", supported_aspects=["recorded stock port"], snapshot_as_of="2026-11-01"),
            Topic(id="T3", topic="Order cost estimates with discounts")],
        gaps=[Gap(id="G-LIVE-1", kind="live_data", topic_id="T1", aspect="live stock availability",
                  recommended_action="refuse_with_redirect",
                  description='The brief asks for current information: "It should also tell them what is in stock right now at each port."',
                  evidence=["It should also tell them what is in stock right now at each port."]),
              Gap(id="G-T2", kind="partial_coverage", topic_id="T3", recommended_action="warn", description="partial")])


def case(cid: str, **kw) -> EvalCase:
    base = dict(id=cid, type="answerable", question=f"question {cid}", expected_status="answered", split="dev")
    return EvalCase(**{**base, **kw})


def test_items_cover_topics_aspects_gaps_tools_and_every_ask_with_its_capability() -> None:
    items = {i.id: i for i in build_items(req(), BRIEF, FakeEmbedder())}
    assert {"topic:T1", "topic:T3", "aspect:T1/recorded stock port", "gap:G-LIVE-1"} <= set(items)
    assert "gap:G-T2" not in items  # a warning is not refused, so it needs no refusal case
    assert items["aspect:T1/recorded stock port"].expects == "answered"
    assert items["gap:G-LIVE-1"].expects == "refused"
    assert {"negtool:weather"} <= set(items)  # the requirements do not call for live weather
    asks = {i.text for i in items.values() if i.kind == "ask"}
    assert asks == {"finding parts", "estimating order cost with discounts", "knowing when parts will arrive",
                    "tell them what is in stock right now at each port"}
    caps = sorted(i.capability for i in items.values() if i.kind == "ask_capability")
    assert caps == ["arithmetic", "structured_lookup", "temporal"]


def test_an_ask_needs_a_tool_only_if_the_requirements_list_that_capability() -> None:
    plain = req().model_copy(update={"capabilities_needed": ["document_qa"]})
    assert ask_capabilities("estimating order cost with discounts", plain) == []
    assert ask_capabilities("estimating order cost with discounts", req()) == ["arithmetic"]
    assert ask_capabilities("knowing when parts will arrive", req()) == ["temporal"]


def matrix_for(cases, items=None):
    items = items or build_items(req(), BRIEF, FakeEmbedder())
    return items, build_matrix(items, cases, {})


def row_of(matrix, item_id):
    return next(r for r in matrix["items"] if r["id"] == item_id)


def test_a_topic_answered_only_by_rate_lookups_does_not_cover_the_cost_ask() -> None:
    """The Demo B hole: cases about discount RATES covered the topic, yet no case computed a cost with a discount."""
    rate_lookup = case("a1", covers=["topic:T3"], topic_ref="T3", expected_tools=[])
    items, matrix = matrix_for([rate_lookup])
    assert row_of(matrix, "topic:T3")["status"] == "COVERED"
    cost = next(i.id for i in items if i.kind == "ask_capability" and i.capability == "arithmetic")
    assert row_of(matrix, cost)["status"] == "UNCOVERED" and cost in matrix["summary"]["uncovered"]
    computed = case("t1", type="tool_required", covers=[cost], expected_tools=["calculator", "csv_lookup"])
    _, matrix = matrix_for([rate_lookup, computed], items)
    assert row_of(matrix, cost)["status"] == "COVERED"
    wrong_tool = case("t2", type="tool_required", covers=[cost], expected_tools=["csv_lookup"])
    _, matrix = matrix_for([wrong_tool], items)
    assert row_of(matrix, cost)["status"] == "UNCOVERED"  # tagged, but it does not use the tool


def test_snapshot_aspects_need_a_case_that_expects_the_as_of_date() -> None:
    plain = case("a1", topic_ref="T1", aspect="recorded stock port", origin="snapshot", covers=["topic:T1"],
                 key_facts=[KeyFact(text="Busan", kind="phrase")])
    dated = case("a2", topic_ref="T1", aspect="recorded stock port", origin="snapshot", covers=["topic:T1"],
                 key_facts=[KeyFact(text="Busan", kind="phrase"), KeyFact(text="2026-11-01", kind="snapshot_date")])
    assert row_of(matrix_for([plain])[1], "aspect:T1/recorded stock port")["status"] == "UNCOVERED"
    assert row_of(matrix_for([plain, dated])[1], "aspect:T1/recorded stock port")["cases"] == ["a2"]


def test_refusal_items_are_covered_by_refusals_of_that_gap_or_tool_only() -> None:
    gap = case("o1", type="out_of_scope", expected_status="refused", origin="gap", topic_ref="G-LIVE-1")
    tool = case("o2", type="out_of_scope", expected_status="refused", origin="negative_tool", topic_ref="weather")
    matrix = matrix_for([gap, tool])[1]
    assert row_of(matrix, "gap:G-LIVE-1")["cases"] == ["o1"] and row_of(matrix, "negtool:weather")["cases"] == ["o2"]
    assert not any(r["id"] == "negtool:calculator" for r in matrix["items"])  # the requirements call for arithmetic


def test_an_ask_is_covered_when_everything_it_maps_to_is_covered_and_unmapped_asks_say_so() -> None:
    items = build_items(req(), BRIEF, FakeEmbedder())
    stock = next(i for i in items if i.kind == "ask" and "in stock" in i.text)
    assert {"topic:T1", "gap:G-LIVE-1"} <= set(stock.targets)  # the answerable and the refused part of the same ask
    row = row_of(matrix_for([], items)[1], stock.id)
    assert row["status"] == "UNCOVERED" and "no case for" in row["note"]
    orphan = build_items(req().model_copy(update={"must_answer_topics": [], "gaps": []}), BRIEF, FakeEmbedder())
    assert all(i.kind != "ask" or not i.targets for i in orphan)
    assert row_of(build_matrix(orphan, [], {}), next(i.id for i in orphan if i.kind == "ask"))["note"].startswith("no topic or gap")


def test_the_matrix_reports_uncovered_items_with_an_explanation_and_is_written_next_to_the_eval_set(tmp_path) -> None:
    items = build_items(req(), BRIEF, FakeEmbedder())
    matrix = build_matrix(items, [], {}, rejection_notes={"topic:T1": "3 candidates were written and all were rejected"})
    assert matrix["summary"]["uncovered"] and matrix["summary"]["covered"] == 0
    row = row_of(matrix, "topic:T1")
    assert row["status"] == "UNCOVERED" and row["explanation"].startswith("3 candidates")
    write_eval_set(EvalSetResult([], [], {"valid_cases": 0}, matrix), tmp_path)
    assert json.loads((tmp_path / "coverage_matrix.json").read_text())["summary"]["uncovered"] == matrix["summary"]["uncovered"]


def test_selection_takes_one_case_per_item_preferring_cases_that_cover_several() -> None:
    wanted = ("topic:T1", "aspect:T1/recorded stock port", "topic:T3")
    items = [i for i in build_items(req(), BRIEF, FakeEmbedder()) if i.id in wanted]
    both = case("a1", topic_ref="T1", aspect="recorded stock port", origin="snapshot", covers=["topic:T1"],
                key_facts=[KeyFact(text="2026-11-01", kind="snapshot_date")])
    only_t1 = case("a2", covers=["topic:T1"])
    t3 = case("a3", covers=["topic:T3"])
    chosen = _pick_for_coverage(items, [Candidate(only_t1), Candidate(both), Candidate(t3)], {})
    assert sorted(c.case.id for c in chosen) == ["a1", "a3"]  # a1 covers the topic AND the aspect; a2 adds nothing


def test_every_item_keeps_a_dev_case_because_the_optimizer_never_sees_test() -> None:
    items = [i for i in build_items(req(), BRIEF, FakeEmbedder()) if i.id == "topic:T3"]
    cases = [case(f"x{n}", covers=["topic:T3"] if n == 0 else []) for n in range(10)]
    only = cases[0]
    for attempt in range(3):  # whatever the hash puts the sole covering case into, the split must end with it in dev
        for c in cases:
            c.split = ""
        only.id = f"x-{attempt}"
        split = assign_split(cases, items, {})
        assert only.split == "dev" and only.id in split["dev"]
        assert len(split["dev"]) + len(split["test"]) == 10


# ---- answer-bearing key facts -----------------------------------------------------------------------------------


def test_a_fact_that_restates_the_question_is_context_not_an_answer() -> None:
    assert restates_the_question("per calendar year", "How many days do I get each calendar year?")
    assert not restates_the_question("24 working days", "How many days do I get each calendar year?")


def passage_case(question: str, facts: list[str]) -> Candidate:
    span = SeedSpan(doc="a.md", char_start=0, char_end=60, span_text="Employees receive 24 working days of annual leave per calendar year.")
    c = case("a1", question=question, key_facts=[KeyFact(text=f, kind="phrase") for f in facts], seed_spans=[span])
    return Candidate(c, [span.span_text])


def test_detail_facts_are_dropped_rather_than_relaxing_the_threshold(config, tmp_path) -> None:
    install(config, tmp_path, AnswerBearing=lambda u: {"reason": "r", "needed": "24 working days" in u.split("Key fact:")[1].split("\n")[0]})
    cand = passage_case("How much annual leave do full-time staff get?", ["24 working days", "per calendar year", "paid"])
    kept, rejected = prune_facts([cand], config)
    assert [f.text for f in kept[0].case.key_facts] == ["24 working days"] and not rejected
    dropped = kept[0].case.validation["dropped_detail_facts"]
    assert {d["fact"] for d in dropped} == {"per calendar year", "paid"}


def test_a_case_whose_every_fact_is_detail_is_rejected(config, tmp_path) -> None:
    install(config, tmp_path, AnswerBearing=lambda u: {"reason": "detail only", "needed": False})
    kept, rejected = prune_facts([passage_case("How much annual leave do staff get?", ["annual leave"])], config)
    assert not kept and rejected[0].reason == "no answer-bearing key fact"


def test_snapshot_dates_and_cases_without_passages_are_not_pruned(config, tmp_path) -> None:
    router = install(config, tmp_path, AnswerBearing=lambda u: {"reason": "", "needed": False})
    computed = Candidate(case("t1", type="tool_required", key_facts=[KeyFact(text="4788", kind="number")]))
    kept, _ = prune_facts([computed], config)
    assert kept == [computed] and router.count("AnswerBearing") == 0
