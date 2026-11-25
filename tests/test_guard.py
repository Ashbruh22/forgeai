"""Groundability guard: two independent support checks, splitting, snapshot dates, calibration."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from forgeai.config import ForgeConfig, ModelSpec
from forgeai.guard.guard import (
    TopicCoverage, agreement_stats, calibrate_min_score, check_probe_pair, run_guard, span_is_relevant,
    verdict_for,
)
from forgeai.guard.snapshot import find_snapshot_date
from forgeai.ingest.models import Chunk
from forgeai.llm import LLMClient, set_client
from forgeai.runner.retrieval import HybridIndex, load_corpus_chunks
from tests.helpers import FakeEmbedder, SchemaRouter

SMOKE_DOCS = Path(__file__).parents[1] / "data" / "_smoke" / "docs"


@pytest.fixture
def config(tmp_path: Path) -> ForgeConfig:
    return ForgeConfig(agent_model=ModelSpec(name="qwen3:8b"), judge_model=ModelSpec(name="gemma3:4b"),
                       cache_dir=tmp_path / "cache")


@pytest.fixture
def index() -> HybridIndex:
    chunks, _ = load_corpus_chunks(SMOKE_DOCS, 350, 60)
    return HybridIndex.build(chunks, FakeEmbedder(), "fake")


def install(config: ForgeConfig, tmp_path: Path, handlers: dict) -> SchemaRouter:
    router = SchemaRouter({**guard_handlers(), **handlers})
    set_client(LLMClient(config, transport=router, trace_path=tmp_path / "t.jsonl"))
    return router


def first_excerpt(user: str) -> tuple[str, str]:
    match = re.search(r"\[chunk_id: ([^\]]+)\]\n(.+?)(?:\n\n|\Z)", user, re.S)
    return match.group(1), match.group(2)


def honest_support(user: str) -> dict:
    """Quote the first retrieved excerpt unless the question is about the cafe (not covered)."""
    from forgeai.textmatch import content_terms

    if "cafe" in user.lower():
        return {"chunk_id": "", "span": "NONE"}
    question = re.search(r"Question: (.+)", user).group(1)
    excerpts = re.findall(r"\[chunk_id: ([^\]]+)\]\n(.+?)(?:\n\n|\Z)", user, re.S)
    chunk_id, text = max(excerpts, key=lambda e: len(content_terms(question) & content_terms(e[1])))
    return {"chunk_id": chunk_id, "span": text.split(". ")[0]}


def probes_for(user: str) -> dict:
    topic = re.search(r"Topic: (.+)", user).group(1)
    if "near-miss" in user.lower() or "NOT contain" in user or "not answerable" in user:
        return {"questions": [f"Does the cafe near the library serve {topic}?", f"What is on the cafe menu about {topic}?"]}
    return {"questions": [f"What are the rules on {topic}?", f"How long is {topic}?", f"Tell me about {topic}."]}


def agreeable_judge(user: str) -> dict:
    """A judge that accepts any span and quotes its first words (verbatim)."""
    span = re.search(r'^Span: "(.+)"$', user, re.M).group(1)
    return {"answers": True, "quote": " ".join(span.split()[:3])}


def guard_handlers() -> dict:
    return {"Probes": probes_for, "Support": honest_support, "JudgeSupport": agreeable_judge,
            "TopicSplit": lambda u: {"aspects": []}}


def run(config, tmp_path, topics, handlers=None, index=None):
    install(config, tmp_path, handlers or {})
    return run_guard(topics, "library assistant", index, config)


def test_supported_and_unsupported_topics(config, tmp_path, index) -> None:
    result = run(config, tmp_path, [("T1", "book loan periods"), ("T2", "cafe menu")], index=index)
    assert [c.verdict for c in result.coverage] == ["SUPPORTED", "UNSUPPORTED"]
    assert result.coverage[0].evidence and not result.coverage[1].evidence


def test_fabricated_span_is_rejected_whatever_the_model_claims(config, tmp_path, index) -> None:
    liar = lambda user: {"chunk_id": first_excerpt(user)[0], "span": "Loans last for ninety nine years."}
    result = run(config, tmp_path, [("T1", "book loan periods")], {"Support": liar}, index)
    assert result.coverage[0].verdict == "UNSUPPORTED"
    assert all("verbatim" in p.reason for p in result.coverage[0].probes)


def test_real_but_unrelated_span_is_rejected_by_the_topic_term_check(config, tmp_path, index) -> None:
    result = run(config, tmp_path, [("T1", "parking permits")], index=index)
    assert result.coverage[0].verdict == "UNSUPPORTED"
    assert all("content term" in p.reason for p in result.coverage[0].probes)


def test_span_relevance_helper() -> None:
    prose = Chunk(chunk_id="a#c0", doc="a", index=0, text="Staff receive 24 days of leave.", heading_path=["Leave"])
    assert span_is_relevant("annual leave entitlements", "Staff receive 24 days of leave.", prose)
    assert not span_is_relevant("salary slips", "Staff receive 24 days of leave.", prose)
    row = Chunk(chunk_id="t#c1", doc="t", index=1, kind="row", text="part_no: HD-1 | lead_time_days: 21")
    assert span_is_relevant("part arrival", "lead_time_days: 21", row)  # whole row counts for tables


def test_drifting_probe_cannot_launder_unrelated_text_into_support(config, tmp_path, index) -> None:
    """Regression for the first grounded-probe run: leave text accepted as support for salary slips."""
    drifting = {"Probes": lambda u: {"questions": [
        "How many days can a book be borrowed according to the parking permits section?",
        "How long is the loan period, as parking permits are discussed?",
        "What does the parking permits chapter say about fines per day?"]}}
    result = run(config, tmp_path, [("T1", "parking permits")], drifting, index)
    assert result.coverage[0].verdict == "UNSUPPORTED"
    assert all("content term" in p.reason for p in result.coverage[0].probes)


# ---- the second check: the judge model ------------------------------------------------------

def test_judge_check_can_overrule_a_span_that_passes_the_code_check(config, tmp_path, index) -> None:
    """A span on the right subject that does not answer the question: code check passes, judge says no."""
    stern = lambda u: {"answers": False, "quote": ""}
    result = run(config, tmp_path, [("T1", "book loan periods")], {"JudgeSupport": stern}, index)
    cov = result.coverage[0]
    assert cov.verdict == "UNSUPPORTED"
    assert all(p.code_ok and p.judge_ok is False for p in cov.probes)
    assert all("judge says the span does not answer" in p.reason for p in cov.probes)
    assert result.agreement["code_check_only"] == len(cov.probes) and result.agreement["disagreement_rate"] > 0


def test_judge_quote_must_be_verbatim_in_the_span(config, tmp_path, index) -> None:
    sycophant = lambda u: {"answers": True, "quote": "words that are not in the span at all"}
    result = run(config, tmp_path, [("T1", "book loan periods")], {"JudgeSupport": sycophant}, index)
    assert result.coverage[0].verdict == "UNSUPPORTED"
    assert all("not a verbatim part" in p.reason for p in result.coverage[0].probes)


def test_judge_runs_on_a_different_model_than_extraction(config, tmp_path, index) -> None:
    router = install(config, tmp_path, {})
    run_guard([("T1", "book loan periods")], "library", index, config)
    assert router.count("JudgeSupport") > 0  # the second check really ran


def test_agreement_stats_counts_each_combination() -> None:
    from forgeai.guard.guard import ProbeResult

    results = [
        ProbeResult("a", substring_ok=True, code_ok=True, judge_ok=True),
        ProbeResult("b", substring_ok=True, code_ok=True, judge_ok=False),
        ProbeResult("c", substring_ok=True, code_ok=False, judge_ok=True),
        ProbeResult("d", substring_ok=True, code_ok=False, judge_ok=False),
        ProbeResult("e"),  # no span at all: not counted as judged
    ]
    stats = agreement_stats(results)
    assert (stats["both_pass"], stats["code_check_only"], stats["judge_only"], stats["neither"]) == (1, 1, 1, 1)
    assert stats["disagreement_rate"] == 0.5 and stats["probes_total"] == 5


@pytest.mark.parametrize(("verified", "verdict"), [(3, "SUPPORTED"), (2, "SUPPORTED"), (1, "PARTIAL"), (0, "UNSUPPORTED")])
def test_verdict_thresholds(verified: int, verdict: str) -> None:
    assert verdict_for(verified) == verdict


# ---- splitting a partly supported topic -----------------------------------------------------

def test_a_topic_is_split_into_a_supported_part_and_an_unsupported_part(config, tmp_path, index) -> None:
    """Generic: nothing here knows about stock or shelves. 'Cafe' is simply what the corpus lacks."""
    split = lambda u: {"aspects": [
        {"aspect": "book loan periods", "needs_live_values": False},
        {"aspect": "cafe menu", "needs_live_values": False},
        {"aspect": "cafe menu seating right now", "needs_live_values": True}]}
    install(config, tmp_path, {"TopicSplit": split})
    result = run_guard([("T1", "book loan periods and the cafe menu")], "library", index, config,
                       live_topic_ids=frozenset({"T1"}))
    cov: TopicCoverage = result.coverage[0]
    assert cov.verdict != "SUPPORTED"
    verdicts = {a.aspect: a.verdict for a in cov.aspects}
    assert verdicts["book loan periods"] == "SUPPORTED" and verdicts["cafe menu"] == "UNSUPPORTED"
    assert verdicts["cafe menu seating right now"] is None  # live: never probed
    assert cov.effective_verdict == "PARTIAL" and [a.aspect for a in cov.supported_aspects] == ["book loan periods"]


def test_the_live_flag_is_ignored_for_topics_not_tied_to_a_live_gap(config, tmp_path, index) -> None:
    """A loose 'needs live values' label must not turn a missing document into a live-data gap."""
    split = lambda u: {"aspects": [{"aspect": "cafe menu items", "needs_live_values": True}]}
    install(config, tmp_path, {"TopicSplit": split})
    result = run_guard([("T1", "cafe menu")], "library", index, config)  # no live_topic_ids
    [aspect] = result.coverage[0].aspects
    assert aspect.needs_live_values is False and aspect.verdict == "UNSUPPORTED"


def test_a_supported_topic_tied_to_a_live_gap_is_still_split(config, tmp_path, index) -> None:
    """The static snapshot supports the recorded part of a topic, never the live part."""
    split = lambda u: {"aspects": [{"aspect": "loan periods of a book", "needs_live_values": False},
                                   {"aspect": "book loan periods right now", "needs_live_values": True}]}
    install(config, tmp_path, {"TopicSplit": split})
    result = run_guard([("T1", "book loan periods")], "library", index, config, live_topic_ids=frozenset({"T1"}))
    cov = result.coverage[0]
    assert cov.verdict == "SUPPORTED" and len(cov.aspects) == 2
    assert cov.effective_verdict == "PARTIAL"  # a live part exists, so the whole is not "supported"


def test_aspects_that_drift_off_the_parent_topic_are_dropped(config, tmp_path, index) -> None:
    split = lambda u: {"aspects": [{"aspect": "weather in Oslo", "needs_live_values": False},
                                   {"aspect": "cafe menu items", "needs_live_values": False}]}
    install(config, tmp_path, {"TopicSplit": split})
    result = run_guard([("T1", "cafe menu")], "library", index, config)
    assert [a.aspect for a in result.coverage[0].aspects] == ["cafe menu items"]


def test_a_fully_unsupported_topic_stays_unsupported_after_splitting(config, tmp_path, index) -> None:
    split = lambda u: {"aspects": [{"aspect": "cafe prices", "needs_live_values": False},
                                   {"aspect": "cafe opening hours", "needs_live_values": False}]}
    result = run(config, tmp_path, [("T1", "cafe menu")], {"TopicSplit": split}, index)
    assert result.coverage[0].effective_verdict == "UNSUPPORTED"


# ---- snapshot dates --------------------------------------------------------------------------

def test_snapshot_date_is_read_from_table_rows_and_prose() -> None:
    row = Chunk(chunk_id="b#c1", doc="b", index=1, kind="row", text="title: Moby Dick | shelf: F-12 | snapshot_date: 2026-11-20")
    prose = Chunk(chunk_id="p#c0", doc="p", index=0, text="Stock list. Data as of 3 November 2026. Contact stores.")
    assert find_snapshot_date([row]) == "2026-11-20"
    assert find_snapshot_date([prose]) == "2026-11-03"
    assert find_snapshot_date([row, prose]) == "2026-11-20"  # the latest wins
    assert find_snapshot_date([Chunk(chunk_id="x#c0", doc="x", index=0, text="No dates here.")]) is None


# ---- calibration -----------------------------------------------------------------------------

def test_calibration_places_threshold_between_the_distributions() -> None:
    cal = calibrate_min_score([0.72, 0.75, 0.80, 0.78], [0.41, 0.45, 0.50, 0.47])
    assert cal.separable and cal.calibrated
    assert 0.50 < cal.threshold < 0.72
    assert cal.bounds[0] < cal.threshold < cal.bounds[1]
    assert cal.bounds[1] - cal.bounds[0] == pytest.approx(0.20)


def test_overlapping_distributions_keep_recall_instead_of_the_median_midpoint() -> None:
    in_scope = [0.62, 0.70, 0.75, 0.80]
    cal = calibrate_min_score(in_scope, [0.60, 0.72, 0.81])
    assert not cal.separable and "LOW CONFIDENCE" in cal.note
    ordered = sorted(in_scope)
    low = ordered[0] + (ordered[1] - ordered[0]) * 0.3  # 10th percentile
    assert cal.threshold == pytest.approx(low - 0.05)
    assert cal.threshold < min(in_scope), "no answerable probe may lose its best chunk"


def test_calibration_without_both_distributions_keeps_the_default() -> None:
    cal = calibrate_min_score([0.7, 0.8], [])
    assert not cal.calibrated and cal.threshold == 0.25 and "uncalibrated" in cal.note


def test_calibration_bounds_are_clamped() -> None:
    cal = calibrate_min_score([0.99], [0.98])
    assert cal.bounds[1] <= 0.95 and cal.bounds[0] >= 0.05


def test_calibration_uses_guard_probes_only(config, tmp_path, index) -> None:
    """No eval data enters calibration: its inputs are exactly the guard probes' similarities."""
    result = run(config, tmp_path, [("T1", "book loan periods"), ("T2", "cafe menu")], index=index)
    evidence_sims = {round(p.evidence_dense, 4) for c in result.coverage for p in c.probes}
    assert result.calibration.in_scope_maxima and set(round(x, 4) for x in result.calibration.in_scope_maxima) <= evidence_sims


def test_probes_that_drift_off_topic_are_replaced_by_on_topic_ones(config, tmp_path, index) -> None:
    from forgeai.guard.guard import probe_questions

    install(config, tmp_path, {"Probes": lambda u: {"questions": [
        "What is the weather in Oslo?", "How long can a book be borrowed?", "Which animals live in Oslo?"]}})
    questions = probe_questions("book loans", "library", index, config)
    assert len(questions) == 3
    assert all("book loans" in q or "borrowed" in q for q in questions)
    assert not any("Oslo" in q for q in questions)


def test_probe_prompt_receives_real_corpus_excerpts(config, tmp_path, index) -> None:
    from forgeai.guard.guard import probe_questions

    router = install(config, tmp_path, {})
    probe_questions("book loan periods", "library", index, config)
    _, user = next(c for c in router.calls if c[0] == "Probes")
    assert "Books may be borrowed for 21 days" in user  # grounded, so probes can use real names/codes


def test_compact_table_row_spans_are_accepted_but_short_prose_spans_are_not(config, tmp_path) -> None:
    rows = [Chunk(chunk_id="t.csv#c1", doc="t.csv", index=1, kind="row", row=1,
                  text="part_no: HD-1123 | lead_time_days: 21 | stock_port: Singapore"),
            Chunk(chunk_id="p.md#c0", doc="p.md", index=0, text="Fines apply. Pay promptly.")]
    idx = HybridIndex.build(rows, FakeEmbedder(), "fake")
    install(config, tmp_path, {"Support": lambda u: {"chunk_id": "t.csv#c1", "span": "lead_time_days: 21"}})
    assert check_probe_pair("lead time", "What is the lead time for HD-1123?", idx, config).supported
    install(config, tmp_path, {"Support": lambda u: {"chunk_id": "p.md#c0", "span": "Fines apply."}})
    result = check_probe_pair("fines", "What happens with fines?", idx, config)
    assert not result.supported and "too short" in result.reason


def test_the_judge_is_shown_the_record_the_span_came_from(config, tmp_path) -> None:
    """A table fragment such as 'stock_port: Singapore' means nothing without its row (found with llama)."""
    rows = [Chunk(chunk_id="t.csv#c1", doc="t.csv", index=1, kind="row", row=1,
                  text="part_no: HD-1123 | lead_time_days: 21 | stock_port: Singapore")]
    idx = HybridIndex.build(rows, FakeEmbedder(), "fake")
    router = install(config, tmp_path, {"Support": lambda u: {"chunk_id": "t.csv#c1", "span": "stock_port: Singapore"}})
    check_probe_pair("stock port", "Which port holds HD-1123?", idx, config)
    _, user = next(c for c in router.calls if c[0] == "JudgeSupport")
    assert "part_no: HD-1123" in user  # the whole row is in the judge's prompt
