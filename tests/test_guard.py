"""Groundability guard: verification in code, verdicts, calibration."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from forgeai.config import ForgeConfig, ModelSpec
from forgeai.guard.guard import (
    calibrate_min_score, run_guard, span_is_relevant, verdict_for,
)
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
    router = SchemaRouter(handlers)
    set_client(LLMClient(config, transport=router, trace_path=tmp_path / "t.jsonl"))
    return router


def first_excerpt(user: str) -> tuple[str, str]:
    match = re.search(r"\[chunk_id: ([^\]]+)\]\n(.+?)(?:\n\n|\Z)", user, re.S)
    return match.group(1), match.group(2)


def honest_support(user: str) -> dict:
    """Quote the first retrieved excerpt unless the question is about the cafe (not covered)."""
    if "cafe" in user.lower():
        return {"chunk_id": "", "span": "NONE"}
    chunk_id, text = first_excerpt(user)
    return {"chunk_id": chunk_id, "span": text.split(". ")[0]}


def probes_for(user: str) -> dict:
    topic = re.search(r"Topic: (.+)", user).group(1)
    if "near-miss" in user.lower() or "NOT contain" in user:
        return {"questions": [f"Does the cafe near the library serve {topic}?", f"What is on the cafe menu about {topic}?"]}
    return {"questions": [f"What are the rules on {topic}?", f"How long is {topic}?", f"Tell me about {topic}."]}


def test_supported_and_unsupported_topics(config, tmp_path, index) -> None:
    install(config, tmp_path, {"Probes": probes_for, "Support": honest_support})
    coverage, _ = run_guard([("T1", "book loan periods"), ("T2", "cafe menu")], "library assistant", index, config)
    assert [c.verdict for c in coverage] == ["SUPPORTED", "UNSUPPORTED"]
    assert coverage[0].evidence and not coverage[1].evidence


def test_fabricated_span_is_rejected_whatever_the_model_claims(config, tmp_path, index) -> None:
    liar = lambda user: {"chunk_id": first_excerpt(user)[0], "span": "Loans last for ninety nine years."}
    install(config, tmp_path, {"Probes": probes_for, "Support": liar})
    coverage, _ = run_guard([("T1", "book loan periods")], "library", index, config)
    assert coverage[0].verdict == "UNSUPPORTED"
    assert all("verbatim" in p.reason for p in coverage[0].probes)


def test_real_but_unrelated_span_is_rejected_by_the_topic_term_check(config, tmp_path, index) -> None:
    install(config, tmp_path, {"Probes": probes_for, "Support": lambda u: {
        "chunk_id": first_excerpt(u)[0], "span": first_excerpt(u)[1].split(". ")[0]}})
    coverage, _ = run_guard([("T1", "parking permits")], "library", index, config)
    assert coverage[0].verdict == "UNSUPPORTED"
    assert all("content term" in p.reason for p in coverage[0].probes)


def test_span_relevance_helper() -> None:
    from forgeai.ingest.models import Chunk

    prose = Chunk(chunk_id="a#c0", doc="a", index=0, text="Staff receive 24 days of leave.", heading_path=["Leave"])
    assert span_is_relevant("annual leave entitlements", "Staff receive 24 days of leave.", prose)
    assert not span_is_relevant("salary slips", "Staff receive 24 days of leave.", prose)
    row = Chunk(chunk_id="t#c1", doc="t", index=1, kind="row", text="part_no: HD-1 | lead_time_days: 21")
    assert span_is_relevant("part arrival", "lead_time_days: 21", row)  # whole row counts for tables


def test_drifting_probe_cannot_launder_unrelated_text_into_support(config, tmp_path, index) -> None:
    """Regression for the first grounded-probe run: the model saw leave text, asked a leave question
    that merely mentioned the topic, and the (real) leave span was accepted as support for salary slips."""
    drifting = {"Probes": lambda u: {"questions": [
        "How many days can a book be borrowed according to the parking permits section?",
        "How long is the loan period, as parking permits are discussed?",
        "What does the parking permits chapter say about fines per day?"]},
        "Support": lambda u: {"chunk_id": first_excerpt(u)[0], "span": first_excerpt(u)[1].split(". ")[0]}}
    install(config, tmp_path, drifting)
    coverage, _ = run_guard([("T1", "parking permits")], "library", index, config)
    assert coverage[0].verdict == "UNSUPPORTED"
    assert all("content term" in p.reason for p in coverage[0].probes)


@pytest.mark.parametrize(("verified", "verdict"), [(3, "SUPPORTED"), (2, "SUPPORTED"), (1, "PARTIAL"), (0, "UNSUPPORTED")])
def test_verdict_thresholds(verified: int, verdict: str) -> None:
    assert verdict_for(verified) == verdict


def test_calibration_places_threshold_between_the_distributions() -> None:
    cal = calibrate_min_score([0.72, 0.75, 0.80, 0.78], [0.41, 0.45, 0.50, 0.47])
    assert cal.separable and cal.calibrated
    assert 0.50 < cal.threshold < 0.72
    assert cal.bounds[0] < cal.threshold < cal.bounds[1]
    assert cal.bounds[1] - cal.bounds[0] == pytest.approx(0.20)


def test_calibration_flags_overlap_as_low_confidence() -> None:
    cal = calibrate_min_score([0.55, 0.60, 0.70], [0.52, 0.58, 0.66])
    assert not cal.separable and "LOW CONFIDENCE" in cal.note


def test_calibration_without_both_distributions_keeps_the_default() -> None:
    cal = calibrate_min_score([0.7, 0.8], [])
    assert not cal.calibrated and cal.threshold == 0.25 and "uncalibrated" in cal.note


def test_calibration_bounds_are_clamped() -> None:
    cal = calibrate_min_score([0.99], [0.98])
    assert cal.bounds[1] <= 0.95 and cal.bounds[0] >= 0.05


def test_calibration_uses_guard_probes_only(config, tmp_path, index) -> None:
    """No eval data enters calibration: its inputs are exactly the guard probes' similarities."""
    install(config, tmp_path, {"Probes": probes_for, "Support": honest_support})
    coverage, cal = run_guard([("T1", "book loan periods"), ("T2", "cafe menu")], "library", index, config)
    probe_sims = {round(p.max_dense, 4) for c in coverage for p in c.probes}
    assert set(round(x, 4) for x in cal.in_scope_maxima) <= probe_sims


# ---- fixes found by the first live run on Demo B (DECISIONS.md, 2026-11-25) -----------------------

def test_overlapping_distributions_keep_recall_instead_of_the_median_midpoint() -> None:
    in_scope = [0.62, 0.70, 0.75, 0.80]
    cal = calibrate_min_score(in_scope, [0.60, 0.72, 0.81])
    assert not cal.separable and "LOW CONFIDENCE" in cal.note
    low = sorted(in_scope)[0] + (sorted(in_scope)[1] - sorted(in_scope)[0]) * 0.3  # 10th percentile
    assert cal.threshold == pytest.approx(low - 0.05)
    assert cal.threshold < min(in_scope), "no answerable probe may lose its best chunk"


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

    router = install(config, tmp_path, {"Probes": probes_for})
    probe_questions("book loan periods", "library", index, config)
    _, user = next(c for c in router.calls if c[0] == "Probes")
    assert "Books may be borrowed for 21 days" in user  # grounded, so probes can use real names/codes


def test_compact_table_row_spans_are_accepted_but_short_prose_spans_are_not(config, tmp_path) -> None:
    from forgeai.guard.guard import check_probe
    from forgeai.ingest.models import Chunk

    rows = [Chunk(chunk_id="t.csv#c1", doc="t.csv", index=1, kind="row", row=1,
                  text="part_no: HD-1123 | lead_time_days: 21 | stock_port: Singapore"),
            Chunk(chunk_id="p.md#c0", doc="p.md", index=0, text="Fines apply. Pay promptly.")]
    idx = HybridIndex.build(rows, FakeEmbedder(), "fake")
    install(config, tmp_path, {"Support": lambda u: {"chunk_id": "t.csv#c1", "span": "lead_time_days: 21"}})
    assert check_probe("What is the lead time for HD-1123?", "lead time", idx, config).supported
    install(config, tmp_path, {"Support": lambda u: {"chunk_id": "p.md#c0", "span": "Fines apply."}})
    result = check_probe("What happens with fines?", "fines", idx, config)
    assert not result.supported and "too short" in result.reason
