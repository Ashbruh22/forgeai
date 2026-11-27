"""P3 exit tests (live): real models, real embeddings, the real demo corpora. No mocks.

Run with:  pytest -m live tests/test_live_p3.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from forgeai.config import ForgeConfig, load_config
from forgeai.embeddings import SentenceTransformerEmbedder
from forgeai.intake.orchestrate import run_intake_and_guard
from forgeai.intake.schema import Requirements
from forgeai.llm import LLMClient, set_client
from tests.live import require_models

pytestmark = pytest.mark.live
ROOT = Path(__file__).parents[1]
DATA = ROOT / "data"


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory):
    config = load_config().model_copy(update={"cache_dir": ROOT / ".cache"})
    require_models(config.agent_model.name)
    embedder = SentenceTransformerEmbedder(config.embedding_model, config.embedding_revision)
    work = tmp_path_factory.mktemp("p3")

    def run(name: str, brief: Path, docs: Path) -> tuple[Requirements, Path]:
        out = work / name
        set_client(LLMClient(config, trace_path=out / "trace.jsonl"))
        req = run_intake_and_guard(brief.read_text(encoding="utf-8"), docs, out, config, embedder)
        set_client(None)
        return req, out

    return run, config


def questions(out: Path) -> list[dict]:
    return json.loads((out / "00_intake" / "questions.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def demo_a(setup):
    return setup[0]("hr_shore", DATA / "hr_shore" / "brief.txt", DATA / "hr_shore" / "docs")


@pytest.fixture(scope="module")
def demo_b(setup):
    return setup[0]("spares", DATA / "spares_procurement" / "brief.txt", DATA / "spares_procurement" / "docs")


def test_demo_a_flags_the_payroll_gap(demo_a) -> None:
    req, out = demo_a
    salary_gaps = [g for g in req.gaps if g.kind == "unsupported_topic" and "salary" in g.description.lower()]
    assert salary_gaps, [(g.kind, g.description) for g in req.gaps]
    assert not any("salary" in t.topic.lower() for t in req.must_answer_topics)
    assert "salary" in (out / "01_guard" / "gaps.md").read_text(encoding="utf-8").lower()


def test_demo_a_supported_topics_are_not_flagged(demo_a) -> None:
    req, _ = demo_a
    assert any(t.coverage == "SUPPORTED" for t in req.must_answer_topics)
    unsupported = [g.description for g in req.gaps if g.kind == "unsupported_topic"]
    assert len(unsupported) == 1, unsupported  # only the planted payroll gap


def test_demo_b_flags_the_live_stock_gap(demo_b) -> None:
    req, _ = demo_b
    live = [g for g in req.gaps if g.kind == "live_data"]
    assert live and "right now" in live[0].evidence[0]


def test_demo_b_catalogue_topics_are_not_wrongly_unsupported(demo_b) -> None:
    """Added after reading the first run's artifacts: 3 of 4 topics were false UNSUPPORTED."""
    req, _ = demo_b
    wrongly = [g.description for g in req.gaps if g.kind == "unsupported_topic"]
    assert len(wrongly) <= 1, wrongly
    assert sum(t.coverage == "SUPPORTED" for t in req.must_answer_topics) >= 2


def test_question_cap_holds_on_the_demos(demo_a, demo_b) -> None:
    for _, out in (demo_a, demo_b):
        assert len(questions(out)) <= 5


def test_non_interactive_assumptions_are_labelled(demo_a, demo_b) -> None:
    for req, out in (demo_a, demo_b):
        saved = yaml.safe_load((out / "00_intake" / "requirements.yaml").read_text(encoding="utf-8"))
        assert all(a["source"].startswith("ASSUMPTION") for a in saved["assumptions"])
        answers = json.loads((out / "00_intake" / "answers.json").read_text(encoding="utf-8"))
        for answer in answers:
            assert answer["answered_by"] == "ASSUMPTION"
            assert any(a["field"] == answer["field"] for a in saved["assumptions"]), answer


def test_guard_is_not_tuned_to_the_demos_non_demo_fixture(setup) -> None:
    """The library fixture asks for a cafe menu and live shelf status; its documents cover neither."""
    run, _ = setup
    req, out = run("library_gap", DATA / "_smoke" / "brief_gap.txt", DATA / "_smoke" / "docs")
    kinds = {g.kind for g in req.gaps}
    assert "live_data" in kinds, req.gaps
    assert any(g.kind == "unsupported_topic" and "cafe" in g.description.lower() for g in req.gaps), \
        [(g.kind, g.description) for g in req.gaps]
    assert len(questions(out)) <= 5


def test_zero_questions_when_the_brief_settles_everything(setup) -> None:
    run, _ = setup
    req, out = run("library_complete", DATA / "_smoke" / "brief_complete.txt", DATA / "_smoke" / "docs")
    asked = questions(out)
    assert asked == [], f"asked {len(asked)} question(s): {[q['text'] for q in asked]}"
    assert not any(a.source.startswith("ASSUMPTION (non-interactive") for a in req.assumptions)


# ---- P3 corrections (precedence, split, second check), added after the P3 review --------------

def test_demo_a_keeps_misconduct_reporting_in_scope(demo_a) -> None:
    """Assumption precedence: a self-answered exclusion may not remove what the corpus supports."""
    req, _ = demo_a
    excluded = [i.lower() for i in req.scope.out] + [t.topic.lower() for t in req.must_refuse_topics]
    assert not any("misconduct" in item for item in excluded), excluded
    assert any(r.item.lower() == "misconduct reporting" and r.conflict_with == "corpus"
               for r in req.assumptions_rejected) or not any("misconduct" in a.value.lower() for a in req.assumptions)


def test_demo_b_splits_stock_into_a_snapshot_part_and_a_live_gap(demo_b) -> None:
    req, _ = demo_b
    live = next(g for g in req.gaps if g.kind == "live_data")
    topic = next(t for t in req.must_answer_topics if t.id == live.topic_id)
    assert topic.supported_aspects, "the recorded (snapshot) part must stay answerable"
    assert topic.snapshot_as_of == "2026-11-01"
    assert "IS answerable" in live.description and live.redirect


def test_library_fixture_splits_shelf_location_from_live_availability(setup) -> None:
    run, _ = setup
    req, _ = run("library_split", DATA / "_smoke" / "brief_gap.txt", DATA / "_smoke" / "docs")
    live = next(g for g in req.gaps if g.kind == "live_data")
    topic = next(t for t in req.must_answer_topics if t.id == live.topic_id)
    assert topic.supported_aspects and topic.snapshot_as_of == "2026-11-20", topic
    assert "on the shelf right now" in live.evidence[0]


def test_both_support_checks_are_reported(demo_a) -> None:
    _, out = demo_a
    agreement = json.loads((out / "01_guard" / "coverage.json").read_text(encoding="utf-8"))["check_agreement"]
    assert {"both_pass", "code_check_only", "judge_only", "neither", "disagreement_rate"} <= set(agreement)
    assert agreement["probes_with_verbatim_span"] > 0


# ---- P5.1 correction 4: the refusal check must not over-reject, and must still reject real conflicts -----------

def test_demo_a_keeps_personal_data_and_legal_advice_as_refusals(demo_a) -> None:
    """Rejecting them as 'part of' the generic purpose sentence lost both in the first P5 build."""
    req, _ = demo_a
    refused = " | ".join([*(i.lower() for i in req.scope.out), *(t.topic.lower() for t in req.must_refuse_topics)])
    assert "personal data" in refused and "legal advice" in refused, (refused, [(r.item, r.detail) for r in req.assumptions_rejected])


def test_demo_b_still_rejects_pricing_details(demo_b) -> None:
    req, _ = demo_b
    assert not any("pricing" in t.topic.lower() for t in req.must_refuse_topics)
    assert any(r.item.lower() == "pricing details" and r.conflict_with == "corpus" for r in req.assumptions_rejected) or \
        not any("pricing" in a.value.lower() for a in req.assumptions)
