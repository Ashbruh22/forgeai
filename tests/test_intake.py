"""Intake interview, structural gap detection and the stage-1-to-3 orchestration (scripted LLM)."""

from __future__ import annotations

import json
import re
import shutil
from datetime import date
from pathlib import Path

import pytest
import yaml

from forgeai.config import ForgeConfig, ModelSpec
from forgeai.intake import interviewer as iv
from forgeai.intake.orchestrate import parse_gap_action, run_intake_and_guard
from forgeai.intake.schema import ListField, Question, RequirementsDraft, ScalarField
from forgeai.intake.structural import (
    detect_capability_gaps, detect_freshness_gaps, detect_live_data_gaps,
)
from forgeai.llm import LLMClient, set_client
from tests.helpers import FakeEmbedder, SchemaRouter
from tests.test_guard import agreeable_judge, honest_support, probes_for

SMOKE_DOCS = Path(__file__).parents[1] / "data" / "_smoke" / "docs"
BRIEF = "Help library visitors with lending rules, fines and room hire."
RESOLVED = RequirementsDraft(
    agent_name=ScalarField(value="Library helper", confidence="resolved_from_brief"),
    purpose=ScalarField(value="Answer lending rules for library visitors", confidence="resolved_from_brief"),
    audience=ScalarField(value="Library visitors", confidence="resolved_from_brief"),
    tone=ScalarField(value="Friendly lending answers", confidence="resolved_from_brief"),
    scope_in=ListField(values=["lending rules", "fines", "room hire"], confidence="resolved_from_brief"),
    scope_out=ListField(values=["fines waived"], confidence="resolved_from_brief"),
    must_answer=ListField(values=["lending rules", "fines"], confidence="resolved_from_brief"),
    must_refuse=ListField(values=["library staff fines waived"], confidence="resolved_from_brief"),
    capabilities=ListField(values=["document_qa", "arithmetic", "teleportation"], confidence="resolved_from_brief"),
)


@pytest.fixture
def config(tmp_path: Path) -> ForgeConfig:
    return ForgeConfig(agent_model=ModelSpec(name="qwen3:8b"), judge_model=ModelSpec(name="gemma3:4b"),
                       cache_dir=tmp_path / "cache")


def handlers(draft: RequirementsDraft) -> dict:
    return {
        "DocSummary": lambda u: {"title": "Rules", "summary": "Library lending rules and room hire."},
        "TopicLabel": lambda u: {"label": "Lending", "description": "Loans, fines and rooms."},
        "RequirementsDraft": lambda u: draft.model_dump(),
        "QuestionText": lambda u: {"text": "About " + re.search(r"undecided: (.+)", u).group(1) + "?"},
        "SelfAnswer": lambda u: {"answer": "Reasonable default answer about lending."},
        "AnswerItems": lambda u: {"values": ["item one", "item two"]},
        "Probes": probes_for,
        "Support": honest_support,
        "JudgeSupport": agreeable_judge,
        "TopicSplit": lambda u: {"aspects": []},
        "ConflictCheck": lambda u: {"reason": "different things", "quote": "", "contradicts": False},
    }


def run(config, tmp_path, draft, brief=BRIEF, ask=None, extra=None):
    docs = tmp_path / "docs"
    if not docs.exists():
        shutil.copytree(SMOKE_DOCS, docs)
    router = SchemaRouter({**handlers(draft), **(extra or {})})
    set_client(LLMClient(config, transport=router, trace_path=tmp_path / "trace.jsonl"))
    out = tmp_path / "run"
    req = run_intake_and_guard(brief, docs, out, config, FakeEmbedder(), ask=ask, today=date(2026, 11, 25))
    set_client(None)
    return req, out, router


def all_unresolved() -> RequirementsDraft:
    draft = RequirementsDraft()
    draft.purpose = ScalarField(value="Answer lending rules", confidence="resolved_from_brief")
    draft.must_answer = ListField(values=["lending rules", "fines"], confidence="resolved_from_brief")
    return draft


# ---- the <=5 question cap ---------------------------------------------------------------------

def test_question_cap_holds_with_everything_unresolved_and_a_gap(config, tmp_path) -> None:
    brief = BRIEF + " Also tell me what is on the cafe menu."
    draft = all_unresolved()
    draft.must_answer = ListField(values=["lending rules", "cafe menu for lending"], confidence="resolved_from_brief")
    req, out, _ = run(config, tmp_path, draft, brief=brief)
    questions = json.loads((out / "00_intake" / "questions.json").read_text())
    assert len(questions) == 5
    assert [q["kind"] for q in questions] == ["interview"] * 4 + ["gap"]
    assert any(g.kind == "unsupported_topic" for g in req.gaps)
    # The fifth unresolved field (tone) got no question: it must carry a labelled default.
    assert any(a.field == "tone" and a.source.startswith("ASSUMPTION") for a in req.assumptions)


def test_budget_refuses_to_exceed_the_cap() -> None:
    budget = iv.QuestionBudget(5)
    for i in range(4):
        budget.register(Question(id=f"Q{i}", field="tone", text="x"))
    with pytest.raises(iv.QuestionLimitError):
        budget.register(Question(id="Q5", field="tone", text="x"))  # 5th interview question: slot reserved
    budget.register(Question(id="Q5", field="gaps", text="x", kind="gap"))
    with pytest.raises(iv.QuestionLimitError):
        budget.register(Question(id="Q6", field="gaps", text="x", kind="gap"))


def test_zero_questions_when_nothing_is_unresolved(config, tmp_path) -> None:
    req, out, router = run(config, tmp_path, RESOLVED)
    assert json.loads((out / "00_intake" / "questions.json").read_text()) == []
    assert router.count("QuestionText") == 0 and router.count("SelfAnswer") == 0
    assert req.assumptions == []  # nothing guessed
    assert "teleportation" not in req.capabilities_needed
    assert req.capabilities_needed == ["document_qa", "arithmetic"]


# ---- non-interactive mode ---------------------------------------------------------------------

def test_every_self_answered_question_is_an_assumption(config, tmp_path) -> None:
    req, out, _ = run(config, tmp_path, all_unresolved())
    answers = json.loads((out / "00_intake" / "answers.json").read_text())
    assert answers and all(a["answered_by"] == "ASSUMPTION" for a in answers)
    for answer in answers:
        assert any(a.field == answer["field"] and a.source.startswith("ASSUMPTION (non-interactive mode)")
                   for a in req.assumptions), answer
    saved = yaml.safe_load((out / "00_intake" / "requirements.yaml").read_text())
    assert all(a["source"].startswith("ASSUMPTION") for a in saved["assumptions"])
    assert saved["scope"]["in"] and "in_" not in saved["scope"]  # YAML uses the spec's `in` key


def test_interactive_answers_come_from_the_user_not_assumptions(config, tmp_path) -> None:
    asked: list[str] = []

    def ask(question: Question) -> str:
        asked.append(question.field)
        return "Library visitors and staff"

    req, out, _ = run(config, tmp_path, all_unresolved(), ask=ask)
    answers = json.loads((out / "00_intake" / "answers.json").read_text())
    assert asked and all(a["answered_by"] == "user" for a in answers)
    assert req.audience == "Library visitors and staff"
    assert not any(a.source.startswith("ASSUMPTION (non-interactive") for a in req.assumptions)


def test_gap_question_reply_is_parsed_deterministically() -> None:
    assert parse_gap_action("refuse") == "refuse_with_redirect"
    assert parse_gap_action("We will add the payroll documents next week") == "request_documents"


# ---- confidence is verified, not trusted -------------------------------------------------------

def test_unsupported_confidence_claims_are_downgraded() -> None:
    draft = RequirementsDraft(
        audience=ScalarField(value="enterprise sysadmins", confidence="resolved_from_brief"),
        scope_out=ListField(values=["quantum computing"], confidence="resolved_from_docs"),
        tone=ScalarField(value="Friendly lending tone", confidence="resolved_from_brief"),
    )
    checked = iv.verify_confidence(draft, brief=BRIEF, docs_text="Library lending rules and fines")
    assert checked.audience.confidence == "unresolved" and checked.audience.value == ""
    assert checked.scope_out.confidence == "unresolved" and checked.scope_out.values == []
    assert checked.tone.confidence == "resolved_from_brief"  # supported by the brief


# ---- structural gaps ---------------------------------------------------------------------------

@pytest.mark.parametrize("brief", [
    "Tell them what is in stock right now at each port.",
    "It should report live data on vessel positions.",
])
def test_live_data_cues_are_flagged(brief: str) -> None:
    gaps = detect_live_data_gaps(brief)
    assert len(gaps) == 1 and gaps[0].kind == "live_data" and brief.rstrip(".") in gaps[0].evidence[0]


@pytest.mark.parametrize("brief", [
    "Answer questions about leave and travel claims.",
    "Give the weather forecast right now for the port.",   # served by the weather tool
    "Explain how many days are left until the deadline.",
])
def test_non_live_or_tool_served_briefs_are_not_flagged(brief: str) -> None:
    assert detect_live_data_gaps(brief) == []


def test_capability_cues_are_flagged() -> None:
    gaps = detect_capability_gaps("Answer questions, and email the supplier when stock is low.")
    assert len(gaps) == 1 and gaps[0].kind == "capability"
    assert detect_capability_gaps("Explain how to submit a claim and what an email policy says.") == []


def test_freshness_warns_on_old_documents_only() -> None:
    docs = {"old.pdf": "Policy. Effective date: 1 January 2019", "new.pdf": "Policy. Effective date: 1 April 2026"}
    gaps = detect_freshness_gaps(docs, today=date(2026, 11, 25))
    assert [g.evidence for g in gaps] == [["old.pdf"]] and gaps[0].recommended_action == "warn"


# ---- artifacts ---------------------------------------------------------------------------------

def test_stage_artifacts_are_written(config, tmp_path) -> None:
    brief = BRIEF + " Tell them whether a book is on the shelf right now."
    req, out, _ = run(config, tmp_path, RESOLVED, brief=brief)
    for name in ("brief.txt", "doc_inventory.json", "topic_map.json", "questions.json", "answers.json",
                 "requirements.draft.yaml", "requirements.yaml", "requirements.md", "chunks.jsonl"):
        assert (out / "00_intake" / name).is_file(), name
    for name in ("coverage.json", "gaps.md"):
        assert (out / "01_guard" / name).is_file(), name
    coverage = json.loads((out / "01_guard" / "coverage.json").read_text())
    assert "min_score" in coverage["calibration"]
    assert any(g.kind == "live_data" for g in req.gaps)
    assert "live_data" in (out / "01_guard" / "gaps.md").read_text()


def test_a_requirement_from_the_brief_is_kept_even_if_the_model_credits_the_documents() -> None:
    """Regression: the model said resolved_from_docs for a topic the BRIEF asked for (salary slips);
    checking it against the documents only dropped it, and with it the planted payroll gap."""
    draft = RequirementsDraft(must_answer=ListField(
        values=["Annual leave entitlements", "Salary slip interpretation", "Quantum computing"],
        confidence="resolved_from_docs"))
    log: list[str] = []
    checked = iv.verify_confidence(
        draft, brief="Answer leave questions. Also help people with salary slip questions.",
        docs_text="Leave policy: annual leave entitlements", log=log)
    assert checked.must_answer.values == ["Annual leave entitlements", "Salary slip interpretation"]
    assert any("Quantum computing" in note for note in log)  # unsupported items are dropped AND logged


def test_sentences_and_inverted_rules_are_not_accepted_as_topic_items() -> None:
    """Regression: Demo B got scope_out items like 'The assistant should not avoid topics related to
    compliance', which would flip a rule once compiled into the agent prompt."""
    cleaned = iv.clean_items([
        "legal advice", "The assistant should not avoid topics related to compliance",
        "These topics are integral to the spare parts process", "personal data.", "legal advice",
        "one two three four five six seven eight nine ten eleven twelve thirteen",
    ])
    assert cleaned == ["legal advice", "personal data"]


def test_live_gap_evidence_is_the_clause_not_the_whole_sentence() -> None:
    """Regression: a second request in the same sentence (the cafe) got tied to the live gap."""
    brief = ("It should also tell visitors whether a specific book is on the shelf right now, "
             "and answer questions about the cafe menu.")
    [gap] = detect_live_data_gaps(brief)
    assert gap.evidence == ["It should also tell visitors whether a specific book is on the shelf right now"]
    assert "cafe" not in gap.evidence[0]
