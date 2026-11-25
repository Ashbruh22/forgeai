"""Agent generator (P4): tool selection logic, prompt sections, retrieval rules, few-shot hook."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from forgeai.config import ForgeConfig, ModelSpec
from forgeai.generator.fewshot import check_few_shot_leakage, validate_few_shots
from forgeai.generator.generate import generate_agent
from forgeai.generator.retrieval_rules import retrieval_from_profile
from forgeai.generator.sections import build_sections, refusal_items
from forgeai.generator.tool_selector import ToolSelection, select_tools
from forgeai.ingest.models import DocRecord, StructureProfile
from forgeai.intake.schema import Gap, RefuseTopic, Requirements, Scope, Topic
from forgeai.llm import LLMClient, set_client
from forgeai.runner.agent_def import AgentDef, FewShot, compile_system_prompt, load_agent
from forgeai.runner.runner import AgentRunner
from tests.helpers import FakeEmbedder, SchemaRouter
from tests.test_intake import RESOLVED, config, run  # noqa: F401  (config is a fixture)

TOPICS = {"T1": "find parts in the catalogue", "T2": "work out the total cost with discounts"}


def install(config: ForgeConfig, tmp_path: Path, proposals: list[dict], need=None) -> SchemaRouter:
    need = need or (lambda u: {"stated_in_documents": False, "needed": True, "quote": "total cost"})
    router = SchemaRouter({
        "ToolProposals": lambda u: {"proposals": proposals},
        "ToolNeed": need,
        "OpeningSections": lambda u: {"identity": "You are Helper Bot, serving visitors.",
                                      "audience_tone": "Visitors with no expertise; friendly."},
    })
    set_client(LLMClient(config, transport=router, trace_path=tmp_path / "t.jsonl"))
    return router


# ---- tool selection -------------------------------------------------------------------------

def test_a_justified_and_verified_tool_is_attached(config, tmp_path) -> None:
    install(config, tmp_path, [{"tool": "calculator", "requirement_id": "T2", "justification": "totals"}])
    selection = select_tools("brief", TOPICS, has_table=True, config=config)
    [decision] = selection.attached
    assert (decision.tool, decision.requirement_id, decision.necessity_quote) == ("calculator", "T2", "total cost")
    assert "calculator" not in selection.unattached_tools and "weather" in selection.unattached_tools


@pytest.mark.parametrize(("proposal", "why"), [
    ({"tool": "teleporter", "requirement_id": "T1"}, "not a registered tool"),
    ({"tool": "calculator", "requirement_id": ""}, "not a must-answer topic"),
    ({"tool": "calculator", "requirement_id": "T9"}, "not a must-answer topic"),
])
def test_unjustified_proposals_are_dropped_by_the_validator(config, tmp_path, proposal, why) -> None:
    router = install(config, tmp_path, [proposal])
    selection = select_tools("brief", TOPICS, True, config)
    assert selection.attached == [] and why in selection.rejected[0].reason
    assert router.count("ToolNeed") == 0  # never even reached the necessity review


def test_csv_lookup_without_a_table_is_dropped_by_a_precondition(config, tmp_path) -> None:
    router = install(config, tmp_path, [{"tool": "csv_lookup", "requirement_id": "T1", "justification": "x"}])
    selection = select_tools("brief", TOPICS, has_table=False, config=config)
    assert selection.attached == [] and "no table" in selection.rejected[0].reason
    assert router.count("ToolNeed") == 0


def test_duplicate_proposals_attach_once(config, tmp_path) -> None:
    p = {"tool": "calculator", "requirement_id": "T2", "justification": "x"}
    install(config, tmp_path, [p, p])
    assert len(select_tools("brief", TOPICS, True, config).attached) == 1


@pytest.mark.parametrize(("need", "why"), [
    (lambda u: {"stated_in_documents": True, "needed": True, "quote": "total cost"}, "simply stated in a document"),
    (lambda u: {"stated_in_documents": False, "needed": False, "quote": ""}, "answered without it"),
    (lambda u: {"stated_in_documents": False, "needed": True, "quote": "words not in the topic"}, "quote is not in the topic"),
])
def test_the_necessity_review_can_reject_a_proposal(config, tmp_path, need, why) -> None:
    install(config, tmp_path, [{"tool": "calculator", "requirement_id": "T2", "justification": "x"}], need)
    selection = select_tools("brief", TOPICS, True, config)
    assert selection.attached == [] and why in selection.rejected[0].reason


def test_the_necessity_review_runs_on_the_judge_model_family(config, tmp_path) -> None:
    router = install(config, tmp_path, [{"tool": "calculator", "requirement_id": "T2", "justification": "x"}])
    select_tools("brief", TOPICS, True, config)
    assert router.count("ToolProposals") == 1 and router.count("ToolNeed") == 1
    transport_models = set()  # models used: proposals on the agent model, review on the judge model
    trace = [json.loads(line) for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    for line in trace:
        transport_models.add((line["role"], line["model"]))
    assert ("generator.tool_proposal", config.agent_model.name) in transport_models
    assert ("generator.tool_need", config.judge_model.name) in transport_models


def test_selection_serialises_attached_rejected_and_unattached(config, tmp_path) -> None:
    install(config, tmp_path, [{"tool": "calculator", "requirement_id": "T2", "justification": "x"},
                               {"tool": "weather", "requirement_id": "T1", "justification": "y"}],
            need=lambda u: {"stated_in_documents": False, "needed": "weather" not in u, "quote": "total cost" if "weather" not in u else ""})
    data = select_tools("brief", TOPICS, True, config).to_dict()
    assert [d["tool"] for d in data["attached"]] == ["calculator"]
    assert [d["tool"] for d in data["rejected"]] == ["weather"]
    assert "weather" in data["unattached_tools"] and "calculator" not in data["unattached_tools"]


# ---- prompt sections -------------------------------------------------------------------------

def requirements_with_everything() -> Requirements:
    return Requirements(
        agent_name="Helper Bot", purpose="Help visitors.", audience="Visitors with no expertise.", tone="Friendly.",
        scope=Scope(**{"in": ["lending rules"], "out": ["legal advice"]}),
        must_answer_topics=[Topic(id="T1", topic="loan periods", snapshot_as_of="2026-11-20")],
        must_refuse_topics=[RefuseTopic(id="R1", topic="personal borrowing records", reason="Private data.",
                                        redirect="Ask a librarian at the desk."),
                            RefuseTopic(id="R2", topic="fine waivers", reason="Staff decision.", redirect="Email fines@example.org.")],
        gaps=[Gap(id="G-T2", kind="unsupported_topic", topic_id="T2", recommended_action="refuse_with_redirect",
                  description="The brief asks for 'cafe menu', but none of the probes were answered.",
                  redirect="Visit the cafe counter."),
              Gap(id="G-LIVE-1", kind="live_data", recommended_action="refuse_with_redirect", topic_id="T1",
                  aspect="whether a book is on the shelf right now", description="live",
                  evidence=["on the shelf right now"], redirect="Ask at the desk."),
              Gap(id="G-CAP-1", kind="capability", recommended_action="refuse_with_redirect",
                  description="action", evidence=["email the supplier"], redirect="Email them yourself."),
              Gap(id="G-FRESH-1", kind="freshness", recommended_action="warn", description="old", evidence=["old.pdf"])],
    )


def test_refusal_policy_contains_every_must_refuse_topic_and_every_guard_gap_with_its_redirect(config, tmp_path) -> None:
    req = requirements_with_everything()
    install(config, tmp_path, [])
    refusal = build_sections(req, ToolSelection(), config).refusal_policy
    for topic in req.must_refuse_topics:
        assert topic.topic in refusal and topic.redirect in refusal
    for gap in req.gaps:
        if gap.recommended_action == "warn":
            assert gap.evidence[0] not in refusal  # warnings are not refusals
            continue
        assert gap.redirect in refusal, gap.id
    assert "cafe menu" in refusal  # the unsupported topic, named as the user would ask about it
    assert "whether a book is on the shelf right now" in refusal  # the live aspect, not the whole topic
    assert "email the supplier" in refusal and "Visit the cafe counter." in refusal
    assert len(refusal_items(req)) == 5  # 2 must-refuse topics + 3 refusing gaps


def test_snapshot_dates_become_an_as_of_rule(config, tmp_path) -> None:
    install(config, tmp_path, [])
    grounding = build_sections(requirements_with_everything(), ToolSelection(), config).grounding_rules
    assert "as of 2026-11-20" in grounding and "Never present it as current or live" in grounding


def test_all_nine_sections_exist_and_unattached_tools_are_refused(config, tmp_path) -> None:
    install(config, tmp_path, [])
    sections = build_sections(requirements_with_everything(), ToolSelection(), config)
    for name in ("identity", "audience_tone", "scope", "refusal_policy", "grounding_rules", "citation_rules",
                 "ambiguity_policy", "tool_policy", "output_contract"):
        assert getattr(sections, name).strip(), name
    assert "no tools" in sections.tool_policy and "weather" in sections.tool_policy  # negative-tool rule
    assert sections.few_shot == []


def test_model_written_sections_are_checked_and_fall_back_to_the_template(config, tmp_path) -> None:
    router = SchemaRouter({"OpeningSections": lambda u: {"identity": "You are a generic bot.", "audience_tone": "Be nice."}})
    set_client(LLMClient(config, transport=router, trace_path=tmp_path / "t.jsonl"))
    sections = build_sections(requirements_with_everything(), ToolSelection(), config)
    assert "Helper Bot" in sections.identity  # the model dropped the name: template used
    assert "Visitors with no expertise" in sections.audience_tone  # the model ignored the audience: template used


# ---- retrieval rules ----------------------------------------------------------------------------

def record(kind: str, words: int, avg_section: float) -> DocRecord:
    return DocRecord(file=f"{kind}.x", status="ok", structure=StructureProfile(
        kind=kind, total_words=words, n_blocks=10, n_headings=3, max_heading_depth=2, avg_section_words=avg_section))


@pytest.mark.parametrize(("avg", "size", "overlap"), [(40.0, 200, 40), (150.0, 350, 60), (400.0, 500, 80)])
def test_chunk_size_follows_section_length(avg, size, overlap) -> None:
    retrieval, why = retrieval_from_profile([record("prose", 3000, avg)])
    assert (retrieval.chunking.size_tokens, retrieval.chunking.overlap_tokens) == (size, overlap)
    assert "avg" in why["chunking"] and retrieval.top_k == 5


def test_tables_get_row_records_balanced_hybrid_and_more_hits() -> None:
    retrieval, why = retrieval_from_profile([record("tabular", 5000, 0.0), record("prose", 1000, 150.0)])
    assert retrieval.csv_mode == "row_as_record" and retrieval.top_k == 6
    assert (retrieval.hybrid.dense_weight, retrieval.hybrid.bm25_weight) == (0.5, 0.5)
    assert "exact matching" in why["hybrid"]


def test_min_score_comes_from_the_calibration_with_its_bounds() -> None:
    calibration = {"calibrated": True, "min_score": 0.6571, "min_score_bounds": [0.5571, 0.7571], "note": "n"}
    retrieval, why = retrieval_from_profile([record("prose", 1000, 150.0)], calibration)
    assert retrieval.min_score == 0.6571 and retrieval.min_score_bounds == (0.5571, 0.7571)
    assert "calibrated" in why["min_score"]
    default, _ = retrieval_from_profile([record("prose", 1000, 150.0)], {"calibrated": False})
    assert default.min_score == 0.25 and default.min_score_bounds == (0.05, 0.5)


# ---- few-shot leakage hook -----------------------------------------------------------------------

def test_few_shots_too_similar_to_an_eval_question_are_rejected() -> None:
    shots = [FewShot(question="How many days of annual leave do I get", reply={"status": "answered"}),
             FewShot(question="Where is the cafe located in the building", reply={"status": "refused"})]
    report = check_few_shot_leakage(shots, ["How many days of annual leave do I get?"], FakeEmbedder(), 0.80)
    assert [s.question for s in report.kept] == ["Where is the cafe located in the building"]
    assert report.rejected[0][1].startswith("How many days") and report.rejected[0][2] > 0.80


def test_the_leakage_hook_is_pending_until_an_eval_set_exists(config) -> None:
    agent = AgentDef(name="a", model={"name": "m"}, corpus={"path": "."},
                     prompt_sections={"identity": "i", "few_shot": [{"question": "q", "reply": {}}]})
    same, report = validate_few_shots(agent, None, FakeEmbedder(), 0.8)
    assert report is None and same.prompt_sections.few_shot  # nothing can be checked yet
    cleaned, report = validate_few_shots(agent, ["q"], FakeEmbedder(), 0.8)
    assert report is not None and cleaned.prompt_sections.few_shot == []


# ---- generate_agent end to end (scripted LLM) ------------------------------------------------------

def test_generate_agent_writes_a_valid_definition_the_generic_runner_can_load(config, tmp_path) -> None:
    from tests.test_precedence_and_split import BRIEF

    req, out, _ = run(config, tmp_path, RESOLVED, brief=BRIEF)
    install(config, tmp_path, [{"tool": "calculator", "requirement_id": "T1", "justification": "fine totals"}],
            need=lambda u: {"stated_in_documents": False, "needed": True, "quote": "lending rules"})
    agent = generate_agent(out, tmp_path / "docs", config)

    path = out / "02_agent" / "agent.v0.yaml"
    assert path.is_file() and (out / "02_agent" / "tool_selection.json").is_file()
    loaded = load_agent(path)
    assert loaded.tools == ["calculator"] and loaded.tool_justifications["calculator"].startswith("T1:")
    assert loaded.corpus.path == "../../docs"  # relative to the agent file, so the run folder is portable
    assert list(yaml.safe_load(path.read_text())["prompt_sections"]) == [
        "identity", "audience_tone", "scope", "refusal_policy", "grounding_rules", "citation_rules",
        "ambiguity_policy", "tool_policy", "output_contract", "few_shot"]
    assert loaded.provenance["fewshot_leakage_check"].startswith("pending")
    assert "retrieval_rationale" in loaded.provenance and agent.retrieval.min_score_bounds[0] < agent.retrieval.min_score

    prompt = compile_system_prompt(loaded)  # compiled in the fixed order
    assert prompt.index("## Scope") < prompt.index("## Refusal policy") < prompt.index("## Grounding rules")
    runner = AgentRunner(loaded, path.parent, config, FakeEmbedder())  # loads in the generic runner
    assert runner.index.chunks and "calculator" in runner.tools


def test_a_live_gap_on_a_topic_with_a_recorded_part_allows_the_labelled_snapshot_answer(config, tmp_path) -> None:
    """Regression from the library smoke test: asked 'is it on the shelf right now?' the agent said
    'Yes, on the shelf as of <date>', claiming the live state from a snapshot."""
    req = Requirements(
        agent_name="Helper Bot", purpose="p", audience="a", tone="t", scope=Scope(),
        must_answer_topics=[Topic(id="T1", topic="a book on the shelf", coverage="PARTIAL",
                                  supported_aspects=["which shelf the catalogue lists the book on"],
                                  snapshot_as_of="2026-11-20")],
        gaps=[Gap(id="G-LIVE-1", kind="live_data", topic_id="T1", recommended_action="refuse_with_redirect",
                  aspect="whether the book is on the shelf right now", description="live",
                  redirect="Ask at the desk.")])
    [item] = refusal_items(req)
    assert "never answer yes or no about the current state" in item
    assert "which shelf the catalogue lists the book on" in item and 'as of 2026-11-20' in item
    assert "Ask at the desk." in item and "whether the book is on the shelf right now" in item
    install(config, tmp_path, [])
    policy = build_sections(req, ToolSelection(), config).refusal_policy
    assert "- {{" not in policy and "- - " not in policy  # one bullet per item, never doubled
