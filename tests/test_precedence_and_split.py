"""Assumption precedence (brief > corpus > assumptions) and splitting partial topics."""

from __future__ import annotations

import pytest

from forgeai.intake.schema import ListField, RequirementsDraft, ScalarField
from tests.test_intake import RESOLVED, all_unresolved, config, run  # noqa: F401  (config is a fixture)

BRIEF = "Help library visitors with lending rules and fines."


def self_answer_items(items: list[str]) -> dict:
    return {"AnswerItems": lambda u: {"values": items}}


# ---- fix 1: assumption precedence -------------------------------------------------------------

def test_assumed_exclusion_of_a_corpus_supported_topic_is_rejected_and_logged(config, tmp_path) -> None:
    """Non-interactive self-answer excludes 'room hire' and 'cafe menu'. The documents answer room-hire
    questions, so that exclusion must be dropped; the cafe is not covered, so that one may stay."""
    req, out, _ = run(config, tmp_path, all_unresolved(), brief=BRIEF,
                      extra=self_answer_items(["room hire", "cafe menu"]))
    assert "room hire" not in req.scope.out and "cafe menu" in req.scope.out
    rejected = {r.item: r for r in req.assumptions_rejected}
    assert set(rejected) >= {"room hire"} and "cafe menu" not in rejected
    assert rejected["room hire"].conflict_with == "corpus" and rejected["room hire"].evidence
    assert "rooms.csv" in " ".join(rejected["room hire"].evidence) or "library_rules" in " ".join(rejected["room hire"].evidence)
    import yaml
    saved = yaml.safe_load((out / "00_intake" / "requirements.yaml").read_text())
    assert any(r["item"] == "room hire" for r in saved["assumptions_rejected"])


def test_assumed_exclusion_overlapping_the_briefs_own_topics_is_rejected(config, tmp_path) -> None:
    req, _, _ = run(config, tmp_path, all_unresolved(), brief=BRIEF,
                    extra=self_answer_items(["fines for late returns", "cafe menu"]))
    assert req.scope.out == ["cafe menu"]
    clashes = [r for r in req.assumptions_rejected if r.item == "fines for late returns"]
    assert {r.field for r in clashes} == {"scope_out", "must_refuse"}  # the same answer fed both lists
    assert all(r.conflict_with == "brief" for r in clashes)


def test_assumed_refusal_of_a_supported_topic_is_rejected_too(config, tmp_path) -> None:
    req, _, _ = run(config, tmp_path, all_unresolved(), brief=BRIEF,
                    extra=self_answer_items(["room hire", "the cafe menu"]))
    assert "room hire" not in [t.topic for t in req.must_refuse_topics]
    assert [t.topic for t in req.must_refuse_topics] == ["the cafe menu"]
    assert [t.id for t in req.must_refuse_topics] == ["R1"]  # ids stay contiguous


def test_an_exclusion_the_brief_itself_states_is_never_rejected(config, tmp_path) -> None:
    """Brief > corpus: the requester said it, so it stays even though the corpus could answer it."""
    draft = all_unresolved()
    draft.scope_out = ListField(values=["room hire"], confidence="resolved_from_brief")
    req, _, _ = run(config, tmp_path, draft, brief=BRIEF + " Do not discuss room hire.")
    assert "room hire" in req.scope.out and req.assumptions_rejected == []


def test_user_answers_are_not_treated_as_assumptions(config, tmp_path) -> None:
    from forgeai.intake.schema import Question

    answers = iter(["room hire", "x", "x", "x"])
    req, _, _ = run(config, tmp_path, all_unresolved(), brief=BRIEF, ask=lambda q: next(answers),
                    extra=self_answer_items(["room hire"]))
    assert req.assumptions_rejected == []  # the human said it; precedence only polices self-answers


# ---- fix 2: split a partial topic ---------------------------------------------------------------

LIVE_BRIEF = "Help visitors find books. Tell visitors whether a book is on the shelf right now."


def split_handlers() -> dict:
    topic_probes_fail = lambda u: ({"chunk_id": "", "span": "NONE"}
                                   if any(x in u for x in ("rules on", "How long is", "Tell me about"))
                                   else None)

    def support(user: str) -> dict:
        from tests.test_guard import honest_support

        return topic_probes_fail(user) or honest_support(user)

    return {
        "Support": support,
        "TopicSplit": lambda u: {"aspects": [
            {"aspect": "which shelf a book is on", "needs_live_values": False},
            {"aspect": "whether a book is on the shelf right now", "needs_live_values": True}]},
        "Probes": lambda u: ({"questions": ["On which shelf is Moby Dick?", "Which shelf holds Walden?",
                                            "What shelf is Treasure Island on?"]}
                             if "Topic: which shelf a book is on" in u
                             else __import__("tests.test_guard", fromlist=["probes_for"]).probes_for(u)),
    }


def test_partial_topic_keeps_its_supported_part_and_refuses_only_the_live_part(config, tmp_path) -> None:
    """The library fixture: 'which shelf is this book on' is answerable (static catalogue), 'is it on the
    shelf right now' is a live-data gap. Nothing here is specific to this domain."""
    draft = all_unresolved()
    draft.must_answer = ListField(values=["books on the shelf"], confidence="resolved_from_brief")
    req, out, _ = run(config, tmp_path, draft, brief=LIVE_BRIEF, extra=split_handlers())

    [topic] = req.must_answer_topics
    assert topic.coverage == "PARTIAL"
    assert topic.supported_aspects == ["which shelf a book is on"]
    assert topic.snapshot_as_of == "2026-11-20"  # read from the evidence rows, not invented

    kinds = [(g.kind, g.aspect) for g in req.gaps]
    assert [k for k, _ in kinds] == ["live_data"], kinds  # one live gap, no unsupported_topic gap
    [live] = req.gaps
    assert live.topic_id == topic.id and live.aspect == "whether a book is on the shelf right now"
    assert "IS answerable" in live.description and "2026-11-20" in live.description
    assert live.recommended_action == "refuse_with_redirect" and live.redirect
    assert "supported part" in (out / "00_intake" / "requirements.md").read_text()


def test_a_split_with_nothing_supported_still_becomes_a_whole_gap(config, tmp_path) -> None:
    draft = all_unresolved()
    draft.must_answer = ListField(values=["cafe menu"], confidence="resolved_from_brief")
    handlers = {"TopicSplit": lambda u: {"aspects": [{"aspect": "cafe prices", "needs_live_values": False},
                                                     {"aspect": "cafe opening hours", "needs_live_values": False}]}}
    req, _, _ = run(config, tmp_path, draft, brief="Tell visitors about the cafe menu.", extra=handlers)
    assert req.must_answer_topics == []
    assert [g.kind for g in req.gaps] == ["unsupported_topic"]


def test_unsupported_nonlive_aspect_of_a_partial_topic_becomes_a_gap(config, tmp_path) -> None:
    draft = all_unresolved()
    draft.must_answer = ListField(values=["book loan periods and the cafe menu"], confidence="resolved_from_brief")
    handlers = {"TopicSplit": lambda u: {"aspects": [{"aspect": "book loan periods", "needs_live_values": False},
                                                     {"aspect": "cafe menu", "needs_live_values": False}]}}
    req, _, _ = run(config, tmp_path, draft, brief="Tell visitors about book loan periods and the cafe menu.", extra=handlers)
    [topic] = req.must_answer_topics
    assert topic.coverage == "PARTIAL" and topic.supported_aspects == ["book loan periods"]
    [gap] = [g for g in req.gaps if g.kind == "unsupported_topic"]
    assert gap.aspect == "cafe menu" and gap.recommended_action == "refuse_with_redirect"


def live_gap(clause: str):
    from forgeai.intake.schema import Gap

    return Gap(id="G-LIVE-1", kind="live_data", description="d", recommended_action="refuse_with_redirect", evidence=[clause])


def test_a_live_gap_is_tied_to_the_topic_that_contains_most_of_its_words() -> None:
    from forgeai.intake.orchestrate import tie_live_gaps
    from forgeai.intake.schema import Topic

    topics = [Topic(id="T1", topic="fines for overdue books"), Topic(id="T2", topic="is a book on the shelf"),
              Topic(id="T3", topic="library card membership")]
    gap = live_gap("Tell visitors whether a specific book is on the shelf right now")
    assert tie_live_gaps(topics, [gap]) == frozenset({"T2"}) and gap.topic_id == "T2" and len(topics) == 3


def test_a_live_gap_no_topic_covers_becomes_its_own_must_answer_topic() -> None:
    """Regression: with only a shared word ('book'), the fines topic was blamed for the shelf request."""
    from forgeai.intake.orchestrate import tie_live_gaps
    from forgeai.intake.schema import Topic

    topics = [Topic(id="T1", topic="fines for overdue books"), Topic(id="T2", topic="Is the book 'Pride and Prejudice' available?")]
    gap = live_gap("It should also tell visitors whether a specific book is on the shelf right now")
    assert tie_live_gaps(topics, [gap]) == frozenset({"T3"})
    assert topics[-1].id == "T3" and topics[-1].topic == "tell visitors whether a specific book is on the shelf right now"
    assert gap.topic_id == "T3"


def test_a_live_gap_with_no_topics_at_all_still_gets_one() -> None:
    from forgeai.intake.orchestrate import tie_live_gaps

    topics: list = []
    gap = live_gap("Report live vessel positions")
    assert tie_live_gaps(topics, [gap]) == frozenset({"T1"}) and topics[0].topic == "Report live vessel positions"
