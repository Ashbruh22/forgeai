"""Compositional cases (P5.1 correction 3): multi-document by merging validated cases, ambiguous by code-found conflicts."""

from __future__ import annotations

import numpy as np
import pytest

from forgeai.evals.build import dedupe
from forgeai.evals.compose import (
    compose_ambiguous, compose_multidoc, conflicting_pairs, discriminators, find_attributes, merge_facts, related_pairs,
)
from forgeai.evals.corpus import EvalCorpus
from forgeai.evals.generators import Candidate, Generator
from forgeai.evals.models import EvalCase, KeyFact
from forgeai.ingest.spans import SeedSpan
from tests.helpers import FakeEmbedder
from tests.test_evals import config, corpus, docs, install, requirements  # noqa: F401  (fixtures and helpers)


def single(i: int, doc: str, question: str, fact: str, extra: str = "") -> Candidate:
    span = SeedSpan(doc=doc, char_start=0, char_end=20, span_text=f"{fact} is the rule{extra}")
    case = EvalCase(id=f"s{i}", type="answerable", question=question, expected_status="answered", expected_sources=[doc],
                    key_facts=[KeyFact(text=fact, kind="phrase")] + ([KeyFact(text=extra, kind="phrase")] if extra else []),
                    seed_spans=[span], origin="topic", covers=[f"topic:T{i}"])
    return Candidate(case, [span.span_text], group=doc)


def test_pairs_come_from_different_documents_in_a_similarity_band() -> None:
    cands = [single(1, "a.md", "how long is the warranty for new parts", "12 months"),
             single(2, "b.md", "how long is the warranty for used parts", "6 months"),
             single(3, "a.md", "how long is the warranty for new parts again", "12 months"),
             single(4, "c.md", "what is the hotel cap per night", "USD 150")]
    vectors = FakeEmbedder().encode([c.case.question for c in cands])
    pairs = related_pairs(cands, vectors, 0.3, 0.95, 5)
    assert pairs and all(not set(cands[i].case.expected_sources) & set(cands[j].case.expected_sources) for i, j, _ in pairs)
    assert related_pairs(cands, vectors, 0.99, 1.01, 5) == []  # near-duplicates (and unrelated pairs) are outside the band
    assert related_pairs(cands, np.eye(4, dtype=np.float32), 0.3, 0.95, 5) == []  # nothing related at all


def test_merged_facts_are_the_union_of_the_validated_sets_at_most_three_primary_facts_first() -> None:
    a = single(1, "a.md", "q1", "12 months", extra="from delivery").case
    b = single(2, "b.md", "q2", "18 months", extra="new parts only").case
    facts = [f.text for f in merge_facts(a, b)]
    assert facts == ["12 months", "18 months", "from delivery"] and len(facts) <= 3


def test_a_merged_multidoc_case_keeps_both_spans_and_needs_both_documents(corpus, config, tmp_path) -> None:
    install(config, tmp_path, MergedQuestion=lambda u: {"question": "Which warranty applies, and what are the payment days for used parts and new parts?"})
    gen = Generator(requirements(), corpus, config)
    cands = [single(1, "policy.md", "how long is the warranty for new parts", "12 months"),
             single(2, "terms.md", "how long is the payment period for used parts", "45 days")]
    compose_multidoc(gen, cands, 3)
    assert gen.out.candidates, gen.out.rejected
    cand = gen.out.candidates[0]
    case = cand.case
    assert case.type == "multidoc" and case.origin == "composed" and len(case.expected_sources) == 2
    assert [f.text for f in case.key_facts] == ["12 months", "45 days"] and len(cand.fact_groups) == 2
    assert len(case.seed_spans) == 2 and len(cand.passages) == 2
    assert case.provenance["composed_from"] == [cands[0].case.question, cands[1].case.question]


def test_a_merge_that_drops_one_of_the_two_questions_is_rejected(corpus, config, tmp_path) -> None:
    install(config, tmp_path, MergedQuestion=lambda u: {"question": "How long is the warranty for new parts?"})
    gen = Generator(requirements(), corpus, config)
    compose_multidoc(gen, [single(1, "policy.md", "how long is the warranty for new parts", "12 months"),
                           single(2, "terms.md", "how long is the payment period for used parts", "45 days")], 3)
    assert not gen.out.candidates and "dropped one of the two questions" in gen.out.rejected[0].reason


def test_a_merged_question_is_not_a_duplicate_of_its_own_parents() -> None:
    parent = single(1, "a.md", "how many days of annual leave do I get", "24 days").case
    merged = EvalCase(id="m", type="multidoc", question="how many days of annual leave do I get", expected_status="answered",
                      provenance={"composed_from": [parent.question, "something else"]})
    kept, _ = dedupe([Candidate(parent), Candidate(merged)], FakeEmbedder(), 0.9)
    assert len(kept) == 2


# ---- ambiguous ------------------------------------------------------------------------------------------------


@pytest.fixture
def leave_corpus(docs, config):
    text = "\n".join([
        "# Benefits", "", "## Annual Leave", "", "Employees receive 24 working days of annual leave per calendar year.", "",
        "## Sick Leave", "", "Employees receive 12 working days of sick leave per calendar year.", ""])
    (docs / "benefits.md").write_text(text, encoding="utf-8")
    return EvalCorpus.load(docs, config, FakeEmbedder())


def benefits(candidates):
    return [c.case for c in candidates if c.case.expected_sources == ["benefits.md"]]


def test_conflicting_statements_are_found_by_code_with_values_and_discriminators(leave_corpus) -> None:
    attrs = [a for a in find_attributes(leave_corpus) if a.doc == "benefits.md"]
    assert {a.quantity.text for a in attrs} == {"24 working days", "12 working days"}
    vectors = FakeEmbedder().encode([a.masked for a in attrs])
    pairs = conflicting_pairs(attrs, vectors, 0.0, 5)
    assert len(pairs) == 1
    only_a, only_b = discriminators(pairs[0][0], pairs[0][1], leave_corpus)
    assert {"annual", "sick"} <= set(only_a + only_b)  # the words that would settle the question
    assert conflicting_pairs(attrs, vectors, 1.01, 5) == []  # not similar enough to be the same statement


def test_equal_values_and_the_same_section_are_not_conflicts(leave_corpus) -> None:
    attrs = [a for a in find_attributes(leave_corpus) if a.doc == "benefits.md"]
    same = [attrs[0], attrs[0].__class__(**{**attrs[0].__dict__, "sentence": attrs[0].sentence + " ", "span": attrs[0].span})]
    assert conflicting_pairs(same, FakeEmbedder().encode([a.masked for a in same]), 0.0, 5) == []


def test_an_ambiguous_case_has_two_distinct_anchored_values_and_hides_its_discriminators(leave_corpus, config, tmp_path) -> None:
    config.thresholds.ambiguity_similarity_min = 0.0
    install(config, tmp_path, AmbiguousDraft=lambda u: {"question": "How many working days of leave do I get per calendar year?",
                                                         "reading_a": "annual leave", "reading_b": "sick leave"})
    gen = Generator(requirements(), leave_corpus, config)
    compose_ambiguous(gen, 5)
    made = benefits(gen.out.candidates)
    assert made, gen.out.rejected
    case = made[0]
    assert case.expected_status == "needs_clarification" and case.origin == "conflict"
    texts = [i.key_facts[0].text for i in case.interpretations]
    assert sorted(texts) == ["12 working days", "24 working days"] and len(set(texts)) == 2
    for interpretation in case.interpretations:  # each value is verbatim in its own span
        assert interpretation.key_facts[0].text in interpretation.seed_span.span_text
    lowered = case.question.lower()
    assert not any(w in lowered for i in case.interpretations for w in i.discriminators)


def test_a_question_that_contains_a_discriminator_or_forgets_the_attribute_is_rejected(leave_corpus, config, tmp_path) -> None:
    config.thresholds.ambiguity_similarity_min = 0.0
    install(config, tmp_path, AmbiguousDraft=lambda u: {"question": "How many working days of annual leave do I get per calendar year?",
                                                         "reading_a": "a", "reading_b": "b"})
    gen = Generator(requirements(), leave_corpus, config)
    compose_ambiguous(gen, 5)
    assert not benefits(gen.out.candidates)
    assert any("discriminator" in r.reason for r in gen.out.rejected)
    fresh = config.model_copy(update={"cache_dir": tmp_path / "fresh"})
    install(fresh, tmp_path, AmbiguousDraft=lambda u: {"question": "What is the cap?", "reading_a": "a", "reading_b": "b"})
    gen = Generator(requirements(), leave_corpus, fresh)
    compose_ambiguous(gen, 5)
    assert any("does not name the attribute" in r.reason for r in gen.out.rejected)
