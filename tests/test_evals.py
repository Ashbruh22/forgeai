"""Eval generator (P5): matching, computed cases, generators, selection, split, consistency, isolation."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from forgeai.config import ForgeConfig, ModelSpec
from forgeai.evals.build import (
    TARGET_TOTAL, assign_split, build_eval_set, dedupe, select, targets_for, write_eval_set,
)
from forgeai.evals.consistency import EvalConsistencyError, assert_cases_consistent, violation
from forgeai.evals.corpus import EvalCorpus
from forgeai.evals.generators import Candidate, Generator
from forgeai.evals.matching import dates_in, fact_in_text, ngram_overlap, numbers_in, phrase_in_text
from forgeai.evals.models import EvalCase, KeyFact
from forgeai.intake.schema import Gap, RefuseTopic, Requirements, Scope, Topic
from forgeai.llm import LLMClient, set_client
from tests.helpers import FakeEmbedder, SchemaRouter

PACKAGE = Path(__file__).parents[1] / "forgeai"


@pytest.fixture
def config(tmp_path: Path) -> ForgeConfig:
    return ForgeConfig(agent_model=ModelSpec(name="qwen3:8b"), judge_model=ModelSpec(name="llama3.1:8b"),
                       cache_dir=tmp_path / "cache")


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "policy.md").write_text(
        "# Leave Policy\n\n## Annual Leave\n\nFull-time employees receive 24 working days of annual leave per calendar year, "
        "accruing monthly at a steady rate.\n\n## Sick Leave\n\nEmployees receive 12 working days of paid sick leave per calendar "
        "year and must provide a medical certificate after 2 consecutive working days.\n", encoding="utf-8")
    (folder / "terms.md").write_text(
        "# Supplier Terms\n\n## Volume Discounts\n\nSuppliers S01 and S02 terms. Orders from USD 5,000 to USD 14,999 receive a 5% discount on the list price, "
        "and orders from USD 15,000 to USD 49,999 receive an 8% discount on the list price.\n", encoding="utf-8")
    (folder / "parts.csv").write_text(
        "part_no,supplier,description,unit_price_usd,snapshot_date\n"
        "HD-1,S01,Piston ring,420.00,2026-11-20\n"
        "HD-2,S01,Cylinder liner,950.50,2026-11-20\n"
        "HD-1,S02,Piston ring,433.00,2026-11-20\n", encoding="utf-8")
    return folder


@pytest.fixture
def corpus(docs: Path, config: ForgeConfig) -> EvalCorpus:
    return EvalCorpus.load(docs, config, FakeEmbedder())


def requirements(**kwargs) -> Requirements:
    base = dict(
        agent_name="Helper", purpose="Answer leave and ordering questions.", audience="Staff", tone="Plain",
        scope=Scope(), capabilities_needed=["document_qa", "structured_lookup", "arithmetic"],
        must_answer_topics=[Topic(id="T1", topic="annual leave entitlements"), Topic(id="T2", topic="order discounts")],
        must_refuse_topics=[RefuseTopic(id="R1", topic="medical advice", reason="Not covered.", redirect="See a doctor.")],
        gaps=[Gap(id="G-T3", kind="unsupported_topic", topic_id="T3", recommended_action="refuse_with_redirect",
                  description="The brief asks for 'parking permits', but nothing covers it.", redirect="Ask facilities.")])
    return Requirements(**{**base, **kwargs})


# ---- matching ------------------------------------------------------------------------------------

def test_numbers_dates_and_phrases_are_normalised() -> None:
    assert numbers_in("USD 5,000 and 24 working days, 1.75 per month, twenty four hours") == [5000.0, 24.0, 1.75, 24.0]
    assert [d.isoformat() for d in dates_in("As of 3 November 2026 and 2026-11-01")] == ["2026-11-01", "2026-11-03"]
    assert fact_in_text("number", "24 working days", "entitled to 24 days") is True
    assert fact_in_text("number", "24 working days", "entitled to 23 days") is False
    assert fact_in_text("snapshot_date", "2026-11-20", "snapshot_date: 2026-11-20")
    assert fact_in_text("date", "2026-11-20", "as of 20 November 2026")


def test_strict_phrase_matching_rejects_a_dropped_negation() -> None:
    text = "public holidays inside annual leave are counted as leave days"
    assert phrase_in_text("not counted as leave days", text, strict=False)  # token-set similarity is lenient
    assert not phrase_in_text("not counted as leave days", text, strict=True)  # anchoring needs every word
    assert phrase_in_text("counted as leave days", text, strict=True)


def test_paraphrase_overlap() -> None:
    source = "Employees receive twelve days of paid sick leave each year"
    assert ngram_overlap("Employees receive twelve days of paid sick leave each year?", source) > 0.9
    assert ngram_overlap("How long can I stay off when ill?", source) < 0.2


# ---- generators with a scripted model --------------------------------------------------------------

def passage_of(user: str, label: str = "Passage") -> str:
    return re.search(rf'{label}.*?:\n"""\n(.+?)\n"""', user, re.S).group(1)


def first_sentence(text: str) -> str:
    return re.split(r"(?<=\.)\s", text)[0]


def good_answerable(user: str) -> dict:
    passage = passage_of(user)
    span = first_sentence(passage)
    number = re.search(r"\d[\d,]*", span)
    return {"question": "How much is the entitlement mentioned in this section?", "answer_span": span,
            "key_facts": [number.group(0) if number else span.split()[0]]}


def install(config, tmp_path, **handlers) -> SchemaRouter:
    router = SchemaRouter(handlers)
    set_client(LLMClient(config, transport=router, trace_path=tmp_path / "t.jsonl"))
    return router


def generator(req, corpus, config) -> Generator:
    return Generator(req, corpus, config)


def test_a_good_answerable_candidate_is_anchored_with_character_offsets(corpus, config, tmp_path) -> None:
    install(config, tmp_path, AnswerableDraft=good_answerable)
    gen = generator(requirements(), corpus, config)
    gen.answerable(6)
    assert gen.out.candidates, gen.out.rejected
    case = gen.out.candidates[0].case
    span = case.seed_spans[0]
    assert corpus.docs[span.doc].canonical_text[span.char_start:span.char_end] == span.span_text  # offsets are real
    assert case.expected_status == "answered" and case.validation["fact_anchoring"] is True


@pytest.mark.parametrize(("handler", "reason"), [
    (lambda u: {**good_answerable(u), "answer_span": "A sentence that is nowhere in the passage."}, "not a verbatim quote"),
    (lambda u: {**good_answerable(u), "key_facts": ["99999"]}, "fact anchoring failed"),
    (lambda u: {**good_answerable(u), "question": passage_of(u)}, "copies the passage"),
    (lambda u: {"question": "", "answer_span": "", "key_facts": []}, "model output invalid"),
])
def test_bad_answerable_drafts_are_rejected_with_a_reason(corpus, config, tmp_path, handler, reason) -> None:
    install(config, tmp_path, AnswerableDraft=handler)
    gen = generator(requirements(), corpus, config)
    gen.answerable(4)
    assert not gen.out.candidates and any(reason in r.reason for r in gen.out.rejected)


def test_snapshot_cases_expect_the_snapshot_date_as_a_key_fact(corpus, config, tmp_path) -> None:
    row = next(c for c in corpus.chunks if c.kind == "row" and "HD-1 | supplier: S01" in c.text)
    req = requirements(must_answer_topics=[Topic(id="T1", topic="parts", evidence=[row.chunk_id],
                                                 supported_aspects=["recorded price"], snapshot_as_of="2026-11-20")])
    install(config, tmp_path, AnswerableDraft=lambda u: {
        "question": "What does the catalogue list as the price of HD-1 from S01?", "answer_span": "unit_price_usd: 420.00",
        "key_facts": ["420.00"]})
    gen = generator(req, corpus, config)
    gen.snapshot_cases()
    assert gen.out.candidates, gen.out.rejected
    case = gen.out.candidates[0].case
    assert KeyFact(text="2026-11-20", kind="snapshot_date") in case.key_facts and "snapshot" in case.tags
    assert "2026-11-20" in case.seed_spans[0].span_text  # the date is anchored in the seed row itself


def test_refusal_candidates_come_from_must_refuse_gaps_and_unrequired_tools(corpus, config, tmp_path) -> None:
    install(config, tmp_path, QuestionsDraft=lambda u: {"questions": ["A realistic question one?", "A realistic question two?"]},
            Probes=lambda u: {"questions": []})
    req = requirements(capabilities_needed=["document_qa"])
    gen = generator(req, corpus, config)
    gen.refusals(14)
    origins = {c.case.origin for c in gen.out.candidates}
    assert {"must_refuse", "gap", "negative_tool"} <= origins
    negatives = {c.case.topic_ref for c in gen.out.candidates if c.case.origin == "negative_tool"}
    assert negatives and negatives <= {"csv_lookup", "calculator", "current_date", "weather"}
    assert all(c.case.expected_status == "refused" for c in gen.out.candidates)
    gap_cases = [c for c in gen.out.candidates if c.case.origin == "gap"]
    assert all(c.probe is not None for c in gap_cases)  # an unsupported-topic refusal must fail the guard probe


def test_lookup_cases_take_the_expected_value_from_the_table_cell(corpus, config, tmp_path) -> None:
    def question(user: str) -> dict:
        key = re.search(r"Identifying values: (.+)", user).group(1)
        values = [part.split(" = ")[1] for part in key.split(", ")]
        return {"question": f"What is the price for the item {' from '.join(values)}?"}

    install(config, tmp_path, QuestionDraft=question)
    gen = generator(requirements(capabilities_needed=["structured_lookup"]), corpus, config)
    gen.tool_cases(4)
    cases = [c.case for c in gen.out.candidates]
    assert cases and all(c.expected_tools == ["csv_lookup"] and c.origin == "lookup" for c in cases)
    table = corpus.tables["parts.csv"]
    cells = {cell for row in table.rows for cell in row.values()}
    assert all(c.key_facts[0].text in cells for c in cases)  # taken from a real cell, never from the model


def test_identifying_columns_find_the_smallest_unique_key(corpus) -> None:
    assert Generator.identifying_columns(corpus.tables["parts.csv"]) == ["part_no", "supplier"]


# ---- selection, dedup, split, consistency -----------------------------------------------------------

def case(i: int, type_: str = "answerable", doc: str = "a", question: str | None = None, **kwargs) -> Candidate:
    return Candidate(EvalCase(id=f"c{i}", type=type_, question=question or f"unique question number {i} about {doc}",
                              expected_status="answered", **kwargs), group=doc)


def test_selection_takes_turns_across_groups() -> None:
    chosen = select([case(i, doc="a") for i in range(6)] + [case(10 + i, doc="b") for i in range(2)], 4)
    assert [c.group for c in chosen] == ["a", "b", "a", "b"]


def test_near_duplicate_questions_keep_only_the_first() -> None:
    kept, rejected = dedupe([case(1, question="how many days of annual leave do I get"),
                             case(2, question="how many days of annual leave do I get again"),
                             case(3, question="what is the hotel cap per night")], FakeEmbedder(), 0.9)
    assert [c.case.id for c in kept] == ["c1", "c3"] and rejected[0].reason.startswith("duplicate")


def test_split_is_stratified_by_type_and_stable() -> None:
    cases = [case(i, type_=t).case.model_copy(update={"id": f"{t}-{i}"})
             for t in ("answerable", "multidoc") for i in range(10)]
    first = assign_split(cases)
    second = assign_split(cases)
    assert first == second
    for type_ in ("answerable", "multidoc"):
        in_test = [c for c in cases if c.type == type_ and c.split == "test"]
        assert len(in_test) == 3  # 30% of 10, per type
    assert set(first["dev"]).isdisjoint(first["test"]) and len(first["dev"]) + len(first["test"]) == 20


def test_targets_move_tool_slots_when_no_tool_is_needed() -> None:
    assert sum(targets_for(requirements(), True).values()) == TARGET_TOTAL
    plain = targets_for(requirements(capabilities_needed=["document_qa"]), True)
    assert plain["tool_required"] == 0 and plain["answerable"] == 14 and plain["multidoc"] == 7
    assert sum(plain.values()) == TARGET_TOTAL


def refusal(ref: str, origin: str, question: str = "Can you help with parking permits?") -> EvalCase:
    return EvalCase(id="x", type="out_of_scope", question=question, expected_status="refused", origin=origin, topic_ref=ref)


def test_consistency_catches_contradictory_cases() -> None:
    req = requirements()
    assert violation(refusal("T1", "topic"), req) == "refusal expected on must-answer topic T1"
    assert violation(refusal("G-T3", "gap"), req) is None  # a real gap
    answered_refused = EvalCase(id="y", type="answerable", question="q", expected_status="answered", topic_ref="R1")
    assert "must-refuse topic R1" in violation(answered_refused, req)
    answered_gap = EvalCase(id="z", type="answerable", question="q", expected_status="answered", topic_ref="G-T3")
    assert "gap G-T3" in violation(answered_gap, req)
    restating = refusal("R1", "must_refuse", "What are the annual leave entitlements for my team?")
    assert "restates must-answer topic T1" in violation(restating, req)
    with pytest.raises(EvalConsistencyError):
        assert_cases_consistent([refusal("T1", "topic")], req)


def test_a_live_aspect_refusal_on_a_split_topic_is_allowed_but_a_whole_topic_refusal_is_not() -> None:
    split = requirements(
        must_answer_topics=[Topic(id="T1", topic="a book on the shelf", supported_aspects=["recorded location"])],
        gaps=[Gap(id="G-LIVE-1", kind="live_data", topic_id="T1", aspect="whether it is on the shelf now",
                  recommended_action="refuse_with_redirect", description="live"),
              Gap(id="G-WHOLE", kind="live_data", topic_id="T1", recommended_action="refuse_with_redirect", description="live")])
    assert violation(refusal("G-LIVE-1", "gap", "Is the book on the shelf right now?"), split) is None
    assert "whole must-answer topic" in violation(refusal("G-WHOLE", "gap"), split)


# ---- the whole pipeline with a scripted model ---------------------------------------------------------

def full_handlers() -> dict:
    return {
        "AnswerableDraft": good_answerable,
        "MergedQuestion": lambda u: {"question": "Putting both together, what leave and ordering rules apply to me?"},
        "AmbiguousDraft": lambda u: {"question": "How many days do I get?", "reading_a": "annual", "reading_b": "sick"},
        "QuestionsDraft": lambda u: {"questions": ["A realistic refusal question " + (re.search(r"Subject[^:]*: (.+)", u).group(1) if "Subject" in u else "about tools") + "?"]},
        "QuestionDraft": lambda u: {"question": "What is the price for HD-1 from S01?"},
        "TemplateChoice": lambda u: {"template": "product", "bindings": [], "question": "?"},
        "AnswerBearing": lambda u: {"reason": "it is the answer", "needed": True},
        "AnswerFromContext": lambda u: {"answer": passage_of(u, "Passage\(s\)")},
        "Probes": lambda u: {"questions": ["Is the cafe open on Sunday?", "What does the cafe sell near the entrance?"]},
        "Support": lambda u: {"chunk_id": "", "span": "NONE"},
        "JudgeSupport": lambda u: {"reason": "", "quote": "", "answers": False},
    }


def test_the_full_build_writes_every_artifact_and_never_exceeds_the_target(corpus, config, tmp_path) -> None:
    install(config, tmp_path, **full_handlers())
    result = build_eval_set(requirements(), corpus, config, FakeEmbedder(), prefix="t")
    assert 0 < len(result.cases) <= TARGET_TOTAL
    types = {c.type for c in result.cases}
    assert {"answerable", "out_of_scope"} <= types
    assert all(c.id.startswith("t-") and c.split in ("dev", "test") for c in result.cases)
    assert_cases_consistent(result.cases, requirements())
    out = tmp_path / "03_evals"
    write_eval_set(result, out)
    for name in ("eval_set.jsonl", "rejected_cases.jsonl", "split.json", "generation_stats.json"):
        assert (out / name).is_file(), name
    assert (out / "prompts_used" / "answerable.md").is_file() and (out / "prompts_used" / "versions.yaml").is_file()
    stats = json.loads((out / "generation_stats.json").read_text())
    assert stats["valid_cases"] == len(result.cases) and "rejections_by_reason" in stats
    for line in (out / "eval_set.jsonl").read_text().splitlines():  # every seed span is real text at real offsets
        for span in json.loads(line)["seed_spans"]:
            assert corpus.docs[span["doc"]].canonical_text[span["char_start"]:span["char_end"]] == span["span_text"]


# ---- isolation ----------------------------------------------------------------------------------------

def test_the_eval_generator_never_touches_the_agent_definition_or_fact_sheets() -> None:
    forbidden_modules = ("forgeai.generator", "forgeai.audit", "forgeai.runner.agent_def", "forgeai.runner.runner")
    forbidden_text = ("agent.yaml", "agent.v0", "tool_selection", "fact_sheet", "agent_def")
    for path in (PACKAGE / "evals").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef)) and n.body
                      and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant)}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(forbidden_modules), (path.name, node.module)
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith(forbidden_modules) for a in node.names), path.name
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                assert not any(t in node.value for t in forbidden_text), (path.name, node.value[:40])


def test_the_audit_package_is_the_only_reader_of_fact_sheets() -> None:
    readers = [p.relative_to(PACKAGE).parts[0] for p in PACKAGE.rglob("*.py")
               if "fact_sheet" in p.read_text(encoding="utf-8")]
    assert set(readers) <= {"audit"}, readers  # even the CLI only passes an opaque path through


# ---- key facts are atomic; the audit --------------------------------------------------------------

def test_sentence_length_key_facts_are_not_accepted_as_facts(corpus, config, tmp_path) -> None:
    """Regression: the first full run produced 'key facts' that were whole sentences (up to 20 of them)."""
    def handler(user: str) -> dict:
        draft = good_answerable(user)
        draft["key_facts"] = ["Employees must submit a written request to their line manager by 31 January to carry over"]
        return draft

    install(config, tmp_path, AnswerableDraft=handler)
    gen = generator(requirements(), corpus, config)
    gen.answerable(4)
    assert not gen.out.candidates and {r.reason for r in gen.out.rejected} >= {"no key facts"}


def test_the_audit_reports_coverage_and_checks_key_facts_against_planted_text(tmp_path) -> None:
    from forgeai.audit.audit import audit_eval_set
    from forgeai.ingest.spans import SeedSpan

    sheet = tmp_path / "planted.json"
    sheet.write_text(json.dumps({"demo": "x", "facts": [
        {"id": "F1", "doc": "a.md", "statement": "24 working days of annual leave", "must_contain": ["24 working days"]},
        {"id": "F2", "doc": "a.md", "statement": "5% discount", "must_contain": ["5%"]},
        {"id": "F3", "doc": "b.md", "statement": "never reached", "must_contain": ["xyzzy"]}]}), encoding="utf-8")

    def span(text: str) -> SeedSpan:
        return SeedSpan(doc="a.md", char_start=0, char_end=len(text), span_text=text)

    cases = [
        EvalCase(id="1", type="answerable", question="q", expected_status="answered",
                 key_facts=[KeyFact(text="24", kind="number")], seed_spans=[span("They receive 24 working days of leave.")]),
        EvalCase(id="2", type="answerable", question="q", expected_status="answered",
                 key_facts=[KeyFact(text="25", kind="number")], seed_spans=[span("They receive 24 working days (25 on request).")]),
        EvalCase(id="3", type="answerable", question="q", expected_status="answered",
                 key_facts=[KeyFact(text="15", kind="number")], seed_spans=[span("A 15% expedite surcharge applies.")]),
    ]
    result = audit_eval_set(cases, sheet)
    assert result["coverage"]["fully_covered"] == 1 and result["coverage"]["not_reached_ids"] == ["F2", "F3"]  # "5%" is not inside "15%"
    agreement = result["key_fact_agreement"]
    assert (agreement["checked"], agreement["agree"]) == (2, 1)  # 24 agrees; 25 contradicts the planted 24
    assert agreement["not_checkable_no_planted_fact_in_span"] == 1  # the 15% span holds no planted fact
    assert agreement["disagreements"][0]["case"] == "2"


def test_fact_kinds_and_dangling_fragments() -> None:
    kinds = {t: Generator.fact_kind(t) for t in ["24 working days", "USD 5,000", "5%", "31 January", "Ethics Hotline", "2026-11-20"]}
    assert kinds == {"24 working days": "number", "USD 5,000": "number", "5%": "number", "31 January": "phrase",
                     "Ethics Hotline": "phrase", "2026-11-20": "date"}


def test_dangling_fragments_and_page_footers_are_not_used(corpus, config, tmp_path) -> None:
    install(config, tmp_path)
    gen = generator(requirements(), corpus, config)
    assert gen._facts(["and confirm the surcharge accordingly", "or the invoice", "24 working days"]) == [
        KeyFact(text="24 working days", kind="number")]
    footer = corpus.chunks[0].model_copy(update={"chunk_id": "f#c0", "kind": "prose",
                                                   "text": "Calder Bay Ship Management Ltd: Leave and Holiday Policy Page 1 of the document"})
    corpus.chunks.append(footer)
    assert "f#c0" not in {c.chunk_id for c in corpus.seed_chunks("prose")}


def test_a_question_about_a_table_row_must_name_the_values_that_identify_it(corpus, config, tmp_path) -> None:
    row = next(c for c in corpus.chunks if c.kind == "row" and "HD-1 | supplier: S01" in c.text)
    fact_handler = lambda q: (lambda u: {"question": q, "answer_span": "unit_price_usd: 420.00", "key_facts": ["420.00"]})
    install(config, tmp_path, AnswerableDraft=fact_handler("What is the price of the piston ring?"))
    gen = generator(requirements(), corpus, config)
    gen._one_answerable(row, salt=1, snapshot=None)
    assert [r.reason for r in gen.out.rejected] == ["question does not identify one table row"]
    fresh = config.model_copy(update={"cache_dir": tmp_path / "fresh"})
    install(fresh, tmp_path, AnswerableDraft=fact_handler("What is the price of HD-1 from S01?"))
    gen = generator(requirements(), corpus, fresh)
    gen._one_answerable(row, salt=1, snapshot=None)
    assert len(gen.out.candidates) == 1


def test_a_multidoc_question_that_one_document_answers_is_rejected(config, tmp_path) -> None:
    """Regression: 'requirements for personal email' was labelled multi-document but only needed one policy."""
    from forgeai.evals.build import judge_answerability

    case = EvalCase(id="", type="multidoc", question="q", expected_status="answered",
                    key_facts=[KeyFact(text="alpha rule", kind="phrase"), KeyFact(text="beta rule", kind="phrase")])
    cand = Candidate(case, ["passage a", "passage b"], fact_groups=[[KeyFact(text="alpha rule", kind="phrase")],
                                                                      [KeyFact(text="beta rule", kind="phrase")]])
    install(config, tmp_path, AnswerFromContext=lambda u: {"answer": "The alpha rule applies."})
    kept, rejected = judge_answerability([cand], config)
    assert not kept and "independent answerability failed" in rejected[0].reason
    fresh = config.model_copy(update={"cache_dir": tmp_path / "fresh"})
    install(fresh, tmp_path, AnswerFromContext=lambda u: {"answer": "The alpha rule and the beta rule both apply."})
    kept, _ = judge_answerability([cand], fresh)
    assert kept == [cand]


def test_a_phrase_fact_with_a_quantity_is_recovered_when_the_answer_states_the_same_quantity() -> None:
    from forgeai.evals.matching import quantity_match

    assert quantity_match("capped at 15 dollars per item", "$15") and quantity_match("copies: 3", "There are 3 copies available.")
    assert not quantity_match("capped at 15 dollars per item", "15 days") and not quantity_match("30 days", "It is 20 days.")
    assert not quantity_match("the Ethics Hotline", "the Ethics Hotline")  # no quantity: not this function's business
