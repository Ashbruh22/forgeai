"""The judge (P6): layer 1 checks, layer 2 with its verification rules, the pass rule, metrics, canaries."""

from __future__ import annotations

import itertools
from datetime import date

import pytest

from forgeai.config import ModelSpec
from forgeai.evals.matching import numbers_in
from forgeai.evals.models import EvalCase, Interpretation, KeyFact
from forgeai.ingest.models import Chunk
from forgeai.ingest.spans import SeedSpan
from forgeai.judge.canaries import (
    GOLD, build_canaries, build_gold, choose_judge, concise_correct, negation, number_swap, padded_correct, run_canaries,
    wrong_document,
)
from forgeai.judge.layer1 import check_layer1, match_fact
from forgeai.judge.layer2 import format_context
from forgeai.judge.metrics import scorecard, wilson
from forgeai.judge.models import CaseRun, CitedChunk, RetrievedChunk
from forgeai.judge.score import score_case
from forgeai.runner.contract import Answer, Citation, ToolCallRecord
from tests.test_evals import config, install as _install  # noqa: F401  (fixture and helper)

TODAY = date(2026, 11, 27)
JUDGE = ModelSpec(name="llama3.1:8b", temperature=0.0)
TEXT_A = "Full-time employees receive 24 working days of annual leave per calendar year, accruing monthly."
TEXT_B = "Orders from USD 15,000 to USD 49,999 receive an 8% discount on the list price."
TEXT_C = "Receipts must be submitted within 30 days of the purchase and are never accepted after the deadline."


_installs = itertools.count()


def install(config, tmp_path, **handlers):
    """A fresh response cache per call, so a second scripted judge is not answered from the first one's cache."""
    return _install(config.model_copy(update={"cache_dir": tmp_path / f"cache{next(_installs)}"}), tmp_path, **handlers)


def chunk(cid: str, doc: str, text: str, start: int) -> Chunk:
    return Chunk(chunk_id=cid, doc=doc, index=0, text=text, char_start=start, char_end=start + len(text))


CHUNKS = [chunk("a.md#c0", "a.md", TEXT_A, 100), chunk("b.md#c0", "b.md", TEXT_B, 40), chunk("c.md#c0", "c.md", TEXT_C, 0)]


def span(doc: str, text: str, start: int) -> SeedSpan:
    return SeedSpan(doc=doc, char_start=start, char_end=start + len(text), span_text=text)


def kf(text: str, kind: str = "phrase") -> KeyFact:
    return KeyFact(text=text, kind=kind)


def case(**kw) -> EvalCase:
    base = dict(id="x-1", type="answerable", question="How much annual leave does grade G7 staff get?", expected_status="answered",
                split="dev", seed_spans=[span("a.md", TEXT_A, 100)], key_facts=[kf("24", "number")], expected_sources=["a.md"])
    return EvalCase(**{**base, **kw})


def run_of(text: str = TEXT_A, *, status: str = "answered", cite: list[Chunk] | None = None, retrieved: list[Chunk] | None = None,
           issues: list[str] | None = None, tools: list[ToolCallRecord] | None = None, clarifying: str | None = None) -> CaseRun:
    cite = CHUNKS[:1] if cite is None else cite
    seen = CHUNKS if retrieved is None else retrieved
    answer = Answer(status=status, answer=text, citations=[Citation(doc=c.doc, chunk_id=c.chunk_id, quote=c.text[:30]) for c in cite],
                    tool_calls=tools or [], clarifying_question=clarifying)
    return CaseRun(case_id="x-1", answer=answer, issues=issues or [],
                   retrieved=[RetrievedChunk(chunk_id=c.chunk_id, doc=c.doc, char_start=c.char_start, char_end=c.char_end, text=c.text) for c in seen],
                   cited=[CitedChunk(chunk_id=c.chunk_id, doc=c.doc, char_start=c.char_start, char_end=c.char_end) for c in cite])


def match(fact: KeyFact, text: str, tolerance=(1e-9, 0.0)):
    return match_fact(fact, text, tolerance=tolerance, today=TODAY, fuzzy_min=85)


# ---- layer 1 -----------------------------------------------------------------------------------------------------


def test_numbers_are_compared_by_value_not_by_spelling() -> None:
    assert match(kf("5000", "number"), "The limit is USD 5,000.").matched
    assert match(kf("24", "number"), "They get twenty-four days.").matched
    assert not match(kf("24", "number"), "They get 42 days.").matched
    assert not match(kf("24", "number"), "They get 240 days.").matched


def test_a_computed_value_is_matched_within_the_tool_tolerance_only_when_tools_are_expected() -> None:
    assert match(kf("18101.67", "number"), "The cost is 18,101.7.", tolerance=(0.01, 0.001)).matched
    assert not match(kf("18101.67", "number"), "The cost is 18,101.7.").matched
    assert not match(kf("18101.67", "number"), "The cost is 19,000.", tolerance=(0.01, 0.001)).matched


def test_phrases_match_by_substring_then_fuzzily_and_a_dropped_negation_or_number_never_matches() -> None:
    assert match(kf("USD 15,000 and USD 49,999"), "Orders between USD 15,000 and USD 49,999 get 8%.").method == "exact"
    assert match(kf("stock_port: Rotterdam"), "The item is held in Rotterdam.").matched  # the value of "column: value" alone
    assert match(kf("annual leave per calendar year"), "Per calendar year, annual leave is 24 days.").method == "fuzzy"
    assert not match(kf("not permitted"), "This is permitted.").matched
    assert not match(kf("USD 15,000 and USD 49,999"), "Orders between USD 15,000 and USD 59,999 get 8%.").matched


def test_dates_match_in_any_format_and_a_relative_date_is_resolved_against_today() -> None:
    assert match(kf("2026-11-01", "snapshot_date"), "As of 1 November 2026 it is 5.").matched
    assert not match(kf("2026-11-01", "snapshot_date"), "It is 5.").matched
    assert match(kf("today + 18 days", "relative_date"), "It arrives on 2026-12-15.").matched
    assert not match(kf("today + 18 days", "relative_date"), "It arrives on 2026-12-16.").matched


def test_source_correctness_uses_seed_span_overlap_and_a_multidoc_case_needs_every_document() -> None:
    multi = case(type="multidoc", seed_spans=[span("a.md", TEXT_A, 100), span("b.md", TEXT_B, 40)], key_facts=[kf("24", "number")])
    assert check_layer1(multi, run_of(cite=CHUNKS[:1]), today=TODAY).sources_missing == ["b.md"]
    assert check_layer1(multi, run_of(cite=CHUNKS[:2]), today=TODAY).source_ok is True
    assert check_layer1(case(), run_of(cite=[CHUNKS[2]]), today=TODAY).source_ok is False  # another document
    narrow = case(seed_spans=[span("a.md", TEXT_A[:20], 100)])
    far = [chunk("a.md#c9", "a.md", "elsewhere in the same document", 900)]
    assert check_layer1(narrow, run_of(cite=far), today=TODAY).source_ok is False  # same document, no overlap


def test_retrieval_hit_and_rank_come_from_overlap_with_the_seed_span() -> None:
    result = check_layer1(case(), run_of(retrieved=[CHUNKS[2], CHUNKS[1], CHUNKS[0]]), today=TODAY)
    assert result.retrieval_hit is True and result.retrieval_rank == 3
    assert check_layer1(case(), run_of(retrieved=[CHUNKS[2]]), today=TODAY).retrieval_hit is False
    refusal = case(type="out_of_scope", expected_status="refused", seed_spans=[], key_facts=[])
    assert check_layer1(refusal, run_of("I cannot help.", status="refused", cite=[]), today=TODAY).retrieval_hit is None


def test_contract_citation_and_status_checks_and_tool_correctness() -> None:
    bad = run_of(issues=["citation: quote is not a verbatim span of chunk 'a.md#c0': 'x'", "schema: <root>: bad"])
    result = check_layer1(case(), bad, today=TODAY)
    assert not result.contract_valid and result.citation_issues and result.citations_valid == 0
    assert not check_layer1(case(), run_of(status="refused"), today=TODAY).status_ok
    tool_case = case(type="tool_required", expected_tools=["calculator"], key_facts=[kf("19", "number")])
    called = run_of("The total is 19.0.", tools=[ToolCallRecord(tool="calculator", args={}, result="19.0")])
    assert check_layer1(tool_case, called, today=TODAY).tool_ok is True
    missing = check_layer1(tool_case, run_of("The total is 19.0."), today=TODAY)
    assert missing.tool_ok is False and "was not called" in missing.tool_notes[0]
    errored = run_of("19", tools=[ToolCallRecord(tool="calculator", args={}, result="ERROR: bad")])
    assert check_layer1(tool_case, errored, today=TODAY).tool_ok is False


def test_an_ambiguous_case_counts_the_readings_an_answer_covers() -> None:
    def reading(text: str, doc: str, start: int) -> Interpretation:
        return Interpretation(reading=text, discriminators=[], key_facts=[kf(text, "number")], seed_span=span(doc, TEXT_A, start))

    amb = case(type="ambiguous", expected_status="needs_clarification", key_facts=[], seed_spans=[],
               interpretations=[reading("12", "a.md", 100), reading("18", "b.md", 40)])
    both = check_layer1(amb, run_of("It is 12 months or 18 months."), today=TODAY)
    assert (both.interpretations_covered, both.interpretations_total) == (2, 2)
    assert check_layer1(amb, run_of("It is 12 months."), today=TODAY).interpretations_covered == 1
    assert check_layer1(amb, run_of("", status="needs_clarification", clarifying="Which one?", cite=[]), today=TODAY).status_ok


# ---- layer 2 -----------------------------------------------------------------------------------------------------


def judge_handlers(*, claims=("Grade G7 staff receive annual leave.",), supported=True, quote="annual leave per calendar year",
                   expressed=False, equiv_quote=""):
    return dict(
        ClaimList=lambda u: {"claims": list(claims)},
        SupportVerdict=lambda u: {"reason": "r", "supported": supported, "missing": "" if supported else "nothing says so", "quote": quote},
        Equivalence=lambda u: {"reason": "r", "expressed": expressed, "quote": equiv_quote})


def test_a_supported_claim_needs_a_quote_that_is_really_in_the_context(config, tmp_path) -> None:
    install(config, tmp_path, **judge_handlers())
    assert score_case(case(), run_of(), JUDGE, today=TODAY).layer2.unsupported == 0
    install(config, tmp_path, **judge_handlers(quote="a sentence the context does not contain"))
    score = score_case(case(), run_of(), JUDGE, today=TODAY)
    assert score.layer2.unsupported == 1 and "not in the evidence" in score.layer2.claims[0].missing
    assert not score.passed and "unsupported claim" in score.reasons[-1]


def test_a_failed_key_fact_is_upgraded_only_by_a_verified_quote_that_carries_its_numbers_and_negation(config, tmp_path) -> None:
    c = case(key_facts=[kf("entitled to 24 days of holiday allowance")])
    text = "Staff get 24 days off in the year."
    install(config, tmp_path, **judge_handlers(expressed=True, equiv_quote="get 24 days off"))
    good = score_case(c, run_of(text), JUDGE, today=TODAY)
    assert good.passed and good.layer1.facts[0].method == "judge_upgrade" and not good.passed_without_layer2_upgrades
    assert [d["kind"] for d in good.disagreements] == ["fact_upgraded"]
    for quote, why in [("get 25 days off", "no verbatim quote"), ("off in the year", "numbers"), ("", "no verbatim quote")]:
        install(config, tmp_path, **judge_handlers(expressed=True, equiv_quote=quote))
        refused = score_case(c, run_of(text), JUDGE, today=TODAY)
        assert not refused.passed and refused.disagreements[0]["kind"] == "upgrade_refused" and why in refused.disagreements[0]["why_not"]
    install(config, tmp_path, **judge_handlers(expressed=True, equiv_quote="are not entitled"))
    neg = score_case(case(key_facts=[kf("entitled to 24 days")]), run_of("Staff are not entitled to 25 days."), JUDGE, today=TODAY)
    assert not neg.passed


def test_layer_two_is_skipped_for_refusals_without_figures_and_flags_claims_in_a_refusal_with_them(config, tmp_path) -> None:
    refusal = case(type="out_of_scope", expected_status="refused", seed_spans=[], key_facts=[])
    router = install(config, tmp_path, **judge_handlers(claims=()))
    plain = score_case(refusal, run_of("I cannot help with that.", status="refused", cite=[]), JUDGE, today=TODAY)
    assert plain.passed and router.calls == [] and not plain.layer2.ran
    install(config, tmp_path, **judge_handlers(claims=("The fee is 40 dollars.",)))
    figures = score_case(refusal, run_of("I cannot help, but the fee is 40 dollars.", status="refused", cite=[]), JUDGE, today=TODAY)
    assert not figures.passed and figures.layer2.refusal_claims and figures.disagreements[0]["kind"] == "refused_but_states_claims"


def test_the_judge_prompts_carry_no_model_name_iteration_or_case_id(config, tmp_path) -> None:
    router = install(config, tmp_path, **judge_handlers(expressed=True, equiv_quote="x"))
    score_case(case(id="sp-T02", key_facts=[kf("full-time staff get 24 days off")]), run_of("Staff get days off."), JUDGE, today=TODAY)
    assert router.calls
    for _, prompt in router.calls:
        lowered = prompt.lower()
        assert not any(word in lowered for word in ("qwen", "llama", "gemma", "iteration", "candidate", "sp-t02", "x-1", "agent.v"))


def test_context_for_the_judge_is_the_retrieved_chunks_and_the_tool_results() -> None:
    run = run_of(tools=[ToolCallRecord(tool="calculator", args={"expression": "2*3"}, result="6")])
    context = format_context(run)
    assert TEXT_A in context and "calculator" in context and "gave 6" in context


# ---- pass rule ---------------------------------------------------------------------------------------------------


def verdict(c: EvalCase, run: CaseRun, config, tmp_path, **handlers):
    install(config, tmp_path, **judge_handlers(**handlers))
    return score_case(c, run, JUDGE, today=TODAY)


def test_answerable_passes_only_when_every_check_passes(config, tmp_path) -> None:
    assert verdict(case(), run_of(), config, tmp_path).passed
    assert not verdict(case(), run_of("They get 42 days."), config, tmp_path).passed  # key fact
    assert not verdict(case(), run_of(cite=[CHUNKS[2]]), config, tmp_path).passed  # source
    assert not verdict(case(), run_of(), config, tmp_path, supported=False).passed  # grounding
    snap = case(key_facts=[kf("24", "number"), kf("2026-11-01", "snapshot_date")])
    assert not verdict(snap, run_of(), config, tmp_path).passed  # the snapshot date is required
    assert verdict(snap, run_of(TEXT_A + " As of 2026-11-01."), config, tmp_path).passed


def test_out_of_scope_ambiguous_and_tool_cases_follow_their_own_rules(config, tmp_path) -> None:
    refusal = case(type="out_of_scope", expected_status="refused", seed_spans=[], key_facts=[])
    assert verdict(refusal, run_of("Please ask someone else.", status="refused", cite=[]), config, tmp_path).passed
    assert not verdict(refusal, run_of("It is 24 days.", cite=[]), config, tmp_path).passed
    amb = case(type="ambiguous", expected_status="needs_clarification", key_facts=[], seed_spans=[], interpretations=[
        Interpretation(reading="r", discriminators=[], key_facts=[kf("12", "number")], seed_span=span("a.md", TEXT_A, 100)),
        Interpretation(reading="s", discriminators=[], key_facts=[kf("18", "number")], seed_span=span("b.md", TEXT_B, 40))])
    ask = run_of("", status="needs_clarification", clarifying="Which policy?", cite=[])
    assert verdict(amb, ask, config, tmp_path).passed
    assert not verdict(amb, run_of("It is 12 months."), config, tmp_path).passed
    tool_case = case(type="tool_required", expected_tools=["calculator"], key_facts=[kf("19", "number")])
    ok = run_of("The total is 19.", tools=[ToolCallRecord(tool="calculator", args={}, result="19")])
    assert verdict(tool_case, ok, config, tmp_path, claims=("The total is 19.",), quote="Tool result: calculator {} gave 19").passed
    assert not verdict(tool_case, run_of("The total is 19."), config, tmp_path).passed


def test_a_layer_two_verdict_is_flagged_low_confidence_when_the_judge_failed_a_canary_type(config, tmp_path) -> None:
    install(config, tmp_path, **judge_handlers(supported=False))
    flagged = score_case(case(), run_of(), JUDGE, today=TODAY, low_confidence_types=["negation"])
    unflagged = score_case(case(), run_of(), JUDGE, today=TODAY)
    assert flagged.low_confidence and not unflagged.low_confidence


# ---- metrics -----------------------------------------------------------------------------------------------------


def test_wilson_interval_matches_the_textbook_values() -> None:
    low, high = wilson(8, 10)
    assert 0.48 < low < 0.50 and 0.93 < high < 0.95
    assert wilson(0, 0) == (0.0, 0.0)
    assert wilson(10, 10)[1] == 1.0 and wilson(10, 10)[0] > 0.7


def test_the_scorecard_reports_every_metric_of_the_spec(config, tmp_path) -> None:
    refusal = case(id="r", type="out_of_scope", expected_status="refused", seed_spans=[], key_facts=[])
    scores = [verdict(case(), run_of(), config, tmp_path), verdict(case(), run_of("42 days."), config, tmp_path),
              verdict(refusal, run_of("No.", status="refused", cite=[]), config, tmp_path),
              verdict(refusal, run_of("It is 24."), config, tmp_path)]
    card = scorecard(scores, [100.0, 300.0], {"agent_in": 10}, ["negation"], k=5)
    assert card["pass_rate"]["passes"] == 2 and card["pass_rate"]["n"] == 4 and len(card["pass_rate"]["wilson95"]) == 2
    assert card["refusal"] == {"precision": 1.0, "recall": 0.5, "refused_by_agent": 1, "refusal_expected": 2, "answered_when_refusal_expected": 1}
    assert card["key_fact_recall"]["total"] == 2 and card["key_fact_recall"]["matched"] == 1
    assert card["retrieval_hit_at_k"]["k"] == 5 and card["latency_ms"]["mean"] == 200
    assert card["layer2_low_confidence_types"] == ["negation"] and "citation_validity" in card and "tool_accuracy" in card


# ---- canaries ----------------------------------------------------------------------------------------------------


def test_gold_answers_are_built_in_code_and_pass_layer_one() -> None:
    built = build_gold(case(), CHUNKS)
    assert built is not None
    gold, _ = built
    assert gold.answer.answer == TEXT_A and gold.answer.citations[0].quote in TEXT_A
    assert check_layer1(case(), gold, today=TODAY).source_ok is True
    assert check_layer1(case(), gold, today=TODAY).contract_valid
    assert build_gold(case(seed_spans=[span("zzz.md", TEXT_A, 0)]), CHUNKS) is None


def test_every_perturbation_changes_what_it_should_and_nothing_else() -> None:
    c = case()
    gold, gold_chunks = build_gold(c, CHUNKS)
    swapped = number_swap(c, gold)
    assert swapped and "24" not in swapped.answer.answer and "annual leave" in swapped.answer.answer
    assert 24.0 not in numbers_in(swapped.answer.answer) and swapped.answer.citations == gold.answer.citations
    assert " not " in negation(gold).answer.answer
    other = wrong_document(c, gold, CHUNKS)
    assert other and other.answer.citations[0].doc != "a.md" and other.answer.answer == gold.answer.answer
    assert check_layer1(c, other, today=TODAY).source_ok is False and check_layer1(c, other, today=TODAY).contract_valid
    padded = padded_correct(gold, gold_chunks).answer.answer
    assert padded.startswith("Thank you") and TEXT_A in padded and len(padded) > len(TEXT_A)
    assert concise_correct(c, gold).answer.answer == "24"
    assert concise_correct(case(key_facts=[]), gold) is None


def test_canary_set_has_a_gold_and_six_perturbations_per_gold_answer() -> None:
    cases = [case(id=f"x-{n}", seed_spans=[span("a.md", TEXT_A, 100)]) for n in range(3)]
    canaries = build_canaries("demo", cases, CHUNKS, per_demo=2)
    kinds = [c.kind for c in canaries]
    assert kinds.count(GOLD) == 2 and len(canaries) == 14 and all(c.should_pass == (c.kind not in (
        "number_swap", "negation", "wrong_document", "fabricated_clause")) for c in canaries)


def test_canary_detection_is_tabulated_per_type_and_a_faithful_judge_scores_well(config, tmp_path) -> None:
    def support(user: str) -> dict:
        claim = user.split("Claim:")[1].split("\n")[0].strip()
        context = user.split('"""')[1]
        ok = all(w.strip(".,") in context for w in claim.split() if w[0].isdigit())  # a toy judge: its numbers are in the context
        return {"reason": "r", "supported": ok, "quote": TEXT_A if ok else ""}

    install(config, tmp_path, ClaimList=lambda u: {"claims": [u.split('"""')[1].strip().split(". ")[0]]},
            SupportVerdict=support, Equivalence=lambda u: {"reason": "", "expressed": False, "quote": ""})
    canaries = build_canaries("demo", [case()], CHUNKS, per_demo=1)
    result = run_canaries(canaries, JUDGE, today=TODAY)
    table = result["table"]
    assert table[GOLD]["pipeline_accuracy"] == 1.0 and table["number_swap"]["pipeline_accuracy"] == 1.0
    assert table["wrong_document"]["pipeline_accuracy"] == 1.0 and table["wrong_document"]["layer2_accuracy"] is None
    assert table["padded_correct"]["pipeline_accuracy"] == 1.0 and table["concise_correct"]["pipeline_accuracy"] == 1.0


def test_the_judge_is_chosen_by_layer_two_accuracy_and_low_types_are_listed() -> None:
    def table(**acc) -> dict:
        base = {k: {"n": 5, "pipeline_accuracy": 1.0, "layer2_accuracy": 1.0} for k in
                ("gold", "number_swap", "negation", "wrong_document", "fabricated_clause", "padded_correct", "concise_correct")}
        base["wrong_document"]["layer2_accuracy"] = None
        for k, v in acc.items():
            base[k]["layer2_accuracy"] = v
        return {"table": base}

    choice = choose_judge({"a": table(negation=0.6, fabricated_clause=0.8), "b": table(negation=1.0, fabricated_clause=0.9)}, 0.9)
    assert choice["chosen"] == "b" and choice["low_confidence_types"] == []
    choice = choose_judge({"a": table(negation=0.6), "b": table(negation=0.7, padded_correct=0.5)}, 0.9)
    assert choice["chosen"] == "a" and choice["low_confidence_types"] == ["negation"]


@pytest.mark.parametrize("kind", ["number", "phrase"])
def test_a_key_fact_that_is_absent_never_matches(kind) -> None:
    assert not match(kf("zebra 77", kind), "Nothing relevant here.").matched
