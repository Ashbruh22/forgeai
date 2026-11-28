"""Judge hardening (P6.1): deduplicated claims, numbers checked by code, answer-bearing rule, negation, NLI, normalisation."""

from __future__ import annotations

from datetime import date

from forgeai.evals.models import KeyFact
from forgeai.judge.claims import (
    answer_bearing, build_spans, dedupe_claims, negation_mismatch, numeric_verdict, question_entities,
)
from forgeai.judge.layer1 import match_fact
from forgeai.judge.layer2 import format_context
from forgeai.judge.score import score_case
from forgeai.runner.contract import ToolCallRecord
from tests.test_evals import config  # noqa: F401  (fixture)
from tests.test_judge import JUDGE, TEXT_A, TEXT_B, case, install, judge_handlers, kf, run_of

TODAY = date(2026, 11, 28)


def test_repeated_claims_are_counted_once_and_a_claim_inside_a_longer_one_is_replaced_by_it() -> None:
    claims = ["Employees get 24 days.", "employees get 24 days", "Employees  get 24 days!", "The limit is 5 items.",
              "Staff may park for 2 hours."]
    assert dedupe_claims(claims) == ["Employees get 24 days.", "The limit is 5 items.", "Staff may park for 2 hours."]
    assert dedupe_claims(["Fujairah is a port.", "Fujairah is a port in the east."]) == ["Fujairah is a port in the east."]
    distinct = ["Staff may park for 2 hours.", "Receipts expire after 30 days.", "Fines are capped at 15 dollars.", "The desk opens at nine.",
                "Parcels need a signature.", "Rooms seat twelve people.", "Loans renew twice.", "Cards cost 5 dollars.", "Quiet hours begin at ten."]
    assert dedupe_claims(distinct) == distinct[:8]


def verdict_of(claim: str, run=None):
    return numeric_verdict(claim, build_spans(run or run_of()))


def test_a_claim_with_a_number_is_decided_by_code_against_the_evidence() -> None:
    assert verdict_of("Full-time employees receive 24 working days of annual leave.").verdict == "supported"
    wrong = verdict_of("Full-time employees receive 25 working days of annual leave.")
    assert wrong.verdict == "unsupported" and "25" in wrong.missing and not wrong.supported
    assert verdict_of("Employees receive annual leave.") is None  # no figure: the judge model decides


def test_a_number_that_occurs_only_in_unrelated_text_does_not_support_a_claim() -> None:
    other = verdict_of("Receipts are due within 24 days of the purchase.")
    assert other.verdict == "unsupported" and "something else" in other.missing


def test_a_negation_the_evidence_lacks_makes_a_numeric_claim_contradicted_and_so_does_the_reverse() -> None:
    flipped = verdict_of("Full-time employees do not receive 24 working days of annual leave.")
    assert flipped.verdict == "contradicted" and "negat" in flipped.missing
    assert verdict_of("Receipts are accepted after 30 days of the purchase.").verdict == "contradicted"  # the text says never accepted after
    assert negation_mismatch("It is allowed", "It is not allowed") and not negation_mismatch("It is allowed", "It is allowed")


def test_a_value_reported_by_a_tool_supports_a_claim_even_when_the_sentence_shares_little_with_it() -> None:
    run = run_of(tools=[ToolCallRecord(tool="calculator", args={"expression": "15 * 14.5"}, result="217.5")])
    assert numeric_verdict("The order comes to USD 217.50 in total.", build_spans(run)).verdict == "supported"
    assert numeric_verdict("The order comes to USD 218 in total.", build_spans(run)).verdict == "unsupported"
    dates = verdict_of("The item is valid on 2026-12-15.")
    assert dates.verdict == "unsupported" and "2026-12-15" in dates.missing


def test_tool_results_are_given_to_the_judge_in_full_and_first() -> None:
    long_result = "row | " * 700  # about 4,000 characters, more than the room left after a chunk
    run = run_of(tools=[ToolCallRecord(tool="csv_lookup", args={}, result=long_result + "LAST")])
    context = format_context(run)
    assert context.startswith("Tool result: csv_lookup") and context.count("row |") == 700 and "LAST" in context


def test_answer_bearing_claims_carry_a_figure_a_key_fact_or_an_entity_of_the_question() -> None:
    c = case(question="How many copies of 'Pride and Prejudice' does branch B12 hold?", key_facts=[kf("Rotterdam")])
    assert question_entities(c.question) >= {"pride and prejudice", "b12"}
    assert answer_bearing("There are 3 copies.", c, TODAY) and answer_bearing("It is held in Rotterdam.", c, TODAY)
    assert answer_bearing("Branch B12 is open.", c, TODAY) and answer_bearing("Pride and Prejudice is a novel.", c, TODAY)
    assert not answer_bearing("Libraries are quiet places.", c, TODAY)


def partial_support(user: str) -> dict:
    mine = "G7" in user.split("Claim:")[1]
    return {"reason": "r", "supported": mine, "missing": "nothing says so", "quote": "annual leave per calendar year" if mine else ""}


def test_one_unsupported_claim_that_bears_no_answer_is_logged_not_failed_but_two_fail(config, tmp_path) -> None:
    quiet = ("Libraries are quiet places.",)
    handlers = judge_handlers(claims=("Grade G7 staff receive annual leave.", *quiet))
    install(config, tmp_path, **{**handlers, "SupportVerdict": partial_support})
    score = score_case(case(), run_of(), JUDGE, today=TODAY)
    assert score.passed and score.low_confidence and score.layer2.minor_unsupported == list(quiet)
    handlers = judge_handlers(claims=("Grade G7 staff receive annual leave.", "Libraries are quiet.", "Books are free."))
    install(config, tmp_path, **{**handlers, "SupportVerdict": partial_support})
    two = score_case(case(), run_of(), JUDGE, today=TODAY)
    assert not two.passed and two.layer2.unsupported == 2


def test_an_unsupported_answer_bearing_claim_fails_the_case_and_names_the_claim_and_what_is_missing(config, tmp_path) -> None:
    install(config, tmp_path, **judge_handlers(supported=False))
    score = score_case(case(), run_of(), JUDGE, today=TODAY)
    assert not score.passed and score.reasons[-1].startswith("unsupported claim 'Grade G7 staff receive annual leave.'")
    assert "nothing says so" in score.reasons[-1]


def test_a_supported_verdict_whose_quote_negates_the_claim_is_overridden(config, tmp_path) -> None:
    install(config, tmp_path, **judge_handlers(claims=("Grade G7 staff may not accrue annual leave.",), quote="accruing monthly"))
    score = score_case(case(), run_of(), JUDGE, today=TODAY)
    assert score.layer2.claims[0].verdict == "contradicted" and not score.passed


class FakeNli:
    def __init__(self, contradiction: float) -> None:
        self.contradiction = contradiction

    def probabilities(self, premise: str, hypothesis: str) -> dict[str, float]:
        return {"contradiction": self.contradiction, "entailment": 1 - self.contradiction, "neutral": 0.0}


def test_an_nli_contradiction_overrides_a_supported_verdict_but_never_creates_support(config, tmp_path) -> None:
    install(config, tmp_path, **judge_handlers())
    assert score_case(case(), run_of(), JUDGE, today=TODAY, nli=FakeNli(0.2)).passed
    install(config, tmp_path, **judge_handlers())
    flagged = score_case(case(), run_of(), JUDGE, today=TODAY, nli=FakeNli(0.95), nli_min=0.7)
    assert not flagged.passed and flagged.layer2.claims[0].verdict == "contradicted" and "NLI" in flagged.layer2.claims[0].missing
    install(config, tmp_path, **judge_handlers(supported=False))
    assert not score_case(case(), run_of(), JUDGE, today=TODAY, nli=FakeNli(0.0)).passed


def fact(text: str, kind: str = "phrase") -> KeyFact:
    return KeyFact(text=text, kind=kind)


def matched(fact_: KeyFact, answer: str) -> bool:
    return match_fact(fact_, answer, tolerance=(1e-9, 0.0), today=TODAY, fuzzy_min=85).matched


def test_a_cell_fact_matches_the_value_in_a_sentence_whatever_the_column_label() -> None:
    assert matched(fact("shelf: H-02"), "The copy is located on shelf H-02.")
    assert matched(fact("shelf: H-02"), "It is on H-02.")
    assert not matched(fact("shelf: H-02"), "The copy is located on shelf H-03.")
    assert matched(fact("copies: 3"), "There are 3 copies of the book available.")
    assert not matched(fact("copies: 3"), "There are 3 branches open today.")  # a bare number needs its label
    assert not matched(fact("copies: 3"), "There are 4 copies of the book.")


def test_a_quantity_fact_matches_a_reworded_answer_with_the_same_amount_and_unit() -> None:
    assert matched(fact("capped at 15 dollars per item"), "The maximum fine for an overdue DVD is 15 dollars.")
    assert not matched(fact("capped at 15 dollars per item"), "The maximum fine for an overdue DVD is 5 dollars.")
    assert not matched(fact("capped at 15 dollars per item"), "The library opens at 15 hours.")
    assert matched(fact("a 5% discount on the list price"), "You get 5 percent off the list price.")
    assert not matched(fact("not refundable after 14 days"), "Refundable after 14 days.")


def test_the_example_texts_are_the_documents_of_these_tests() -> None:
    assert "24 working days" in TEXT_A and "8%" in TEXT_B


def test_a_negation_in_another_clause_is_not_a_contradiction_but_one_about_the_claim_is() -> None:
    from forgeai.judge.claims import has_negation, negation_mismatch

    evidence = "The time off must be taken within 60 days of the travel date; it cannot be carried over to the next year."
    assert not negation_mismatch("The time off must be taken within 60 days of the travel date.", evidence)
    assert negation_mismatch("The time off can be carried over to the next year.", evidence)
    assert not negation_mismatch("The time off may not be carried over.", evidence)  # cannot / may not are one negation
    assert negation_mismatch("Staff may not park here.", "Staff may park here.")
    assert has_negation("It isn't allowed") and not has_negation("It is allowed")


def test_a_number_in_the_answer_that_no_evidence_contains_is_found_even_if_the_decomposition_left_it_out() -> None:
    from forgeai.judge.claims import answer_number_checks, claim_numbers

    spans = build_spans(run_of())
    swapped = answer_number_checks("Employees receive 31 working days of annual leave per year.", "How much leave?", spans)
    assert len(swapped) == 1 and "31" in swapped[0].missing and not swapped[0].supported
    assert not answer_number_checks("Employees receive 24 working days of annual leave.", "How much leave?", spans)
    assert not answer_number_checks("For the 7 staff you asked about, it is 24 days.", "How much leave for 7 staff?", spans)  # the user's number
    assert not answer_number_checks("As of 2026-11-01 it is 24 days.", "q", spans)  # dates are checked by the date claim check
    assert claim_numbers("As of 2026-11-01 there are one hundred and 3 items") == [3.0]  # dates and spelled numbers are not figures


def test_a_number_in_a_row_that_names_the_code_of_the_claim_is_supported_whatever_the_column_is_called() -> None:
    from forgeai.ingest.models import Chunk
    from forgeai.judge.models import RetrievedChunk

    row = "part_no: VC-9668 | description: Gland packing set | supplier_id: S03 | moq: 10 | lead_time_days: 42"
    chunk = Chunk(chunk_id="p.csv#c1", doc="p.csv", index=1, text=row, kind="row", char_start=0, char_end=len(row))
    run = run_of("x").model_copy(update={"retrieved": [RetrievedChunk(chunk_id=chunk.chunk_id, doc=chunk.doc, char_start=0, char_end=len(row), text=row)]})
    assert numeric_verdict("The minimum order quantity for part number VC-9668 is 10.", build_spans(run)).verdict == "supported"
    assert numeric_verdict("The minimum order quantity for part number VC-9668 is 12.", build_spans(run)).verdict == "unsupported"


def test_a_table_cell_supports_a_number_when_its_label_is_the_claims_subject_written_out_or_by_initials() -> None:
    from forgeai.judge.claims import label_names_claim

    assert label_names_claim("The minimum order quantity is 10.", "moq: 10")
    assert label_names_claim("The lead time is 42 days.", "lead_time_days: 42")
    assert not label_names_claim("The unit price is 10.", "moq: 10")
    assert not label_names_claim("The minimum order quantity is 10.", "no label here")


def test_whole_numbers_must_match_exactly_but_decimals_may_differ_by_rounding() -> None:
    from forgeai.judge.claims import answer_number_checks, same_number

    assert not same_number(8604, 8597) and same_number(8597, 8597) and same_number(200.1, 200.10) and same_number(19.995, 20.0)
    spans = build_spans(run_of())
    assert answer_number_checks("There are 24 units in stock", "q", spans) == []
    assert answer_number_checks("There are 25 units in stock", "q", spans)  # 25 is not 24


def test_a_number_the_question_gives_is_supported_by_the_question_only_in_the_same_relation() -> None:
    from forgeai.judge.claims import given_by_question

    question = "What is the discounted cost for ordering 15 units of part MT-2474?"
    assert given_by_question("The cost for 15 units of part MT-2474 is shown.", question, 15.0)
    assert not given_by_question("The discounted cost is USD 15.", question, 15.0)  # same digits, another relation
    assert not given_by_question("The minimum order quantity is 2474.", question, 2474.0)  # an identifier's digits are not a quantity
    spans = build_spans(run_of(tools=[ToolCallRecord(tool="calculator", args={}, result="217.5")]))
    echo = numeric_verdict("The order is for 15 units of part MT-2474.", spans, question)
    assert echo.verdict == "supported" and "question gives" in echo.reason
    assert numeric_verdict("The order is for 15 units and costs USD 218.", spans, question).verdict == "unsupported"
    assert numeric_verdict("The discounted cost is USD 15.", spans, question).verdict == "unsupported"


def test_an_identifier_must_occur_whole_in_the_evidence_and_its_digits_are_not_a_value() -> None:
    from forgeai.ingest.models import Chunk
    from forgeai.judge.models import RetrievedChunk

    row = "part_no: MT-8424 | unit_price_usd: 34.57 | moq: 2"
    chunk = Chunk(chunk_id="p.csv#c1", doc="p.csv", index=1, text=row, kind="row", char_start=0, char_end=len(row))
    run = run_of("x").model_copy(update={"retrieved": [RetrievedChunk(chunk_id=chunk.chunk_id, doc=chunk.doc, char_start=0, char_end=len(row), text=row)]})
    spans = build_spans(run)
    assert numeric_verdict("The unit price of MT-8424 is 34.57.", spans, "price of MT-8424?").verdict == "supported"
    swapped = numeric_verdict("The unit price of MT-8424 is 8424.", spans, "price of MT-8424?")
    assert swapped.verdict == "unsupported" and "8424" in swapped.missing  # the part number is not a price
    fake = numeric_verdict("The part MT-9999 costs 34.57.", spans, "what does it cost?")
    assert fake.verdict == "unsupported" and "mt-9999" in fake.missing
    assert numeric_verdict("Part MT-8424 is a cartridge.", spans, "q") is None  # an identifier that checks out: the judge decides the rest


def test_a_percentage_is_not_supported_by_the_same_digits_in_another_unit() -> None:
    from forgeai.judge.claims import units_agree

    assert not units_agree("A 15000% discount applies.", "Orders between USD 15,000 and USD 49,999 get 8%.")
    assert units_agree("An 8% discount applies.", "Orders between USD 15,000 and USD 49,999 get 8%.")
    assert units_agree("The fee is USD 15,000.", "Orders between USD 15,000 and USD 49,999 get 8%.")
    assert not units_agree("The fee is USD 8.", "An 8% discount applies.")
    assert units_agree("There are 8 copies.", "An 8% discount applies.")  # no unit on the claim side: not a conflict
