"""Computed cases from templates (P5.1 correction 2): the model binds slots to menu facts; code checks and computes."""

from __future__ import annotations

import random

import pytest

from forgeai.evals.generators import Generator
from forgeai.evals.templates import (
    TEMPLATES, Binding, PlanRejected, TemplateChoice, build_menu, instantiate, resolve_relative_fact,
)
from forgeai.evals.units import DAYS, MONEY, QUANTITY, RATE, column_unit, quantities_in
from tests.test_evals import config, corpus, docs, install, requirements  # noqa: F401  (fixtures and helpers)


def menu_for(corpus):
    """A menu whose table row is HD-1 / S01 (the first row, unit price 420.00)."""
    return build_menu(corpus, random.Random(0), row_at=("parts.csv", 0))


def fact_id(menu, **wanted):
    for fact in menu.facts.values():
        if all(getattr(fact, k) == v for k, v in wanted.items()):
            return fact.id
    raise AssertionError(f"no menu fact {wanted}: {list(menu.facts.values())}")


def discounted(menu, question="What do 12 units of HD-1 from S01 cost after the volume discount?", rate_text="5%", **over):
    choice = TemplateChoice(
        template="discounted_product",
        bindings=[Binding(slot="quantity", question_number="12"),
                  Binding(slot="price", fact_id=fact_id(menu, unit=MONEY, source="csv")),
                  Binding(slot="rate", fact_id=fact_id(menu, unit=RATE, text=rate_text))],
        question=question)
    return choice.model_copy(update=over)


def test_units_come_from_column_names_and_the_words_around_a_number() -> None:
    assert column_unit("unit_price_usd", ["420.00"]) == MONEY and column_unit("lead_time_days", ["18"]) == DAYS
    assert column_unit("moq", ["5"]) == QUANTITY and column_unit("snapshot_date", ["2026-11-01"]) == "date"
    assert column_unit("colour", ["red"]) is None  # an unknown unit is never guessed
    found = quantities_in("A 5% discount applies to orders between USD 5,000 and USD 14,999 within 10 working days.")
    assert [(q.text, q.unit) for q in found] == [
        ("5%", RATE), ("USD 5,000", MONEY), ("USD 14,999", MONEY), ("10 working days", "working_days")]


def test_the_menu_holds_only_anchored_facts_with_units_and_tier_bounds(corpus) -> None:
    menu = menu_for(corpus)
    rate = next(f for f in menu.facts.values() if f.unit == RATE and f.text == "5%")
    assert rate.bounds == (5000.0, 14999.0) and rate.span is not None and rate.span.span_text.startswith("Orders from USD 5,000")
    price = next(f for f in menu.facts.values() if f.source == "csv" and f.unit == MONEY)
    assert price.value == 420.0 and price.span.doc == "parts.csv"
    assert "TODAY" in menu.facts


def test_a_discounted_product_is_computed_by_code_with_the_tier_checked(corpus) -> None:
    menu = menu_for(corpus)
    result = instantiate(discounted(menu), menu, corpus)
    assert result.answer_text == "4788"  # 12 x 420 x 0.95
    assert result.cross_source and sorted(result.docs) == ["parts.csv", "terms.md"]
    assert sorted(result.tools) == ["calculator", "csv_lookup"]


@pytest.mark.parametrize(("change", "why"), [
    ({"question": "What do 12 units cost after the discount?"}, "does not name the values"),
    ({"question": "What do 4 units of HD-1 from S01 cost after the discount?"}, "is not a number written in the question"),
    ({"template": "invented"}, "unknown template"),
    ({"bindings": []}, "unbound slots"),
])
def test_bad_choices_are_rejected_with_a_reason(corpus, change, why) -> None:
    menu = menu_for(corpus)
    with pytest.raises(PlanRejected, match=why):
        instantiate(discounted(menu, **change), menu, corpus)


def test_a_rate_whose_tier_does_not_contain_the_order_value_is_rejected(corpus) -> None:
    menu = menu_for(corpus)
    wrong = discounted(menu, question="What do 40 units of HD-1 from S01 cost after the volume discount?",
                       bindings=[Binding(slot="quantity", question_number="40"),
                                 Binding(slot="price", fact_id=fact_id(menu, unit=MONEY, source="csv")),
                                 Binding(slot="rate", fact_id=fact_id(menu, unit=RATE, text="5%"))])
    with pytest.raises(PlanRejected, match="outside the tier"):  # 40 x 420 = 16,800, which is the 8% tier
        instantiate(wrong, menu, corpus)


def test_units_must_fit_the_slot_and_a_fact_cannot_be_invented(corpus) -> None:
    menu = menu_for(corpus)
    swapped = discounted(menu, bindings=[Binding(slot="quantity", question_number="12"),
                                         Binding(slot="price", fact_id=fact_id(menu, unit=RATE, text="5%")),
                                         Binding(slot="rate", fact_id=fact_id(menu, unit=MONEY, source="csv"))])
    with pytest.raises(PlanRejected, match="takes money"):
        instantiate(swapped, menu, corpus)
    invented = discounted(menu, bindings=[Binding(slot="quantity", question_number="12"),
                                          Binding(slot="price", fact_id="F99"),
                                          Binding(slot="rate", fact_id=fact_id(menu, unit=RATE, text="5%"))])
    with pytest.raises(PlanRejected, match="not a fact in the menu"):
        instantiate(invented, menu, corpus)


def test_a_discount_from_another_suppliers_document_is_rejected(docs, config, tmp_path) -> None:
    """The row's own terms must supply the rate: a code that names S02 only in the OTHER document is not a link."""
    (docs / "other_terms.md").write_text(
        "# Other Terms\n\n## Volume Discounts\n\nSupplier S09 terms. Orders from USD 5,000 to USD 14,999 receive a 7% discount on the list price.\n",
        encoding="utf-8")
    from forgeai.evals.corpus import EvalCorpus
    from tests.helpers import FakeEmbedder

    corpus = EvalCorpus.load(docs, config, FakeEmbedder())
    menu = menu_for(corpus)
    assert all(f.doc != "other_terms.md" for f in menu.facts.values())  # not offered: S01 is not named there
    seven = next(f for f in build_menu_for_doc(corpus, "other_terms.md") if f.text == "7%")
    seven.bounds = (5000.0, 14999.0)  # read from its own sentence, as the menu does
    menu.facts["X1"] = seven
    bad = discounted(menu, bindings=[Binding(slot="quantity", question_number="12"),
                                     Binding(slot="price", fact_id=fact_id(menu, unit=MONEY, source="csv")),
                                     Binding(slot="rate", fact_id="X1")])
    with pytest.raises(PlanRejected, match="not the document this row belongs to"):
        instantiate(bad, menu, corpus)


def build_menu_for_doc(corpus, doc):
    from forgeai.evals.templates import prose_menu_facts

    from forgeai.evals.templates import MenuFact
    facts = []
    for sentence, span, quantities in prose_menu_facts(corpus, [doc], random.Random(1), 8):
        for i, q in enumerate(quantities):
            facts.append(MenuFact(f"X{i}", q.value, q.unit, q.text, "prose", doc, span, "x"))
    return facts


def test_date_templates_use_a_stated_date_or_today(corpus) -> None:
    menu = menu_for(corpus)
    days = fact_id(menu, unit="date", source="csv")  # the snapshot date cell
    assert days
    # the table in the fixture has no days column, so build the facts from the prose-free pieces directly
    from forgeai.evals.templates import MenuFact

    menu.facts["D1"] = MenuFact("D1", 18.0, DAYS, "18", "csv", "parts.csv", menu.facts[days].span, "lead = 18 (days)")
    on_date = TemplateChoice(template="date_plus_days", bindings=[Binding(slot="start", fact_id=days), Binding(slot="days", fact_id="D1")],
                             question="Parts HD-1 from S01 are ordered on 2026-11-20; on what date will they arrive?")
    result = instantiate(on_date, menu, corpus)
    assert result.answer_text == "2026-12-08" and result.answer_kind == "date"
    today = on_date.model_copy(update={"bindings": [Binding(slot="start", fact_id="TODAY"), Binding(slot="days", fact_id="D1")],
                                       "question": "If I order HD-1 from S01 today, when will it arrive?"})
    relative = instantiate(today, menu, corpus)
    assert relative.answer_kind == "relative_date" and relative.needs_current_date and "current_date" in relative.tools
    from datetime import date

    assert resolve_relative_fact(relative.answer_text, date(2026, 11, 26)) == "2026-12-14"
    silent = today.model_copy(update={"question": "When will HD-1 from S01 arrive?"})
    with pytest.raises(PlanRejected, match="must say so"):
        instantiate(silent, menu, corpus)


def test_extreme_picks_the_cheapest_among_all_rows_that_share_a_column_value(corpus) -> None:
    menu = menu_for(corpus)
    price = fact_id(menu, unit=MONEY, source="csv")
    choice = TemplateChoice(template="extreme", bindings=[Binding(slot="value", fact_id=price)], group_column="part_no",
                            label_column="supplier", direction="lowest",
                            question="Which supplier has the lowest unit price for HD-1, S01 or S02?")
    result = instantiate(choice, menu, corpus)
    assert result.answer_text == "S01" and result.answer_kind == "phrase"  # 420.00 beats 433.00
    reveals = choice.model_copy(update={"question": "Is S01 the cheapest source of HD-1?"})
    with pytest.raises(PlanRejected, match="names the winner but not the other rows"):
        instantiate(reveals, menu, corpus)


def test_generator_turns_a_valid_choice_into_a_case_and_records_the_rest(corpus, config, tmp_path) -> None:
    def pick(user: str) -> dict:
        import re

        identity = re.search(r"part_no = (\S+), supplier = (\S+)", user)
        price = re.search(r"\[(F\d+)\] unit_price_usd", user).group(1)
        rate = re.search(r"\[(S\d+\.\d+)\] 5% \(rate\)", user)
        if rate is None or identity is None:
            return {"template": "sum", "bindings": [], "question": "unusable"}
        part, supplier = identity.group(1), identity.group(2)
        return {"template": "discounted_product", "question": f"What do 12 units of {part} from {supplier} cost after the volume discount?",
                "bindings": [{"slot": "quantity", "question_number": "12"}, {"slot": "price", "fact_id": price},
                             {"slot": "rate", "fact_id": rate.group(1)}]}

    install(config, tmp_path, TemplateChoice=pick)
    gen = Generator(requirements(capabilities_needed=["document_qa", "arithmetic"]), corpus, config)
    gen.computed("arithmetic", "estimating order cost with discounts", ["ask:A3/arithmetic"], 6)
    cases = [c.case for c in gen.out.candidates]
    assert cases, gen.out.rejected[:3]
    values = cases[0].provenance["values"]
    expected = round(values["quantity"] * values["price"] * (1 - values["rate"] / 100), 2)
    assert float(cases[0].key_facts[0].text) == pytest.approx(expected) and cases[0].covers == ["ask:A3/arithmetic"]
    assert cases[0].origin == "computed" and cases[0].validation["expected_value_computed_by_code"] is True
    assert {r.reason for r in gen.out.rejected} <= {"computed case failed verification", "template is not one of the offered templates"}


def test_every_template_names_its_slots_and_a_formula_or_a_special_computation() -> None:
    assert set(TEMPLATES) == {"product", "discounted_product", "sum", "difference", "date_plus_days", "extreme"}
    for spec in TEMPLATES.values():
        assert spec.slots and (spec.formula or spec.name in ("date_plus_days", "extreme"))
