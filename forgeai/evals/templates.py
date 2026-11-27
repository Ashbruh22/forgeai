"""Computed eval cases from a small library of templates (P5.1 correction 2).

The model never writes a formula. It picks one template and binds each slot of that template to a
fact from a MENU that code built from the documents. Every menu fact is anchored: a table cell
(its row is a verified span) or a number inside a sentence that is a verbatim part of a document.
Code then checks that

* every slot is bound exactly once, to a fact that exists in the menu;
* the units fit the slot (a price slot takes money, a rate slot a percentage, ...);
* a discount tier really contains the order value (bounds are read from the same sentence);
* a table row and a prose rule belong together (a value of the row is named in that document);
* the question names every value needed to find the row, and does not reveal the answer.

The expected answer is then computed by the safe calculator. Nothing here knows what the numbers
mean: "price x quantity x (1 - discount)" is one template among six.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from pydantic import BaseModel, Field

from forgeai.evals.corpus import EvalCorpus, Table
from forgeai.evals.matching import dates_in, numbers_in
from forgeai.evals.units import (
    DATE, DAYS, MONEY, QUANTITY, RATE, Quantity, column_unit, compatible, quantities_in,
)
from forgeai.ingest.spans import SeedSpan
from forgeai.intake.structural import split_sentences
from forgeai.textmatch import normalize
from forgeai.tools.base import ToolError
from forgeai.tools.calculator import evaluate, format_number

OPEN_ENDED = re.compile(r"\b(or more|and above|and over|at least|exceeding|more than|above|over)\b", re.IGNORECASE)
MAX_PROSE_SENTENCES = 8
CLAUSE_NUMBER = re.compile(r"^\d+(?:\.\d+)*\.?\s+")
TODAY = "TODAY"  # menu id of the current date, for "if I order today, when does it arrive?"


class PlanRejected(Exception):
    """A template instance failed a deterministic check; the message is the rejection reason."""


# ---- what the model returns --------------------------------------------------------------------------------


class Binding(BaseModel):
    slot: str = ""
    fact_id: str = ""  # a menu fact such as F3 or S2.1
    question_number: str = ""  # alternatively, a quantity written in the question itself


class TemplateChoice(BaseModel):
    # Field order matters for constrained decoding: decide the template and the facts, then write the question.
    template: str = ""
    bindings: list[Binding] = Field(default_factory=list)
    group_column: str = ""  # extreme only: the column whose value the rows share (e.g. a part number)
    label_column: str = ""  # extreme only: the column that names each row (e.g. a vendor code)
    direction: str = ""  # extreme only: "lowest" or "highest"
    question: str = ""


# ---- the menu: anchored facts ---------------------------------------------------------------------------------


@dataclass
class MenuFact:
    id: str
    value: float
    unit: str | None
    text: str  # as written: "420.00", "USD 5,000", "5%"
    source: str  # csv | prose | today
    doc: str
    span: SeedSpan | None
    description: str
    table: str = ""
    column: str = ""
    row_index: int | None = None
    bounds: tuple[float, float] | None = None  # a rate's tier, read from its own sentence
    date_value: date | None = None


@dataclass
class Menu:
    facts: dict[str, MenuFact] = field(default_factory=dict)
    row_table: Table | None = None
    row_index: int | None = None
    row_identity: str = ""
    text: str = ""

    def add(self, fact: MenuFact) -> None:
        self.facts[fact.id] = fact


def _tier_bounds(sentence: str, rate: Quantity, quantities: list[Quantity]) -> tuple[float, float] | None:
    """The order-value range a rate applies to, from the money amounts in the same sentence."""
    money = [q for q in quantities if q.unit == MONEY and q is not rate]
    values = [q.value for q in money]
    if len(values) >= 2:
        return min(values[:2]), max(values[:2])
    if len(values) == 1 and OPEN_ENDED.search(sentence):
        return values[0], math.inf
    return None


def row_link_values(table: Table, row: dict[str, str], corpus: EvalCorpus) -> dict[str, list[str]]:
    """Prose documents that a table row belongs to: a text cell of the row occurs in exactly that document.

    A vendor code or a name that appears in one prose document and in no other
    ties the row to that document. Without such a value, a price and a discount rule cannot be
    combined safely and no cross-source case is made.
    """
    prose = [name for name in corpus.docs if name not in corpus.tables]
    links: dict[str, list[str]] = {}
    for value in (v.strip() for v in row.values()):
        if len(value) < 2 or re.fullmatch(r"[\d.,\- :/]+", value):
            continue
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])", re.IGNORECASE)
        holding = [name for name in prose if pattern.search(corpus.docs[name].canonical_text)]
        if len(holding) == 1:
            links.setdefault(holding[0], []).append(value)
    return links


def prose_menu_facts(corpus: EvalCorpus, docs: list[str], rng: random.Random, limit: int) -> list[tuple[str, SeedSpan, list[Quantity]]]:
    """Sentences with numbers from ``docs`` as ``(sentence, span, quantities)``; rate sentences first."""
    found: list[tuple[str, SeedSpan, list[Quantity]]] = []
    for name in docs:
        for raw in split_sentences(corpus.docs[name].canonical_text):
            sentence = CLAUSE_NUMBER.sub("", raw)  # "2.1 The discount ..." : the clause number is not a quantity
            quantities = [q for q in quantities_in(sentence) if q.unit is not None]  # a bare number has no unit to check
            if not quantities or len(sentence.split()) < 6 or re.search(r"\bPage \d+\b", sentence):
                continue
            span = corpus.span_of(name, sentence)
            if span is not None:
                found.append((sentence, span, quantities))
    rng.shuffle(found)
    found.sort(key=lambda item: -sum(q.unit == RATE for q in item[2]))
    return found[:limit]


def build_menu(corpus: EvalCorpus, rng: random.Random, with_row: bool = True, today: bool = True,
               row_at: tuple[str, int] | None = None) -> Menu:
    """One context for the model to pick from: a table row, the prose rules tied to it, and today's date."""
    menu = Menu()
    n = 0
    docs = [name for name in corpus.docs if name not in corpus.tables]
    link_docs = docs
    if with_row and corpus.tables:
        table = corpus.tables[row_at[0]] if row_at else rng.choice(list(corpus.tables.values()))
        index = row_at[1] if row_at else _pick_row(table, rng)
        row = table.rows[index]
        menu.row_table, menu.row_index = table, index
        key_cols = _identifying(table)
        menu.row_identity = ", ".join(f"{c} = {row[c]}" for c in key_cols)
        for column in table.columns:
            cell = row[column].strip()
            unit = column_unit(column, [r[column] for r in table.rows[:10]])
            if unit == DATE and dates_in(cell):
                n += 1
                menu.add(MenuFact(f"F{n}", float(dates_in(cell)[0].toordinal()), DATE, cell, "csv", table.doc,
                                  table.row_spans[index], f"{column} = {cell} (a date)", table.doc, column, index,
                                  date_value=dates_in(cell)[0]))
            elif unit and numbers_in(cell) and re.fullmatch(r"-?[\d,]*\.?\d+", cell):
                n += 1
                menu.add(MenuFact(f"F{n}", float(cell.replace(",", "")), unit, cell, "csv", table.doc,
                                  table.row_spans[index], f"{column} = {cell} ({unit})", table.doc, column, index))
        linked = row_link_values(table, row, corpus)
        link_docs = [d for d in docs if d in linked]
    sentences = prose_menu_facts(corpus, link_docs, rng, MAX_PROSE_SENTENCES) if link_docs else []
    for s, (sentence, span, quantities) in enumerate(sentences, 1):
        for q_i, q in enumerate(quantities, 1):
            bounds = _tier_bounds(sentence, q, quantities) if q.unit == RATE else None
            menu.add(MenuFact(f"S{s}.{q_i}", q.value, q.unit, q.text, "prose", span.doc, span,
                              f"{q.text} ({q.unit or 'no unit'}) in sentence S{s}", bounds=bounds))
    if today:
        menu.add(MenuFact(TODAY, 0.0, DATE, "today", "today", "", None, "today's date (from the current_date tool)"))
    menu.text = render_menu(menu, sentences)
    return menu


def _pick_row(table: Table, rng: random.Random) -> int:
    """A random row from the upper half of the table's first money column, so a tier holds a sensible number of units."""
    money = next((c for c in table.columns if column_unit(c, [r[c] for r in table.rows[:10]]) == MONEY), None)
    if money is not None:
        try:
            values = sorted(float(r[money].replace(",", "")) for r in table.rows)
            median = values[len(values) // 2]
            rich = [i for i, r in enumerate(table.rows) if float(r[money].replace(",", "")) >= median]
            if rich:
                return rng.choice(rich)
        except ValueError:
            pass
    return rng.randrange(len(table.rows))


def _identifying(table: Table) -> list[str]:
    from forgeai.evals.generators import Generator

    return Generator.identifying_columns(table)


def render_menu(menu: Menu, sentences: list[tuple[str, SeedSpan, list[Quantity]]]) -> str:
    lines: list[str] = []
    row_facts = [f for f in menu.facts.values() if f.source == "csv"]
    price = next((f for f in row_facts if f.unit == MONEY), None)
    if row_facts and menu.row_table is not None:
        lines.append(f"TABLE ROW ({menu.row_table.doc}): {menu.row_identity}")
        lines += [f"  [{f.id}] {f.description}" for f in row_facts]
        lines.append(f"  columns that name rows: {', '.join(menu.row_table.columns)}")
    for s, (sentence, _, quantities) in enumerate(sentences, 1):
        lines.append(f"SENTENCE S{s}: \"{sentence}\"")
        for q_i, q in enumerate(quantities, 1):
            fact = menu.facts[f"S{s}.{q_i}"]
            hint = ""
            if fact.bounds is not None and price is not None and price.value > 0:
                low, high = fact.bounds
                first = max(1, math.ceil(low / price.value))
                last = first + 25 if math.isinf(high) else math.floor(high / price.value)
                if first <= last:
                    hint = f"; with the row's price, {first} to {min(last, first + 40)} units fall in this tier"
            tier = "" if fact.bounds is None else f" (tier: order value {fact.bounds[0]:g} to {fact.bounds[1]:g}{hint})"
            lines.append(f"  [{fact.id}] {fact.description}{tier}")
    if TODAY in menu.facts:
        lines.append(f"[{TODAY}] {menu.facts[TODAY].description}")
    return "\n".join(lines)


# ---- templates -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    slots: dict[str, tuple[str, ...]]  # slot -> allowed units
    formula: str
    result_unit: str
    description: str
    same_unit: bool = False  # all slots must share one unit (sum, difference)
    capability: str = "arithmetic"


TEMPLATES: dict[str, TemplateSpec] = {t.name: t for t in (
    TemplateSpec("product", {"quantity": (QUANTITY,), "price": (MONEY,)}, "quantity * price", MONEY,
                 "quantity x price"),
    TemplateSpec("discounted_product", {"quantity": (QUANTITY,), "price": (MONEY,), "rate": (RATE,)},
                 "quantity * price * (1 - rate / 100)", MONEY,
                 "quantity x price x (1 - discount rate); the discount tier must contain quantity x price"),
    TemplateSpec("sum", {"a": (MONEY, DAYS, QUANTITY), "b": (MONEY, DAYS, QUANTITY)}, "a + b", "", "a + b",
                 same_unit=True),
    TemplateSpec("difference", {"a": (MONEY, DAYS, QUANTITY), "b": (MONEY, DAYS, QUANTITY)}, "a - b", "", "a - b",
                 same_unit=True),
    TemplateSpec("date_plus_days", {"start": (DATE,), "days": (DAYS,)}, "", DATE,
                 "a date plus a number of calendar days (start may be TODAY)", capability="temporal"),
    TemplateSpec("extreme", {"value": (MONEY, DAYS, QUANTITY)}, "", "",
                 "which row has the lowest or highest value among all rows sharing one column value",
                 capability="structured_lookup"),
)}


@dataclass
class Computed:
    template: str
    question: str
    answer_text: str
    answer_kind: str  # number | date | phrase | relative_date
    spans: list[SeedSpan]
    docs: list[str]
    tools: list[str]
    formula: str
    values: dict[str, float | str]
    cross_source: bool
    needs_current_date: bool = False
    days_offset: int | None = None


def _bound_fact(menu: Menu, binding: Binding, question: str, allowed: tuple[str, ...]) -> MenuFact:
    if binding.question_number.strip():
        if QUANTITY not in allowed:
            raise PlanRejected(f"slot {binding.slot} cannot take a number from the question")
        found = numbers_in(binding.question_number)
        if not found or not any(abs(found[0] - n) < 1e-9 for n in numbers_in(question)):
            raise PlanRejected(f"{binding.question_number!r} is not a number written in the question")
        return MenuFact("Q", found[0], QUANTITY, binding.question_number, "question", "", None, "number in the question")
    fact = menu.facts.get(binding.fact_id.strip().strip("[]"))
    if fact is None:
        raise PlanRejected(f"slot {binding.slot}: {binding.fact_id!r} is not a fact in the menu")
    if not compatible(fact.unit, allowed):
        raise PlanRejected(f"slot {binding.slot} takes {'/'.join(allowed)} but {fact.id} is {fact.unit or 'of unknown unit'}")
    return fact


def _row_named_in(question: str, table: Table, row: dict[str, str]) -> bool:
    return all(row[c].strip().lower() in question.lower() for c in _identifying(table))


def instantiate(choice: TemplateChoice, menu: Menu, corpus: EvalCorpus) -> Computed:
    """Verify one model choice against the menu and compute the answer. Raises ``PlanRejected``."""
    spec = TEMPLATES.get(choice.template.strip())
    question = choice.question.strip()
    if spec is None:
        raise PlanRejected(f"unknown template {choice.template!r}")
    if not question:
        raise PlanRejected("no question")
    if spec.name == "extreme":
        return _extreme(choice, spec, menu, corpus, question)
    by_slot: dict[str, Binding] = {}
    for binding in choice.bindings:
        if binding.slot in by_slot or binding.slot not in spec.slots:
            raise PlanRejected(f"slot {binding.slot!r} is repeated or not a slot of {spec.name}")
        by_slot[binding.slot] = binding
    if missing := [s for s in spec.slots if s not in by_slot]:
        raise PlanRejected(f"unbound slots {missing}")
    facts = {slot: _bound_fact(menu, by_slot[slot], question, allowed) for slot, allowed in spec.slots.items()}
    if spec.same_unit and len({f.unit for f in facts.values()}) != 1:
        raise PlanRejected(f"{spec.name} needs one unit, got {sorted(str(f.unit) for f in facts.values())}")
    spans = [f.span for f in facts.values() if f.span is not None]
    docs = [f.doc for f in facts.values() if f.doc]
    anchored = [f.id for f in facts.values() if f.source != "question"]
    if not anchored:
        raise PlanRejected("a computed case needs at least one fact taken from the documents")
    if len(set(anchored)) < len(anchored):
        raise PlanRejected("two slots are bound to the same fact")
    row_used = any(f.source == "csv" for f in facts.values())
    if row_used and menu.row_table is not None and menu.row_index is not None:
        if not _row_named_in(question, menu.row_table, menu.row_table.rows[menu.row_index]):
            raise PlanRejected("the question does not name the values that identify the table row")
    values: dict[str, float | str] = {s: f.value for s, f in facts.items()}
    tools = ["calculator"] if spec.capability == "arithmetic" else []
    if row_used:
        tools.append("csv_lookup")
    cross = row_used and any(f.source == "prose" for f in facts.values())
    if spec.name == "discounted_product":
        rate = facts["rate"]
        total = facts["quantity"].value * facts["price"].value
        if rate.bounds is None:
            raise PlanRejected("the rate's sentence does not state an order-value range, so its tier cannot be checked")
        low, high = rate.bounds
        if not (total >= low and (math.isinf(high) or total < high + 1)):
            raise PlanRejected(f"order value {total:g} is outside the tier {low:g}-{high:g} of that rate")
        if rate.doc and facts["price"].source == "csv" and menu.row_table is not None and menu.row_index is not None:
            if rate.doc not in row_link_values(menu.row_table, menu.row_table.rows[menu.row_index], corpus):
                raise PlanRejected(f"{rate.doc} is not the document this row belongs to")
        if not cross:
            raise PlanRejected("a discount case must combine a table price with a rule from the documents")
    if spec.name == "date_plus_days":
        return _date_plus_days(facts, spec, question, spans, docs, tools, cross, menu)
    try:
        answer = round(float(evaluate(spec.formula, {k: float(v) for k, v in values.items()})), 2)
    except ToolError as exc:
        raise PlanRejected(f"cannot compute: {exc}") from exc
    if answer < 0:
        raise PlanRejected("the answer is negative")
    text = format_number(answer)
    if any(abs(answer - n) < 1e-9 for n in numbers_in(question)):
        raise PlanRejected("the question already contains the answer")
    return Computed(spec.name, question, text, "number", spans, list(dict.fromkeys(docs)), tools,
                    spec.formula, values, cross)


def _date_plus_days(facts: dict[str, MenuFact], spec: TemplateSpec, question: str, spans: list[SeedSpan],
                    docs: list[str], tools: list[str], cross: bool, menu: Menu) -> Computed:
    start, days = facts["start"], facts["days"]
    offset = int(days.value)
    if abs(days.value - offset) > 1e-9 or not 0 < offset <= 730:
        raise PlanRejected("the number of days must be a whole number between 1 and 730")
    if start.source == "today":
        if not re.search(r"\b(today|now|current date|this moment)\b", question, re.IGNORECASE):
            raise PlanRejected("a case that starts today must say so in the question")
        return Computed("date_plus_days", question, f"today + {offset} days", "relative_date", spans,
                        list(dict.fromkeys(docs)), ["current_date", *tools], "today + days",
                        {"days": float(offset)}, cross, needs_current_date=True, days_offset=offset)
    assert start.date_value is not None
    answer = start.date_value + timedelta(days=offset)
    if answer.isoformat() in question or any(d == answer for d in dates_in(question)):
        raise PlanRejected("the question already contains the answer")
    if not re.search(re.escape(start.text), question) and not any(d == start.date_value for d in dates_in(question)):
        raise PlanRejected("the question must state the start date, because the agent cannot know which date was meant")
    return Computed("date_plus_days", question, answer.isoformat(), "date", spans, list(dict.fromkeys(docs)),
                    [*tools], "start + days", {"days": float(offset)}, cross, days_offset=offset)


def _extreme(choice: TemplateChoice, spec: TemplateSpec, menu: Menu, corpus: EvalCorpus, question: str) -> Computed:
    table, index = menu.row_table, menu.row_index
    if table is None or index is None:
        raise PlanRejected("extreme needs a table row")
    group, label = choice.group_column.strip(), choice.label_column.strip()
    direction = choice.direction.strip().lower()
    binding = next((b for b in choice.bindings if b.slot == "value"), None)
    if group not in table.columns or label not in table.columns or group == label or direction not in ("lowest", "highest"):
        raise PlanRejected("extreme needs a group column, a label column and a direction of lowest or highest")
    if binding is None:
        raise PlanRejected("unbound slots ['value']")
    fact = _bound_fact(menu, binding, question, spec.slots["value"])
    if fact.source != "csv" or fact.column in (group, label):
        raise PlanRejected("the value must be a numeric table column other than the group and label columns")
    row = table.rows[index]
    rows = [i for i, r in enumerate(table.rows) if r[group].strip() == row[group].strip()]
    labels = [table.rows[i][label].strip() for i in rows]
    if not 2 <= len(rows) <= 6 or len(set(labels)) != len(labels):
        raise PlanRejected(f"{len(rows)} rows share {group} = {row[group]!r} with labels {labels}; need 2 to 6 distinct")
    try:
        numbers = [float(table.rows[i][fact.column].replace(",", "")) for i in rows]
    except ValueError as exc:
        raise PlanRejected("a compared cell is not a number") from exc
    best = min(numbers) if direction == "lowest" else max(numbers)
    if numbers.count(best) != 1:
        raise PlanRejected("the extreme is tied")
    winner = labels[numbers.index(best)]
    lowered = question.lower()
    if row[group].strip().lower() not in lowered:
        raise PlanRejected(f"the question must name {group} = {row[group]!r}, the rows being compared")
    mentioned = [lab for lab in labels if lab.lower() in lowered]
    if winner.lower() in lowered and len(mentioned) < len(labels):
        raise PlanRejected("the question names the winner but not the other rows")
    return Computed("extreme", question, winner, "phrase", [table.row_spans[i] for i in rows], [table.doc],
                    ["csv_lookup"], f"{direction} {fact.column} by {label}",
                    {label: winner, fact.column: best}, False)


def compute_prompt_variables(menu: Menu, ask: str) -> dict[str, str]:
    return {"menu": menu.text, "ask": ask or "(no specific focus)"}


def template_help(names: list[str]) -> str:
    return "\n".join(f"- {n}: {TEMPLATES[n].description}; slots: {', '.join(TEMPLATES[n].slots)}" for n in names)


def resolve_relative_fact(text: str, today: date) -> str:
    """The value of a relative key fact such as ``today + 18 days`` on ``today`` (for P6's scorer)."""
    match = re.fullmatch(r"today \+ (\d+) days", normalize(text))
    if not match:
        raise ValueError(f"not a relative date fact: {text!r}")
    return (today + timedelta(days=int(match.group(1)))).isoformat()
