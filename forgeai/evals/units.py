"""Units of the numbers found in documents, decided by code (never by a model).

A computed eval case is only meaningful if its inputs have compatible units: a quantity times a
price, a percentage taken off a sum of money, a number of calendar days added to a date. Every
fact a template slot is bound to carries a unit inferred here, from a table column's name or from
the words around a number in a sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from forgeai.evals.matching import dates_in

MONEY, QUANTITY, RATE, DAYS, WORKING_DAYS, DATE = "money", "quantity", "rate", "days", "working_days", "date"
UNIT_NAMES = frozenset({MONEY, QUANTITY, RATE, DAYS, WORKING_DAYS, DATE, "months", "years", "weeks", "hours"})

_CURRENCY = r"(?:USD|US\$|EUR|GBP|SGD|\$|€|£)"
QUANTITY_RE = re.compile(
    rf"(?P<pre>{_CURRENCY})?\s*(?P<num>(?<![\w.])\d[\d,]*(?:\.\d+)?)"
    rf"(?:\s*(?P<post>%|percent\b|per\s+cent\b|(?:working|business|calendar)\s+days?\b|days?\b|weeks?\b|months?\b|years?\b|hours?\b|dollars?\b|euros?\b|{_CURRENCY}\b))?",
    re.IGNORECASE)

_COLUMN_WORDS: list[tuple[str, tuple[str, ...]]] = [
    (DATE, ("date", "snapshot", "effective")),
    (RATE, ("pct", "percent", "percentage", "rate", "discount")),
    (DAYS, ("day", "days", "lead")),
    (MONEY, ("price", "cost", "usd", "eur", "amount", "fee", "value", "total", "charge", "budget")),
    (QUANTITY, ("qty", "quantity", "moq", "count", "units", "stock", "number", "minimum")),
]


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str | None
    text: str  # as written in the document, e.g. "USD 5,000", "5%", "18 months"
    start: int
    end: int


def column_unit(column: str, sample: list[str]) -> str | None:
    """Unit of a table column from its name (and, for dates, its cells); None when it cannot be told."""
    words = set(re.findall(r"[a-z]+", column.lower()))
    cells = [c.strip() for c in sample if c.strip()]
    if cells and all(dates_in(c) for c in cells[:5]):
        return DATE
    for unit, hints in _COLUMN_WORDS:
        if unit != DATE and words & set(hints):
            return unit
    return None


def _unit_from_words(pre: str | None, post: str | None) -> str | None:
    pre, post = (pre or "").upper(), (post or "").lower().strip()
    if pre or post.upper() in {"USD", "US$", "EUR", "GBP", "SGD", "$", "€", "£"} or post.startswith(("dollar", "euro")):
        return MONEY
    if post in ("%", "percent") or post.startswith("per"):
        return RATE
    if post.startswith(("working", "business")):
        return WORKING_DAYS
    if post.startswith(("day", "calendar")):
        return DAYS
    for unit in ("week", "month", "year", "hour"):
        if post.startswith(unit):
            return unit + "s"
    return None


def quantities_in(text: str) -> list[Quantity]:
    """Every number in ``text`` with the unit its neighbouring words give it (None for a bare number)."""
    found: list[Quantity] = []
    for match in QUANTITY_RE.finditer(text):
        raw = match.group("num")
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        found.append(Quantity(value, _unit_from_words(match.group("pre"), match.group("post")),
                              match.group(0).strip(), match.start(), match.end()))
    return found


def compatible(unit: str | None, allowed: tuple[str, ...]) -> bool:
    """A bound fact fits a slot only if its unit is KNOWN and allowed (an unknown unit is not guessed)."""
    return unit is not None and unit in allowed
