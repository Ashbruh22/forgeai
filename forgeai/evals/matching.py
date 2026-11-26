"""Deterministic matching of key facts against text (spec §6.2 layer 1, used for anchoring).

A key fact is anchored if it is found in the seed span. Numbers, dates and currency are
normalised and compared exactly; phrases are compared by normalised substring or, for the
judge's recall check, by rapidfuzz token-set similarity.
"""

from __future__ import annotations

import re
from datetime import date

import dateparser
from rapidfuzz import fuzz
from word2number import w2n

from forgeai.textmatch import STOPWORDS, normalize

NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?|(?<![\w.])\.\d+")
_NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand"
).split()
_WORD_RUN = re.compile(r"\b(?:(?:%s)(?:[\s-]+(?:and\s+)?)?)+\b" % "|".join(_NUMBER_WORDS), re.IGNORECASE)
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_TEXT_DATE = re.compile(
    r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b", re.IGNORECASE)
FUZZY_THRESHOLD = 85
NEGATIONS = frozenset({"not", "never", "without", "cannot", "none", "nor"})  # dropping one flips a fact


def numbers_in(text: str) -> list[float]:
    """Every number in ``text``: digits (commas removed) and spelled-out numbers (``twenty four``)."""
    found = [float(m.replace(",", "")) for m in NUMBER.findall(text) if m.replace(",", "").replace(".", "", 1).isdigit()]
    for match in _WORD_RUN.finditer(text):
        try:
            found.append(float(w2n.word_to_num(match.group(0).strip().lower().replace("-", " "))))
        except ValueError:
            continue
    return found


def dates_in(text: str) -> list[date]:
    """Dates written as ISO (2026-11-01) or in words (1 November 2026, Nov 1, 2026)."""
    found: list[date] = []
    for y, m, d in _ISO_DATE.findall(text):
        try:
            found.append(date(int(y), int(m), int(d)))
        except ValueError:
            continue
    for match in _TEXT_DATE.finditer(text):
        parsed = dateparser.parse(match.group(0))
        if parsed:
            found.append(parsed.date())
    return found


def phrase_in_text(phrase: str, text: str, strict: bool = True) -> bool:
    """Phrase match. Strict (anchoring): normalised substring, or every content word present and
    a fuzzy score of at least 85. Non-strict (judge recall): token-set similarity of at least 85."""
    needle, haystack = normalize(phrase).lower(), normalize(text).lower()
    if not needle:
        return False
    if needle in haystack:
        return True
    score = fuzz.token_set_ratio(needle, haystack)
    if not strict:
        return score >= FUZZY_THRESHOLD
    words = [w for w in re.findall(r"[a-z0-9]+", needle) if (w not in STOPWORDS or w in NEGATIONS) and len(w) > 2]
    return score >= FUZZY_THRESHOLD and bool(words) and all(w in haystack for w in words)


def fact_in_text(kind: str, fact: str, text: str, strict: bool = True) -> bool:
    """Is the key fact ``fact`` (of ``kind``) expressed in ``text``?"""
    if kind == "number":
        targets = numbers_in(fact)
        return bool(targets) and any(abs(n - targets[0]) < 1e-9 for n in numbers_in(text))
    if kind in ("date", "snapshot_date"):
        wanted = dates_in(fact)
        return bool(wanted) and wanted[0] in dates_in(text)
    return phrase_in_text(fact, text, strict)


def ngram_overlap(question: str, source: str, n: int = 3) -> float:
    """Share of the question's word trigrams that also occur in ``source`` (0 = fully paraphrased).

    Trigrams, not bigrams: a short question that merely reuses the passage's vocabulary shares many bigrams
    ("warranty period", "new parts") without copying it, while a copied question shares nearly all trigrams.
    """
    def grams(text: str) -> list[tuple[str, ...]]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return [tuple(words[i : i + n]) for i in range(max(len(words) - n + 1, 0))]

    wanted = grams(question)
    if not wanted:
        return 0.0
    available = set(grams(source))
    return sum(1 for g in wanted if g in available) / len(wanted)
