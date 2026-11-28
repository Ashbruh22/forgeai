"""Light, generic normalisation so layer 1 recognises a fact that is worded differently from the key fact.

Plurals, number words and currency are handled elsewhere (``evals.matching``, ``evals.units``); this module adds a
small list of synonym groups for words that mean the same thing in any document ("maximum" and "capped at") and a
content-term coverage measure. It is deliberately small: a synonym added here applies to every domain.
"""

from __future__ import annotations

import re

from forgeai.textmatch import STOPWORDS, content_terms

SYNONYM_GROUPS = (
    ("maximum", "max", "capped", "cap", "ceiling", "limit", "upper", "most"),
    ("minimum", "min", "least", "lowest"),
    ("fee", "charge", "cost", "price", "fare"),
    ("fine", "penalty", "surcharge"),
    ("begin", "start", "commence"),
    ("end", "finish", "expire", "terminate"),
    ("buy", "purchase", "order"),
    ("permit", "allow", "allowed", "permitted", "may"),
    ("require", "must", "need", "needs", "required", "mandatory"),
    ("copy", "copies", "item", "items", "unit", "units"),
)
_CANON: dict[str, str] = {}
for group in SYNONYM_GROUPS:
    for word in group:
        _CANON[word] = group[0]
_WORD = re.compile(r"[a-z]+")


def canonical_terms(text: str) -> set[str]:
    """Content words of ``text`` reduced to a stem, with synonyms mapped onto one representative."""
    words = [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS or w in _CANON]
    canon = {_CANON.get(w, w) for w in words if len(w) >= 3 or w in _CANON}
    return {t for term in canon for t in (content_terms(term) or {term})}


def coverage(wanted: str, text: str) -> float:
    """Share of the content terms of ``wanted`` that ``text`` also has (1.0 when ``wanted`` has none)."""
    need = canonical_terms(wanted)
    return len(need & canonical_terms(text)) / len(need) if need else 1.0
