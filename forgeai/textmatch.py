"""Deterministic text matching shared by the runner, guard, eval validators and judge.

Spec §0 "verify, don't trust": a quote is accepted only if it is a verbatim substring
of the source text after whitespace normalisation. No fuzzy matching here.
"""

from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    """NFKC-normalise and collapse all whitespace runs to single spaces."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def is_verbatim(quote: str, source: str) -> bool:
    """True if ``quote`` appears in ``source`` verbatim (whitespace-normalised, case-sensitive)."""
    needle = normalize(quote)
    return bool(needle) and needle in normalize(source)


STOPWORDS = frozenset(
    "about above after again against all also and any are because been before being below between "
    "both but can could did does doing down during each few for from further had has have having "
    "her here hers him his how into its itself just more most must not now off once only other our "
    "out over own same should some such than that the their them then there these they this those "
    "through too under until very was were what when where which while who whom why will with would "
    "you your help question questions policy policies rules rule information general".split()
)


def content_terms(text: str) -> set[str]:
    """Lowercase content words (4+ letters, not stopwords), reduced to a 6-letter stem.

    Six letters keeps "entitled"/"entitlements" together but not "interpretation"/"internal".
    """
    return {w[:6] for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOPWORDS}


def shares_content_term(a: str, b: str) -> bool:
    """True if the two texts have at least one content-term stem in common."""
    return bool(content_terms(a) & content_terms(b))
