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
