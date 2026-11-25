"""Find the date a static data snapshot was taken, from the evidence itself (no LLM).

A document or table that stands in for live data is only honest if the answer says "as of
<date>". The date is read from the evidence chunks: a ``snapshot_date`` / ``as_of`` /
``updated`` field in a table row, or an "as of <date>" phrase in prose.
"""

from __future__ import annotations

import re
from datetime import date

import dateparser

from forgeai.ingest.models import Chunk

_FIELD = re.compile(r"\b(?:snapshot_date|snapshot|as_of|asof|last_updated|updated_on|updated)\s*[:=]\s*([^|\n]+)", re.IGNORECASE)
_PHRASE = re.compile(r"\b(?:as of|snapshot taken on|data as at)\s+([0-9A-Za-z ,/.-]{6,24})", re.IGNORECASE)


def _parse(text: str) -> date | None:
    """Parse a date at the start of ``text``, trying the longest word prefix first."""
    words = text.strip().split()
    for count in range(min(4, len(words)), 0, -1):
        parsed = dateparser.parse(" ".join(words[:count]).rstrip(".,;"), settings={"PREFER_DAY_OF_MONTH": "first"})
        if parsed:
            return parsed.date()
    return None


def find_snapshot_date(chunks: list[Chunk]) -> str | None:
    """Latest snapshot date found in ``chunks`` as an ISO string, or None."""
    found: list[date] = []
    for chunk in chunks:
        for pattern in (_FIELD, _PHRASE):
            for match in pattern.finditer(chunk.text):
                parsed = _parse(match.group(1))
                if parsed:
                    found.append(parsed)
    return max(found).isoformat() if found else None
