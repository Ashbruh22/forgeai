"""Structural gap detection (spec §4.2): rules over the brief and documents, no LLM.

These detect "the brief asks for something no document or tool can provide": live data,
actions no tool performs, and stale documents. The cue lists are generic English; nothing
here knows about any particular domain.
"""

from __future__ import annotations

import re
from datetime import date

import dateparser

from forgeai.intake.schema import Gap
from forgeai.tools import REGISTRY

# Cues that the requester wants current/real-time values.
LIVE_CUES = re.compile(
    r"\b(right now|real[- ]?time|live (?:data|status|feed|tracking|position|prices?|stock|levels?)"
    r"|at the moment|as of now|up[- ]to[- ]the[- ]minute|currently (?:in stock|available|on|at|open))\b",
    re.IGNORECASE,
)
# Cues that a live need IS served by an attached-capable tool (weather, today's date).
LIVE_SERVED = {
    "live_weather": re.compile(r"\b(weather|forecast|temperature|wind|rain)\b", re.IGNORECASE),
    "temporal": re.compile(r"\b(today'?s date|current date|what day|days (?:until|since|left))\b", re.IGNORECASE),
}
# Cues that the requester wants the assistant to DO something in the world.
ACTION_CUES = re.compile(
    r"\b(e-?mail (?:the|a|an|them|him|her|me|someone)|send (?:an?|the|them|him|her|me) (?:e-?mail|message|text|notification|invoice)"
    r"|book (?:a|an|the|my)|reserve (?:a|an|the)|raise (?:a|an) (?:ticket|request)|place (?:an?|the) order"
    r"|submit (?:an?|the|my) (?:claim|request|form|order)|approve (?:the|a|an|my|their)|schedule (?:a|an|the)"
    r"|notify (?:the|a|an|them|him|her|me)|(?:text|call) (?:the|a|an|them|him|her|me))\b",
    re.IGNORECASE,
)
# Wording before an action verb that shows the sentence is about information, not a request to act.
INFORMATIONAL = re.compile(r"\b(how (?:to|do|can|does)|explain|describe|what|which|whether|policy|policies|rules? (?:on|for|about))\b", re.IGNORECASE)
DATE_CUES = re.compile(
    r"(?:effective(?: date| from)?|last (?:updated|revised|reviewed)|version date|issued)[:\s]+([0-9A-Za-z ,/.-]{6,30})",
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


_CLAUSE_SPLIT = re.compile(r",\s+(?:and|but)\s+|;\s+|\s+and also\s+", re.IGNORECASE)


def clause_with(sentence: str, pattern: re.Pattern[str]) -> str:
    """The clause of ``sentence`` that contains ``pattern`` (so a second request in the same
    sentence is not mistaken for part of the first)."""
    for clause in _CLAUSE_SPLIT.split(sentence):
        if pattern.search(clause):
            return clause.strip()
    return sentence


def detect_live_data_gaps(brief: str) -> list[Gap]:
    gaps: list[Gap] = []
    for sentence in split_sentences(brief):
        if not LIVE_CUES.search(sentence):
            continue
        served = [cap for cap, cue in LIVE_SERVED.items() if cue.search(sentence)
                  and any(spec.capability == cap for spec in REGISTRY.values())]
        if served:
            continue  # a registered tool provides this kind of live value
        clause = clause_with(sentence, LIVE_CUES)
        gaps.append(Gap(
            id=f"G-LIVE-{len(gaps) + 1}", kind="live_data",
            description=f"The brief asks for current or real-time information, but the documents are a static snapshot and no tool provides it: {clause!r}",
            recommended_action="refuse_with_redirect", evidence=[clause],
        ))
    return gaps


def detect_capability_gaps(brief: str) -> list[Gap]:
    gaps: list[Gap] = []
    for sentence in split_sentences(brief):
        match = ACTION_CUES.search(sentence)
        if match and not INFORMATIONAL.search(sentence[: match.start()]):
            gaps.append(Gap(
                id=f"G-CAP-{len(gaps) + 1}", kind="capability",
                description=f"The brief asks the assistant to perform an action ({match.group(0)!r}), but no registered tool can: {sentence!r}",
                recommended_action="refuse_with_redirect", evidence=[sentence],
            ))
    return gaps


def detect_freshness_gaps(
    documents: dict[str, str], today: date, max_age_days: int = 730
) -> list[Gap]:
    """Warn about documents whose stated effective date is older than ``max_age_days``."""
    gaps: list[Gap] = []
    for name, text in documents.items():
        for match in DATE_CUES.finditer(text[:3000]):
            parsed = dateparser.parse(match.group(1).strip(), settings={"PREFER_DAY_OF_MONTH": "first"})
            if parsed is None:
                continue
            age = (today - parsed.date()).days
            if age > max_age_days:
                gaps.append(Gap(
                    id=f"G-FRESH-{len(gaps) + 1}", kind="freshness",
                    description=f"{name} states a date of {parsed.date().isoformat()}, {age} days ago (limit {max_age_days}).",
                    recommended_action="warn", evidence=[name],
                ))
            break  # first dated line per document is enough
    return gaps


def detect_structural_gaps(
    brief: str, documents: dict[str, str], today: date, max_age_days: int = 730
) -> list[Gap]:
    gaps = detect_live_data_gaps(brief) + detect_capability_gaps(brief) + detect_freshness_gaps(
        documents, today, max_age_days
    )
    return gaps
