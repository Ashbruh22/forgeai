"""The specific asks of a brief, separated from its generic framing.

A brief opens with a role sentence ("I need an assistant that answers HR policy questions for
our field staff") and then lists what is wanted. The role sentence overlaps almost every
refusal a requester could imagine ("personal data" is about HR), so it must not be matched
against refusals. ``specific_asks`` returns only the things the requester actually asked for,
each reduced to the thing itself ("finding parts", "leave", "travel claims").

Two consumers: the refusal-conflict check in intake, and the coverage matrix in evals (every
distinct ask needs at least one eval case).
"""

from __future__ import annotations

import re

from forgeai.intake.structural import split_sentences
from forgeai.textmatch import content_terms

# A clause that names the assistant itself is the generic purpose sentence, not an ask.
ROLE_NOUN = re.compile(r"\b(assistant|agent|chatbot|bot|copilot|tool|system)\b", re.IGNORECASE)
# Everything before the thing wanted: "Our superintendents need help finding ...", "It should also tell them ...".
_LEAD_IN = re.compile(
    r"^(?:also\s+)?(?:"
    r"(?:(?:i|we)\s+(?:need|want|would like)(?:\s+(?:help|you|it)\s+(?:with|to))?)"
    r"|(?:(?:it|they|this)\s+(?:should|must|needs?\s+to|has\s+to|will)(?:\s+also)?)"
    r"|(?:[\w' ]{1,60}?\b(?:needs?|wants?)\s+(?:help\s+)?(?:with\s+)?)"
    r"|(?:(?:help|assist)\s+(?:\w+\s+){0,2}?with)"
    r"|(?:(?:help|assist)\s+(?!\w+ing\b)\w+)"
    r")\s*",
    re.IGNORECASE,
)
_LIST_SPLIT = re.compile(r",\s+(?:and|or)\s+|\s+and\s+(?=\w+ing\b)|;\s+|,\s+|\s+and also\s+", re.IGNORECASE)


def _strip(text: str) -> str:
    text = text.strip().rstrip(".!?").strip()
    previous = None
    while previous != text:  # a lead-in can be preceded by "Also"
        previous = text
        text = _LEAD_IN.sub("", text, count=1).strip()
    return text


def specific_asks(brief: str) -> list[str]:
    """What the brief asks for, one item per ask, without its generic role sentence."""
    asks: list[str] = []
    for sentence in split_sentences(brief):
        head, colon, tail = sentence.partition(":")
        # "<role sentence>: <list of what is wanted>" - the part before the colon is the framing.
        pieces = _LIST_SPLIT.split(tail if colon and ROLE_NOUN.search(head) else sentence)
        for piece in pieces:
            ask = _strip(piece)
            if ask and content_terms(ask) and not ROLE_NOUN.search(ask):
                asks.append(ask)
    return list(dict.fromkeys(asks))


ACTION_WORDS = frozenset(
    "tell find show explain answer estimate calculate compare check help list give look search identify advise "
    "track book order plan report submit book summarise summarize recommend".split())


def is_task(ask: str) -> bool:
    """A task ("estimating order cost with discounts") rather than a bare category ("travel claims").

    A task starts with an action: a gerund ("estimating") or an imperative ("tell", "find"). A category
    label stands for a whole area, so a judge that finds almost anything to be "part of" it is unreliable
    (it ruled "personal data requests" part of "travel claims"). Only a described activity is specific
    enough for the judge to say an input is needed for it.
    """
    words = re.findall(r"[a-z']+", ask.lower())
    if len(content_terms(ask)) < 2 or not words:
        return False
    return (words[0].endswith("ing") and len(words[0]) > 4) or words[0] in ACTION_WORDS
