"""Consistency of the requirements: nothing is both asked for and refused (P5 blocker fix).

Eval cases are built from the requirements, so a refusal that contradicts an ask would
produce contradictory evals. Self-answered exclusions are therefore checked, in code, against:

1. the topics and clauses the brief asks for (lexical overlap, then embedding similarity plus a
   judge-model yes/no whose supporting quote is verified);
2. the corpus, including TABLE SCHEMAS: column names and cell values count as evidence
   (a column ``unit_price_usd`` is evidence that "pricing" is covered);
3. (in ``orchestrate``) probes against the documents.

``assert_consistent`` is the invariant that holds for every requirements file.
"""

from __future__ import annotations

import re

import numpy as np
from pydantic import BaseModel

from forgeai.config import ForgeConfig
from forgeai.embeddings import Embedder
from forgeai.ingest.models import Chunk
from forgeai.intake.schema import RejectedAssumption, Requirements
from forgeai.intake.structural import split_sentences
from forgeai.llm import call_structured
from forgeai.textmatch import content_terms, is_verbatim, shares_content_term

ASKS_PER_ITEM = 2  # the judge reviews the two closest asks for each refusal
_CLAUSES = re.compile(r",\s+(?:and|but)\s+|;\s+|\s+and also\s+|,\s+", re.IGNORECASE)


class ConsistencyError(Exception):
    """A requirements file asks for something and refuses it."""


class ConflictCheck(BaseModel):
    # Field order matters for constrained decoding: reason, then the evidence, then the verdict.
    reason: str = ""
    quote: str = ""
    contradicts: bool = False


def brief_asks(brief: str, topics: list[str]) -> list[str]:
    """Everything the requester asked for: the must-answer topics and each clause of the brief."""
    clauses = [c.strip() for s in split_sentences(brief) for c in _CLAUSES.split(s) if len(c.split()) >= 3]
    return list(dict.fromkeys([*topics, *clauses]))


def table_vocabulary(chunks: list[Chunk]) -> dict[str, str]:
    """Content terms of every table's column names and cell values, with where they come from."""
    vocab: dict[str, str] = {}
    for chunk in chunks:
        if chunk.kind == "schema":
            for term in content_terms(chunk.text.split("Columns:", 1)[-1]):
                vocab.setdefault(term, f"column of {chunk.doc}: {chunk.text.split('Columns:', 1)[-1].strip()[:120]}")
        elif chunk.kind == "row":
            for term in content_terms(chunk.text):
                vocab.setdefault(term, f"{chunk.chunk_id}: {chunk.text[:100]}")
    return vocab


def find_conflicts(
    candidates: list[tuple[str, str]], req: Requirements, brief: str, chunks: list[Chunk],
    embedder: Embedder, config: ForgeConfig,
) -> list[RejectedAssumption]:
    """Reject ``(field, item)`` exclusions that contradict the brief or a table in the corpus."""
    topics = [t.topic for t in req.must_answer_topics]
    asks = brief_asks(brief, topics + ([] if not req.scope.in_ else list(req.scope.in_)))
    vocab = table_vocabulary(chunks)
    rejected: list[RejectedAssumption] = []
    screened: list[tuple[str, str]] = []
    for field, item in candidates:
        clash = next((t for t in topics if shares_content_term(item, t)), None)
        table_hit = next((vocab[t] for t in content_terms(item) if t in vocab), None)
        if clash:
            rejected.append(RejectedAssumption(
                field=field, item=item, conflict_with="brief", evidence=[],
                detail=f"overlaps the topic the brief asks for: {clash!r}"))
        elif table_hit:
            rejected.append(RejectedAssumption(
                field=field, item=item, conflict_with="corpus", evidence=[table_hit],
                detail="a table in the documents holds this information (column names and cell values are evidence)"))
        else:
            screened.append((field, item))
    if screened and asks:
        rejected += _judge_against_asks(screened, asks, embedder, config)
    return rejected


def _judge_against_asks(
    items: list[tuple[str, str]], asks: list[str], embedder: Embedder, config: ForgeConfig
) -> list[RejectedAssumption]:
    """Embedding similarity picks the closest asks; a judge-model yes/no with a verified quote decides."""
    vectors = embedder.encode([i for _, i in items] + asks)
    item_vectors, ask_vectors = vectors[: len(items)], vectors[len(items):]
    judge = config.judge_model
    rejected: list[RejectedAssumption] = []
    for (field, item), vector in zip(items, item_vectors, strict=True):
        closest = np.argsort(-(ask_vectors @ vector))[:ASKS_PER_ITEM]
        for position in closest:
            ask = asks[int(position)]
            verdict = call_structured(
                "intake.refusal_conflict", "intake/refusal_conflict", {"item": item, "ask": ask},
                ConflictCheck, judge.name, judge.temperature, seed=judge.seed, fallback=ConflictCheck(),
            )
            if verdict.contradicts and verdict.quote.strip() and is_verbatim(verdict.quote, ask):
                rejected.append(RejectedAssumption(
                    field=field, item=item, conflict_with="brief", evidence=[verdict.quote.strip()],
                    detail=f"refusing it would block what was asked: {ask!r} ({verdict.reason.strip()})"))
                break
    return rejected


def contradictions(req: Requirements) -> list[tuple[str, str]]:
    """Must-refuse items that are the SAME topic as a must-answer topic (the invariant).

    "Same" means every content term of the refusal is contained in the answer topic: refusing
    "fines" while answering "overdue fines" is a contradiction, but refusing "waiving fines" (a
    narrower exclusion inside a topic) is not.
    """
    pairs = []
    for refuse in req.must_refuse_topics:
        r_terms = content_terms(refuse.topic)
        for topic in req.must_answer_topics:
            t_terms = content_terms(topic.topic)
            if r_terms and t_terms and r_terms <= t_terms:
                pairs.append((refuse.topic, topic.topic))
    return pairs


def assert_consistent(req: Requirements) -> None:
    """Raise ``ConsistencyError`` if any topic is both must-answer and must-refuse."""
    clashes = contradictions(req)
    if clashes:
        raise ConsistencyError("; ".join(f"must-refuse {r!r} overlaps must-answer {a!r}" for r, a in clashes))
