"""Compositional eval cases: multi-document and ambiguous (P5.1 correction 3).

Both are built from material code has already verified, so the model's part is small.

* Multi-document: take two single-document cases that already passed validation, from different
  documents and about related things (question embeddings in a similarity band: related, not
  near-duplicates). The model only merges the two questions into one natural question. The key
  facts are the two validated sets; the independent-answerability check later runs with both spans.
* Ambiguous: code first FINDS an attribute that has different values in different documents or
  sections ("warranty period ... is 12 months" / "... is 18 months"). The model only writes the
  under-specified question. The key facts are the two differing values (anchored by construction),
  and a case is rejected unless the two readings have distinct values and the question avoids the
  words that tell the readings apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from forgeai.evals.matching import fact_in_text, ngram_overlap
from forgeai.evals.models import EvalCase, Interpretation, KeyFact
from forgeai.evals.units import Quantity, quantities_in
from forgeai.ingest.spans import SeedSpan
from forgeai.intake.structural import split_sentences
from forgeai.textmatch import STOPWORDS, content_terms

if TYPE_CHECKING:
    from forgeai.evals.generators import Candidate, Generator

CLAUSE_NUMBER = re.compile(r"^\d+(?:\.\d+)*\.?\s+")
PAGE_TAIL = re.compile(r"\s*Page \d+\s*$")
PAGE_FOOTER = re.compile(r"(?<!\w)Page \d+(?!\w)")
MAX_FACTS = 3


class MergedQuestion(BaseModel):
    question: str = Field(default="", min_length=1)


class AmbiguousDraft(BaseModel):
    question: str = Field(default="", min_length=1)
    reading_a: str = Field(default="", min_length=1)
    reading_b: str = Field(default="", min_length=1)


# ---- multi-document ---------------------------------------------------------------------------------------


def related_pairs(
    singles: list[Candidate], vectors: np.ndarray, low: float, high: float, limit: int,
) -> list[tuple[int, int, float]]:
    """Pairs of single-document cases from different documents whose questions are related but not twins."""
    sims = vectors @ vectors.T
    order = np.dstack(np.unravel_index(np.argsort(-sims, axis=None), sims.shape))[0]
    used: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for i, j in order:
        i, j = int(i), int(j)
        if i >= j or i in used or j in used:
            continue
        a, b = singles[i].case, singles[j].case
        if set(a.expected_sources) & set(b.expected_sources) or not low <= sims[i, j] < high:
            continue
        used.update({i, j})
        pairs.append((i, j, float(sims[i, j])))
        if len(pairs) >= limit:
            break
    return pairs


def eligible_singles(candidates: list[Candidate]) -> list[Candidate]:
    return [c for c in candidates
            if c.case.type == "answerable" and c.case.origin == "topic" and c.passages and len(c.case.expected_sources) == 1
            and c.case.key_facts and not c.case.tags]


def merge_facts(a: EvalCase, b: EvalCase) -> list[KeyFact]:
    """The union of the two validated fact sets, at most three, the primary fact of each document first."""
    ordered = [a.key_facts[0], b.key_facts[0], *a.key_facts[1:], *b.key_facts[1:]]
    return list({f.text: f for f in ordered}.values())[:MAX_FACTS]


DERIVED = re.compile(
    r"(?<!\w)(total|totals|sum|combined|overall|altogether|net|estimate|estimated|calculate|calculated|"
    r"discounted|payable|how much (?:would|will|do|does)|in all)(?!\w)", re.IGNORECASE)


def derived_terms(text: str) -> set[str]:
    """Words that ask for a value computed from other values (a total, a cost after a discount, an estimate)."""
    return {m.group(0).lower() for m in DERIVED.finditer(text)}


def asks_for_derived_value(merged: str, first: str, second: str) -> set[str]:
    """Derived-value words the merge ADDED. Two validated lookups give two inputs, not their combination: a question
    that asks for the combination ("total cost") has an answer the case's key facts do not contain."""
    return derived_terms(merged) - derived_terms(first) - derived_terms(second)


def compose_multidoc(gen: Generator, singles: list[Candidate], limit: int) -> None:
    from forgeai.evals.generators import Candidate

    pool = eligible_singles(singles)
    if len(pool) < 2:
        return
    vectors = gen.corpus.embedder.encode([c.case.question for c in pool])
    t = gen.thresholds
    made = 0
    for k, (i, j, sim) in enumerate(related_pairs(pool, vectors, t.compose_similarity_min, t.compose_similarity_max,
                                                  limit + len(gen.skip_pairs))):
        a, b = pool[i], pool[j]
        key = frozenset({a.case.question, b.case.question})
        if key in gen.skip_pairs or made >= limit:
            continue
        gen.skip_pairs.add(key)
        made += 1
        draft = gen._call("evals/merge_questions", {
            "doc_a": a.case.expected_sources[0], "question_a": a.case.question,
            "doc_b": b.case.expected_sources[0], "question_b": b.case.question}, MergedQuestion, 100 + k)
        question = draft.question.strip()  # type: ignore[attr-defined]
        if not question:
            gen.out.reject("multidoc", "", "model output invalid", f"{a.case.question!r} + {b.case.question!r}")
            continue
        terms, only_a, only_b = content_terms(question), content_terms(a.case.question) - content_terms(b.case.question), \
            content_terms(b.case.question) - content_terms(a.case.question)
        if not (terms & only_a and terms & only_b):
            gen.out.reject("multidoc", question, "the merged question dropped one of the two questions",
                           f"{a.case.question!r} + {b.case.question!r}")
            continue
        overlap = max(ngram_overlap(question, p) for p in (*a.passages, *b.passages))
        if overlap >= t.paraphrase_overlap_max:
            gen.out.reject("multidoc", question, "question copies the passage", f"overlap {overlap:.2f}")
            continue
        added = asks_for_derived_value(question, a.case.question, b.case.question)
        if added:
            gen.out.reject("multidoc", question, "asks for a value the key facts do not determine",
                           f"the merge added {sorted(added)}; the key facts are inputs, not their combination")
            continue
        facts = merge_facts(a.case, b.case)
        groups = [[f for f in facts if f in a.case.key_facts], [f for f in facts if f in b.case.key_facts]]
        if not all(groups):
            gen.out.reject("multidoc", question, "each document must contribute at least one key fact")
            continue
        case = EvalCase(
            id="", type="multidoc", question=question, expected_status="answered",
            expected_sources=[a.case.expected_sources[0], b.case.expected_sources[0]], key_facts=facts,
            seed_spans=[*a.case.seed_spans, *b.case.seed_spans], origin="composed",
            covers=sorted(set(a.case.covers) | set(b.case.covers)),
            provenance=gen._prov("evals/merge_questions") | {"composed_from": [a.case.question, b.case.question]},
            validation={"fact_anchoring": True, "paraphrase_overlap": round(overlap, 2), "documents": 2,
                        "pair_similarity": round(sim, 3), "key_facts": "union of the two validated sets"})
        gen.out.candidates.append(Candidate(case, [*a.passages, *b.passages],
                                            group=f"{case.expected_sources[0]}+{case.expected_sources[1]}", fact_groups=groups))


# ---- ambiguous --------------------------------------------------------------------------------------------


@dataclass
class Attribute:
    doc: str
    heading: str
    sentence: str
    masked: str
    quantity: Quantity
    span: SeedSpan
    label: str  # what distinguishes where it comes from: the document's title or its section


def doc_label(corpus, doc: str) -> str:  # noqa: ANN001
    for block in corpus.docs[doc].blocks:
        text = PAGE_TAIL.sub("", block.text.strip())
        if text:
            return text[:120]
    return doc


def find_attributes(corpus) -> list[Attribute]:  # noqa: ANN001
    """Sentences stating exactly one quantity with a known unit, with the quantity masked out."""
    found: list[Attribute] = []
    seen: set[str] = set()
    for chunk in (c for c in corpus.chunks if c.kind == "prose"):
        for raw in split_sentences(chunk.text):
            sentence = CLAUSE_NUMBER.sub("", raw)
            quantities = [q for q in quantities_in(sentence) if q.unit is not None]
            if (len(quantities) != 1 or not 6 <= len(sentence.split()) <= 40 or sentence in seen
                    or PAGE_FOOTER.search(sentence)):
                continue
            span = corpus.span_of(chunk.doc, sentence)
            if span is None:
                continue
            seen.add(sentence)
            q = quantities[0]
            heading = " > ".join(chunk.heading_path)
            found.append(Attribute(chunk.doc, heading, sentence, sentence.replace(q.text, "<N>", 1), q, span,
                                   heading or doc_label(corpus, chunk.doc)))
    return found


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3 and w not in STOPWORDS and not w.isdigit()}


def discriminators(a: Attribute, b: Attribute, corpus) -> tuple[list[str], list[str]]:  # noqa: ANN001
    """Words that tell reading A from reading B: in its source's title, section or sentence, and not in the other's."""
    def vocabulary(x: Attribute) -> set[str]:
        return _words(f"{doc_label(corpus, x.doc)} {x.heading} {x.sentence}")

    shared = content_terms(a.sentence) & content_terms(b.sentence)  # the attribute itself is never a discriminator
    only_a = sorted(w for w in vocabulary(a) - vocabulary(b) if content_terms(w) - shared or not content_terms(w))
    only_b = sorted(w for w in vocabulary(b) - vocabulary(a) if content_terms(w) - shared or not content_terms(w))
    return only_a, only_b


def conflicting_pairs(attrs: list[Attribute], vectors: np.ndarray, min_sim: float, limit: int) -> list[tuple[Attribute, Attribute, float]]:
    """The same kind of statement with different values in different documents or sections."""
    sims = vectors @ vectors.T
    order = np.dstack(np.unravel_index(np.argsort(-sims, axis=None), sims.shape))[0]
    used: set[int] = set()
    pairs: list[tuple[Attribute, Attribute, float]] = []
    for i, j in order:
        i, j = int(i), int(j)
        if i >= j or i in used or j in used or sims[i, j] < min_sim:
            continue
        a, b = attrs[i], attrs[j]
        shared = content_terms(a.masked) & content_terms(b.masked)
        union = content_terms(a.masked) | content_terms(b.masked)
        if (a.quantity.unit != b.quantity.unit or a.quantity.value == b.quantity.value
                or len(shared) < 3 or len(shared) / max(len(union), 1) < 0.5):
            continue
        used.update({i, j})
        pairs.append((a, b, float(sims[i, j])))
        if len(pairs) >= limit:
            break
    return pairs


def compose_ambiguous(gen: Generator, limit: int) -> None:
    from forgeai.evals.generators import Candidate

    attrs = find_attributes(gen.corpus)
    if len(attrs) < 2:
        return
    vectors = gen.corpus.embedder.encode([a.masked for a in attrs])
    for k, (a, b, sim) in enumerate(conflicting_pairs(attrs, vectors, gen.thresholds.ambiguity_similarity_min, limit)):
        only_a, only_b = discriminators(a, b, gen.corpus)
        if not (only_a or only_b):
            gen.out.reject("ambiguous", "", "the two readings cannot be told apart by any word", f"{a.sentence!r} / {b.sentence!r}")
            continue
        draft = gen._call("evals/ambiguous_question", {
            "label_a": a.label, "sentence_a": a.sentence, "label_b": b.label, "sentence_b": b.sentence,
            "discriminators": ", ".join(only_a + only_b)}, AmbiguousDraft, 200 + k)
        question = draft.question.strip()  # type: ignore[attr-defined]
        if not question:
            gen.out.reject("ambiguous", "", "model output invalid")
            continue
        lowered = question.lower()
        leaked = [w for w in only_a + only_b if re.search(rf"\b{re.escape(w)}\b", lowered)]
        if leaked:
            gen.out.reject("ambiguous", question, "the question contains a discriminator", f"{leaked}")
            continue
        shared = content_terms(a.sentence) & content_terms(b.sentence)
        if len(content_terms(question) & shared) < 2:
            gen.out.reject("ambiguous", question, "the question does not name the attribute both readings are about",
                           f"shared terms {sorted(shared)}")
            continue
        facts = [KeyFact(text=x.quantity.text, kind=gen.fact_kind(x.quantity.text)) for x in (a, b)]  # type: ignore[arg-type]
        if facts[0].text == facts[1].text or any(not fact_in_text(f.kind, f.text, x.sentence, strict=True) for f, x in zip(facts, (a, b), strict=True)):
            gen.out.reject("ambiguous", question, "the two readings do not have distinct, anchored key facts")
            continue
        interpretations = [
            Interpretation(reading=reading.strip(), discriminators=words[:3], key_facts=[fact], seed_span=x.span)
            for x, fact, words, reading in zip((a, b), facts, (only_a, only_b), (draft.reading_a, draft.reading_b), strict=True)]  # type: ignore[attr-defined]
        case = EvalCase(
            id="", type="ambiguous", question=question, expected_status="needs_clarification",
            expected_sources=sorted({a.doc, b.doc}), seed_spans=[a.span, b.span], interpretations=interpretations,
            origin="conflict", provenance=gen._prov("evals/ambiguous_question"),
            validation={"fact_anchoring": True, "readings": 2, "pair_similarity": round(sim, 3),
                        "found_by": "same statement, different value, different document or section",
                        "accepted_statuses": ["needs_clarification", "answered covering both readings"]})
        gen.out.candidates.append(Candidate(case, [], group=f"{a.doc}+{b.doc}"))
