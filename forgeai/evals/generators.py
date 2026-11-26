"""Candidate generation for every eval case type (spec §6.1).

Each generator asks the agent model to WRITE a question and the exact text it relies on, then
verifies everything it can in code: spans are located in the documents, facts are anchored in the
span, questions must not copy the passage. Anything that fails is returned as a rejection with a
reason, never silently dropped. Judge-model validation happens later, in ``build``.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from forgeai.config import ForgeConfig
from forgeai.evals.compute import ComputePlan, PlanRejected, verify_plan
from forgeai.evals.corpus import EvalCorpus, Table
from forgeai.evals.matching import dates_in, fact_in_text, ngram_overlap, numbers_in
from forgeai.evals.models import EvalCase, Interpretation, KeyFact, RejectedCase
from forgeai.guard.guard import near_miss_questions
from forgeai.ingest.models import Chunk
from forgeai.ingest.spans import SeedSpan
from forgeai.intake.schema import Gap, Requirements
from forgeai.llm import call_structured
from forgeai.prompts import load_prompt
from forgeai.textmatch import is_verbatim, normalize
from forgeai.tools import REGISTRY, tool_description

COMPUTE_ATTEMPTS = 12  # a small model writes a valid, verified computation plan only some of the time
GEN_TEMPERATURE = 0.4  # some variety across candidates (higher values produced runaway repetition); the seed makes each one reproducible


class AnswerableDraft(BaseModel):
    question: str = Field(default="", min_length=1)
    answer_span: str = Field(default="", min_length=1)
    key_facts: list[str] = Field(default_factory=list, min_length=1)


class MultiDraft(BaseModel):
    question: str = Field(default="", min_length=1)
    span_a: str = Field(default="", min_length=1)
    span_b: str = Field(default="", min_length=1)
    facts_a: list[str] = Field(default_factory=list, min_length=1)
    facts_b: list[str] = Field(default_factory=list, min_length=1)


class ReadingDraft(BaseModel):
    reading: str = Field(default="", min_length=1)
    discriminators: list[str] = Field(default_factory=list, min_length=1)
    span: str = Field(default="", min_length=1)
    key_facts: list[str] = Field(default_factory=list, min_length=1)


class AmbiguousDraft(BaseModel):
    question: str = Field(default="", min_length=1)
    readings: list[ReadingDraft] = Field(default_factory=list, min_length=2, max_length=2)


class QuestionsDraft(BaseModel):
    questions: list[str] = Field(default_factory=list)


class QuestionDraft(BaseModel):
    question: str = Field(default="", min_length=1)


@dataclass
class Candidate:
    case: EvalCase
    passages: list[str] = field(default_factory=list)  # what the judge may use to answer (answerability)
    probe: tuple[str, str] | None = None  # (anchor topic, question) that the guard probe must FAIL on
    fact_groups: list[list[KeyFact]] = field(default_factory=list)  # multi-document: facts per document
    group: str = ""  # diversity key for selection


@dataclass
class Generated:
    candidates: list[Candidate] = field(default_factory=list)
    rejected: list[RejectedCase] = field(default_factory=list)

    def reject(self, case_type: str, question: str, reason: str, detail: str = "") -> None:
        self.rejected.append(RejectedCase(case_type=case_type, question=question, reason=reason, detail=detail))


def gap_subject(gap: Gap) -> str:
    if gap.aspect:
        return gap.aspect
    if gap.kind == "unsupported_topic" and "'" in gap.description:
        return gap.description.split("'")[1]
    return gap.evidence[0] if gap.evidence else gap.description


GAP_WHY = {
    "unsupported_topic": "the documents do not cover this",
    "live_data": "it needs a live value and the documents are a static snapshot",
    "capability": "the assistant cannot perform this action",
}


class Generator:
    def __init__(self, req: Requirements, corpus: EvalCorpus, config: ForgeConfig, salt_base: int = 0) -> None:
        self.req, self.corpus, self.config = req, corpus, config
        self.salt_base = salt_base  # a later top-up round writes DIFFERENT candidates, not cached repeats
        self.rng = random.Random(config.agent_model.seed + salt_base)
        self.out = Generated()
        self.thresholds = config.thresholds

    # -- shared helpers -----------------------------------------------------------------------

    def _call(self, prompt_id: str, variables: dict, schema: type[BaseModel], salt: int) -> BaseModel:
        model = self.config.agent_model
        return call_structured(
            f"evals.{prompt_id.split('/')[-1]}", prompt_id, variables, schema, model.name,
            GEN_TEMPERATURE, seed=model.seed + self.salt_base + salt, fallback=schema(),
        )

    def _prov(self, prompt_id: str) -> dict:
        return {"prompt_id": prompt_id, "prompt_version": load_prompt(prompt_id).version}

    MAX_FACT_WORDS = 10  # a key fact is atomic: a number, a date or a short phrase, not a sentence
    MAX_FACTS = 2  # exactly what the question asks for; extra details are not requirements
    UNITS = {"working", "calendar", "business", "consecutive", "day", "days", "week", "weeks", "month", "months",
             "year", "years", "hour", "hours", "characters", "percent", "usd", "dollars", "per", "item", "items"}

    @classmethod
    def fact_kind(cls, text: str) -> str:
        """number only for a bare quantity ("24 working days", "USD 5,000", "5%"); "31 January" is a phrase."""
        tokens = re.findall(r"[A-Za-z]+|\d[\d,.]*%?", text)
        if dates_in(text):
            return "date"
        if tokens and numbers_in(text) and all(re.fullmatch(r"\d[\d,.]*%?", t) or t.lower() in cls.UNITS for t in tokens):
            return "number"
        return "phrase"

    def _facts(self, drafts: list[str]) -> list[KeyFact]:
        facts = []
        for draft in drafts:
            text = draft.strip()
            if not text or len(text.split()) > self.MAX_FACT_WORDS or re.match(r"(?i)^(and|or|but|that|which)\b", text):
                continue  # sentences and dangling fragments are not atomic facts; they cannot be matched reliably
            facts.append(KeyFact(text=text, kind=self.fact_kind(text)))
        return facts[: self.MAX_FACTS]

    @staticmethod
    def _unanchored(facts: list[KeyFact], span_text: str) -> list[str]:
        return [f.text for f in facts if not fact_in_text(f.kind, f.text, span_text, strict=True)]

    def _span_in_chunk(self, chunk: Chunk, text: str) -> SeedSpan | None:
        """The model's quote, located inside this chunk's character range (rows: the whole row)."""
        if chunk.kind == "row":  # a table row is one record: the span is the whole row, but the quote must be in it
            return self.corpus.chunk_span(chunk) if is_verbatim(text, chunk.text) else None
        span = self.corpus.span_of(chunk.doc, text)
        if span is None or chunk.char_start is None or chunk.char_end is None:
            return None
        return span if span.char_start < chunk.char_end and chunk.char_start < span.char_end else None

    # -- answerable (single document) ---------------------------------------------------------

    def _answerable_seeds(self) -> list[Chunk]:
        by_id = {c.chunk_id: c for c in self.corpus.chunks}
        pools: list[list[Chunk]] = []
        for topic in self.req.must_answer_topics:  # stratify by topic first
            pools.append([by_id[e] for e in topic.evidence if e in by_id and by_id[e].kind == "prose"])
        by_doc: dict[str, list[Chunk]] = {}
        for chunk in self.corpus.seed_chunks("prose"):
            by_doc.setdefault(chunk.doc, []).append(chunk)
        for chunks in by_doc.values():
            self.rng.shuffle(chunks)
            pools.append(chunks)
        rows = self.corpus.seed_chunks("row")
        self.rng.shuffle(rows)
        pools.append(rows[:3])  # a few table rows, so tables are tested too
        seeds: list[Chunk] = []
        seen: set[str] = set()
        for rank in range(max((len(p) for p in pools), default=0)):  # round robin across the pools
            for pool in pools:
                if rank < len(pool) and pool[rank].chunk_id not in seen:
                    seen.add(pool[rank].chunk_id)
                    seeds.append(pool[rank])
        return seeds

    def answerable(self, n: int) -> None:
        seeds = self._answerable_seeds()
        for i in range(n if seeds else 0):
            self._one_answerable(seeds[i % len(seeds)], salt=i, snapshot=None)

    def snapshot_cases(self, per_aspect: int = 1) -> None:
        """Cases for the guard's supported aspects of live-tied topics: the snapshot date is a key fact."""
        by_id = {c.chunk_id: c for c in self.corpus.chunks}
        for topic in self.req.must_answer_topics:
            if not (topic.snapshot_as_of and topic.supported_aspects):
                continue
            seeds = [by_id[e] for e in topic.evidence if e in by_id]
            for k, aspect in enumerate(topic.supported_aspects[:2]):
                for j in range(per_aspect if seeds else 0):
                    self._one_answerable(seeds[(k + j) % len(seeds)], salt=500 + 10 * k + j,
                                         snapshot=(topic.id, aspect, topic.snapshot_as_of))

    def _one_answerable(self, chunk: Chunk, salt: int, snapshot: tuple[str, str, str] | None) -> None:
        focus = load_prompt("evals/focus").render({"aspect": snapshot[1]})[1] if snapshot else ""
        draft = self._call("evals/answerable", {"doc": chunk.doc, "passage": chunk.text, "focus": focus},
                           AnswerableDraft, salt)
        question = draft.question.strip()  # type: ignore[attr-defined]
        if not question:
            return self.out.reject("answerable", "", "model output invalid", chunk.chunk_id)
        span = self._span_in_chunk(chunk, draft.answer_span)  # type: ignore[attr-defined]
        if span is None:
            return self.out.reject("answerable", question, "answer_span is not a verbatim quote of the seed", chunk.chunk_id)
        facts = self._facts(draft.key_facts)  # type: ignore[attr-defined]
        if snapshot:
            facts.append(KeyFact(text=snapshot[2], kind="snapshot_date"))
        if not facts:
            return self.out.reject("answerable", question, "no key facts", chunk.chunk_id)
        missing = self._unanchored(facts, span.span_text)
        if missing:
            return self.out.reject("answerable", question, "fact anchoring failed", f"not in the span: {missing}")
        overlap = ngram_overlap(question, chunk.text)
        if overlap >= self.thresholds.paraphrase_overlap_max:
            return self.out.reject("answerable", question, "question copies the passage", f"overlap {overlap:.2f}")
        if chunk.kind == "row" and not self._identifies_row(chunk, question):
            return self.out.reject("answerable", question, "question does not identify one table row", chunk.chunk_id)
        case = EvalCase(
            id="", type="answerable", question=question, expected_status="answered",
            expected_sources=[chunk.doc], key_facts=facts, seed_spans=[span],
            origin="snapshot" if snapshot else "topic", topic_ref=snapshot[0] if snapshot else "",
            tags=["snapshot"] if snapshot else [], provenance=self._prov("evals/answerable"),
            validation={"fact_anchoring": True, "paraphrase_overlap": round(overlap, 2)})
        self.out.candidates.append(Candidate(case, [chunk.text], group=chunk.doc))

    def _identifies_row(self, chunk: Chunk, question: str) -> bool:
        """A question about a table row must name the values that identify it (several rows look alike)."""
        table = self.corpus.tables.get(chunk.doc)
        if table is None or chunk.row is None or chunk.row > len(table.rows):
            return False
        row = table.rows[chunk.row - 1]
        key = self.identifying_columns(table)
        return bool(key) and all(row[c].strip().lower() in question.lower() for c in key)

    # -- multi-document ------------------------------------------------------------------------

    def _pairs(self, same_doc_ok: bool, limit: int) -> list[tuple[Chunk, Chunk, float]]:
        pool = self.corpus.seed_chunks("prose")
        pool += self.rng.sample([c for c in self.corpus.chunks if c.kind == "row"],
                                k=min(3, sum(c.kind == "row" for c in self.corpus.chunks)))
        position = {c.chunk_id: i for i, c in enumerate(self.corpus.chunks)}
        vectors = self.corpus.index.vectors[[position[c.chunk_id] for c in pool]]
        sims = vectors @ vectors.T
        order = np.dstack(np.unravel_index(np.argsort(-sims, axis=None), sims.shape))[0]
        used: set[str] = set()
        pairs: list[tuple[Chunk, Chunk, float]] = []
        for i, j in order:
            a, b = pool[int(i)], pool[int(j)]
            if int(i) >= int(j) or (not same_doc_ok and a.doc == b.doc) or a.chunk_id in used or b.chunk_id in used:
                continue
            if sims[i, j] > 0.985:  # near-identical text is not two readings or two sources
                continue
            used.update({a.chunk_id, b.chunk_id})
            pairs.append((a, b, float(sims[i, j])))
            if len(pairs) >= limit:
                break
        return pairs

    def multidoc(self, n: int) -> None:
        for i, (a, b, _) in enumerate(self._pairs(same_doc_ok=False, limit=n)):
            draft = self._call("evals/multidoc", {"doc_a": a.doc, "passage_a": a.text, "doc_b": b.doc, "passage_b": b.text},
                               MultiDraft, 100 + i)
            question = draft.question.strip()  # type: ignore[attr-defined]
            span_a = self._span_in_chunk(a, draft.span_a)  # type: ignore[attr-defined]
            span_b = self._span_in_chunk(b, draft.span_b)  # type: ignore[attr-defined]
            if not question or span_a is None or span_b is None:
                self.out.reject("multidoc", question, "span is not a verbatim quote of its passage", f"{a.chunk_id} + {b.chunk_id}")
                continue
            facts_a, facts_b = self._facts(draft.facts_a), self._facts(draft.facts_b)  # type: ignore[attr-defined]
            if not facts_a or not facts_b:
                self.out.reject("multidoc", question, "each document must contribute at least one key fact")
                continue
            missing = self._unanchored(facts_a, span_a.span_text) + self._unanchored(facts_b, span_b.span_text)
            if missing:
                self.out.reject("multidoc", question, "fact anchoring failed", f"not in its span: {missing}")
                continue
            overlap = max(ngram_overlap(question, a.text), ngram_overlap(question, b.text))
            if overlap >= self.thresholds.paraphrase_overlap_max:
                self.out.reject("multidoc", question, "question copies the passage", f"overlap {overlap:.2f}")
                continue
            case = EvalCase(
                id="", type="multidoc", question=question, expected_status="answered",
                expected_sources=[a.doc, b.doc], key_facts=facts_a + facts_b, seed_spans=[span_a, span_b],
                origin="topic", provenance=self._prov("evals/multidoc"),
                validation={"fact_anchoring": True, "paraphrase_overlap": round(overlap, 2), "documents": 2})
            self.out.candidates.append(Candidate(case, [a.text, b.text], group=f"{a.doc}+{b.doc}",
                                                 fact_groups=[facts_a, facts_b]))

    # -- ambiguous -----------------------------------------------------------------------------

    def ambiguous(self, n: int) -> None:
        for i, (a, b, _) in enumerate(self._pairs(same_doc_ok=True, limit=n)):
            if a.kind == "row" or b.kind == "row":
                continue
            draft = self._call("evals/ambiguous", {"doc_a": a.doc, "passage_a": a.text, "doc_b": b.doc, "passage_b": b.text},
                               AmbiguousDraft, 200 + i)
            question = draft.question.strip()  # type: ignore[attr-defined]
            readings = draft.readings[:2]  # type: ignore[attr-defined]
            if not question or len(readings) != 2:
                self.out.reject("ambiguous", question, "needs exactly two readings")
                continue
            interpretations, failure = [], ""
            for chunk, reading in zip((a, b), readings, strict=True):
                span = self._span_in_chunk(chunk, reading.span)
                facts = self._facts(reading.key_facts)
                leaked = [d for d in reading.discriminators if d.strip() and re.search(rf"\b{re.escape(d.strip().lower())}\b", question.lower())]
                if span is None or not facts:
                    failure = "a reading's span is not a verbatim quote or has no key facts"
                elif self._unanchored(facts, span.span_text):
                    failure = "fact anchoring failed for a reading"
                elif leaked or not reading.discriminators:
                    failure = f"the question contains a discriminator {leaked}" if leaked else "a reading has no discriminator"
                else:
                    interpretations.append(Interpretation(reading=reading.reading, discriminators=reading.discriminators,
                                                          key_facts=facts, seed_span=span))
                if failure:
                    break
            if failure:
                self.out.reject("ambiguous", question, failure)
                continue
            if interpretations[0].seed_span.span_text == interpretations[1].seed_span.span_text:
                self.out.reject("ambiguous", question, "both readings use the same span")
                continue
            case = EvalCase(
                id="", type="ambiguous", question=question, expected_status="needs_clarification",
                expected_sources=sorted({a.doc, b.doc}), seed_spans=[i.seed_span for i in interpretations],
                interpretations=interpretations, origin="topic", provenance=self._prov("evals/ambiguous"),
                validation={"fact_anchoring": True, "readings": 2, "accepted_statuses": ["needs_clarification", "answered covering both readings"]})
            self.out.candidates.append(Candidate(case, [], group=f"{a.doc}+{b.doc}"))

    # -- out of scope (must refuse) ------------------------------------------------------------

    def refusals(self, n: int) -> None:
        subjects: list[tuple[str, str, str, str, str, bool]] = []  # origin, ref, subject, why, anchor, probe?
        for refuse in self.req.must_refuse_topics[:3]:
            subjects.append(("must_refuse", refuse.id, refuse.topic, refuse.reason, refuse.topic, False))
        for gap in [g for g in self.req.gaps if g.recommended_action != "warn"][:4]:
            subjects.append(("gap", gap.id, gap_subject(gap), GAP_WHY[gap.kind], gap_subject(gap), gap.kind == "unsupported_topic"))
        for tool in self._unattached_tools()[:2]:
            subjects.append(("negative_tool", tool, tool, "this assistant has no such capability", tool, True))
        for k, (origin, ref, subject, why, anchor, probe) in enumerate(subjects):
            prompt = "evals/negative_tool" if origin == "negative_tool" else "evals/refusal_question"
            variables = ({"purpose": self.req.purpose, "tool": tool_description(subject)} if origin == "negative_tool"
                         else {"purpose": self.req.purpose, "subject": subject, "why": why})
            for question in self._call(prompt, variables, QuestionsDraft, 300 + k).questions[:2]:  # type: ignore[attr-defined]
                self._add_refusal(question, origin, ref, subject, anchor, probe, prompt)
        for k, topic in enumerate(self.req.must_answer_topics[:3]):
            for question in near_miss_questions(topic.topic, self.req.purpose, self.corpus.index, self.config):
                self._add_refusal(question, "near_miss", topic.id, topic.topic, topic.topic, True, "guard/near_miss")

    def _add_refusal(self, question: str, origin: str, ref: str, subject: str, anchor: str, probe: bool, prompt: str) -> None:
        question = question.strip()
        if not question:
            return
        case = EvalCase(
            id="", type="out_of_scope", question=question, expected_status="refused", origin=origin, topic_ref=ref,
            provenance=self._prov(prompt), validation={"subject": subject},
            tags=["live"] if origin == "gap" and "live" in ref.lower() else [])
        self.out.candidates.append(Candidate(case, [], probe=(anchor, question) if probe else None, group=f"{origin}:{ref}"))

    def _unattached_tools(self) -> list[str]:
        """Tools the requirements do not call for (isolation: derived from capabilities, not from the agent)."""
        needed = set(self.req.capabilities_needed)
        return [name for name, spec in REGISTRY.items() if spec.capability not in needed]

    # -- tool-required (lookup and computed) ---------------------------------------------------

    def tool_cases(self, n: int) -> None:
        wants_lookup = "structured_lookup" in self.req.capabilities_needed and bool(self.corpus.tables)
        wants_compute = "arithmetic" in self.req.capabilities_needed
        if wants_lookup and wants_compute:
            self._lookups(math.ceil(n / 2)), self._computed(COMPUTE_ATTEMPTS)  # plans fail often; try many
        elif wants_lookup:
            self._lookups(n)
        elif wants_compute:
            self._computed(max(n, COMPUTE_ATTEMPTS))

    @staticmethod
    def identifying_columns(table: Table) -> list[str]:
        """The smallest set of columns that identifies each row uniquely (preferring text columns)."""
        from itertools import combinations

        def numeric(col: str) -> bool:
            return all(re.fullmatch(r"-?\d+(\.\d+)?", row[col].strip() or "0") for row in table.rows[:30])

        columns = sorted(table.columns, key=lambda c: (numeric(c), -len({r[c] for r in table.rows})))
        for pool in ([c for c in columns if not numeric(c)], columns):  # a price or count is a poor identifier
            for size in (1, 2, 3):
                for combo in combinations(pool[:6], size):
                    if len({tuple(r[c] for c in combo) for r in table.rows}) == len(table.rows):
                        return list(combo)
        return []

    def _lookups(self, n: int) -> None:
        tables = list(self.corpus.tables.values())
        for i in range(n):
            table = tables[i % len(tables)]
            key_cols = self.identifying_columns(table)
            targets = [c for c in table.columns if c not in key_cols]
            if not key_cols or not targets:
                self.out.reject("tool_required", "", "table rows cannot be identified uniquely", table.doc)
                continue
            index = self.rng.randrange(len(table.rows))
            row, column = table.rows[index], targets[(i + index) % len(targets)]
            key = ", ".join(f"{c} = {row[c]}" for c in key_cols)
            question = self._call("evals/lookup_question", {"table": table.doc, "key": key, "column": column},
                                  QuestionDraft, 400 + i).question.strip()  # type: ignore[attr-defined]
            value = row[column].strip()
            if not question or not all(row[c].lower() in question.lower() for c in key_cols):
                self.out.reject("tool_required", question, "question does not contain the identifying values", key)
                continue
            if value and re.search(rf"(?<![\w.]){re.escape(value.lower())}(?![\w])", question.lower()):
                self.out.reject("tool_required", question, "question reveals the answer", value)
                continue
            kind = "number" if numbers_in(value) and re.fullmatch(r"-?[\d,]*\.?\d+", value) else "phrase"
            case = EvalCase(
                id="", type="tool_required", question=question, expected_status="answered",
                expected_sources=[table.doc], key_facts=[KeyFact(text=value, kind=kind)],
                seed_spans=[table.row_spans[index]], expected_tools=["csv_lookup"], origin="lookup",
                provenance=self._prov("evals/lookup_question"),
                validation={"expected_value_computed_by_code": True, "source": "table cell", "column": column})
            self.out.candidates.append(Candidate(case, [], group=f"lookup:{table.doc}"))

    def _computed(self, n: int) -> None:
        numeric_chunks = [c for c in self.corpus.chunks if c.kind == "prose" and re.search(r"\d", c.text)]
        tables = list(self.corpus.tables.values())
        for i in range(n):
            if tables:
                table = tables[i % len(tables)]
                pick = self.rng.randrange(len(table.rows))
                row = table.rows[pick]
                rows_text = "\n".join(
                    f"[{table.doc}] " + " | ".join(f"{c}: {r[c]}" for c in table.columns)
                    for r in [row, *self.rng.sample(table.rows, k=min(2, len(table.rows)))])
                # Prose rules that mention this row's values (an identifier, a name). Table rows would crowd
                # prose out of any top-k retrieval, so rank the prose chunks directly by shared values.
                values = [v.lower() for v in row.values() if len(v.strip()) >= 3]
                ranked = sorted(numeric_chunks, key=lambda c: -sum(v in c.text.lower() for v in values))
                passages = [c for c in ranked if any(v in c.text.lower() for v in values)][:3]
            else:
                rows_text, passages = "(no table)", self.rng.sample(numeric_chunks, k=min(3, len(numeric_chunks)))
            if not passages:
                self.out.reject("tool_required", "", "no numeric passages to build a computation from")
                continue
            passages_text = "\n\n".join(f"[{c.doc}]\n{c.text}" for c in passages)
            plan = self._call("evals/compute_plan", {"rows": rows_text, "passages": passages_text}, ComputePlan, 600 + i)
            try:
                verified = verify_plan(plan, self.corpus, cross_source=bool(tables) and bool(passages))  # type: ignore[arg-type]
            except PlanRejected as exc:
                self.out.reject("tool_required", plan.question, "computed case failed verification", str(exc))  # type: ignore[attr-defined]
                continue
            tools = ["calculator"] + (["csv_lookup"] if verified.uses_table else [])
            case = EvalCase(
                id="", type="tool_required", question=verified.question, expected_status="answered",
                expected_sources=verified.docs, key_facts=[KeyFact(text=verified.answer_text, kind="number")],
                seed_spans=verified.spans, expected_tools=tools, origin="computed",
                provenance=self._prov("evals/compute_plan") | {"formula": plan.formula, "values": verified.values},  # type: ignore[attr-defined]
                validation={"expected_value_computed_by_code": True, "conditions_checked": len(plan.conditions)})  # type: ignore[attr-defined]
            self.out.candidates.append(Candidate(case, [], group="computed"))


def normalise_question(text: str) -> str:
    return normalize(text).lower()
