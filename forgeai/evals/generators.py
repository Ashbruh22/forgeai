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

from pydantic import BaseModel, Field

from forgeai.config import ForgeConfig
from forgeai.evals.corpus import EvalCorpus, Table
from forgeai.evals.matching import dates_in, fact_in_text, ngram_overlap, numbers_in
from forgeai.evals.models import EvalCase, KeyFact, RejectedCase
from forgeai.evals.templates import (
    TEMPLATES, PlanRejected, TemplateChoice, build_menu, instantiate, template_help,
)
from forgeai.guard.guard import near_miss_questions
from forgeai.ingest.models import Chunk
from forgeai.ingest.spans import SeedSpan
from forgeai.intake.schema import Gap, Requirements
from forgeai.llm import call_structured
from forgeai.prompts import load_prompt
from forgeai.textmatch import is_verbatim, normalize
from forgeai.tools import REGISTRY, tool_description

COMPUTE_ATTEMPTS = 8  # attempts per computed requirement; a small model binds a valid template only some of the time
GEN_TEMPERATURE = 0.4  # some variety across candidates (higher values produced runaway repetition); the seed makes each one reproducible


class AnswerableDraft(BaseModel):
    question: str = Field(default="", min_length=1)
    answer_span: str = Field(default="", min_length=1)
    key_facts: list[str] = Field(default_factory=list, min_length=1)


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
    ref: str = ""  # set while generating for one requirement item, so its rejections can be attributed to it

    def reject(self, case_type: str, question: str, reason: str, detail: str = "") -> None:
        self.rejected.append(RejectedCase(case_type=case_type, question=question, reason=reason, detail=detail, ref=self.ref))


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
        self.skip_pairs: set[frozenset[str]] = set()  # merged pairs of earlier rounds: not merged again

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
    MAX_FACTS = 3  # at most three, and each must be answer-bearing (pruned later; see build.prune_facts)
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

    def _chunk_topics(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for topic in self.req.must_answer_topics:
            for chunk_id in topic.evidence:
                mapping.setdefault(chunk_id, []).append(topic.id)
        return mapping

    def topic_seeds(self, topic_id: str) -> list[Chunk]:
        """Chunks of a topic's verified evidence (prose passages and table rows) that are worth a question."""
        by_id = {c.chunk_id: c for c in self.corpus.chunks}
        topic = next((t for t in self.req.must_answer_topics if t.id == topic_id), None)
        enough = {c.chunk_id for c in self.corpus.seed_chunks("prose") + self.corpus.seed_chunks("row")}
        seeds = [by_id[e] for e in (topic.evidence if topic else []) if e in by_id and e in enough]
        self.rng.shuffle(seeds)
        return seeds

    def _answerable_seeds(self) -> list[Chunk]:
        pools: list[list[Chunk]] = []
        for topic in self.req.must_answer_topics:  # stratify by topic first
            pools.append(self.topic_seeds(topic.id))
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

    def for_topic(self, topic_id: str, n: int, salt: int = 900) -> None:
        """Directed generation: n questions seeded from the evidence of one must-answer topic."""
        seeds = self.topic_seeds(topic_id)
        for i in range(n if seeds else 0):
            self._one_answerable(seeds[i % len(seeds)], salt=salt + i, snapshot=None)

    def snapshot_cases(self, per_aspect: int = 1, only: set[str] | None = None, salt: int = 500) -> None:
        """Cases for the guard's supported aspects of live-tied topics: the snapshot date is a key fact."""
        by_id = {c.chunk_id: c for c in self.corpus.chunks}
        for topic in self.req.must_answer_topics:
            if not (topic.snapshot_as_of and topic.supported_aspects):
                continue
            seeds = [by_id[e] for e in topic.evidence if e in by_id]
            for k, aspect in enumerate(topic.supported_aspects[:2]):
                if only is not None and f"aspect:{topic.id}/{aspect}" not in only:
                    continue
                for j in range(per_aspect if seeds else 0):
                    self._one_answerable(seeds[(k + j) % len(seeds)], salt=salt + 10 * k + j,
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
        topics = self._chunk_topics().get(chunk.chunk_id, [])
        covers = [f"topic:{t}" for t in topics]
        if snapshot:
            covers = [f"topic:{snapshot[0]}", f"aspect:{snapshot[0]}/{snapshot[1]}"]
        case = EvalCase(
            id="", type="answerable", question=question, expected_status="answered",
            expected_sources=[chunk.doc], key_facts=facts, seed_spans=[span],
            origin="snapshot" if snapshot else "topic", topic_ref=snapshot[0] if snapshot else (topics[0] if topics else ""),
            aspect=snapshot[1] if snapshot else "", covers=covers,
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

    # -- compositional cases (see forgeai.evals.compose) -------------------------------------------

    def multidoc_from(self, singles: list[Candidate], n: int) -> None:
        from forgeai.evals.compose import compose_multidoc

        compose_multidoc(self, singles, n)

    def ambiguous(self, n: int) -> None:
        from forgeai.evals.compose import compose_ambiguous

        compose_ambiguous(self, n)

    # -- out of scope (must refuse) ------------------------------------------------------------

    def refusals(self, n: int, only: set[str] | None = None, per_subject: int = 2, salt: int = 300) -> None:
        """Refusal candidates. ``only`` restricts generation to the given requirement items (directed top-up)."""
        subjects: list[tuple[str, str, str, str, str, bool]] = []  # origin, ref, subject, why, anchor, probe?
        wanted = (lambda item: only is None or item in only)
        for refuse in self.req.must_refuse_topics:
            if wanted(f"refuse:{refuse.id}"):
                subjects.append(("must_refuse", refuse.id, refuse.topic, refuse.reason, refuse.topic, False))
        for gap in [g for g in self.req.gaps if g.recommended_action != "warn"]:
            if wanted(f"gap:{gap.id}"):
                subjects.append(("gap", gap.id, gap_subject(gap), GAP_WHY[gap.kind], gap_subject(gap), gap.kind == "unsupported_topic"))
        for tool in self._unattached_tools()[: 2 if only is None else 6]:
            if wanted(f"negtool:{tool}"):
                subjects.append(("negative_tool", tool, tool, "this assistant has no such capability", tool, True))
        for k, (origin, ref, subject, why, anchor, probe) in enumerate(subjects):
            prompt = "evals/negative_tool" if origin == "negative_tool" else "evals/refusal_question"
            variables = ({"purpose": self.req.purpose, "tool": tool_description(subject)} if origin == "negative_tool"
                         else {"purpose": self.req.purpose, "subject": subject, "why": why})
            for question in self._call(prompt, variables, QuestionsDraft, salt + k).questions[:per_subject]:  # type: ignore[attr-defined]
                self._add_refusal(question, origin, ref, subject, anchor, probe, prompt)
        if only is None:
            for topic in self.req.must_answer_topics[:3]:
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

    def tool_cases(self, n: int, items: list | None = None) -> None:
        """Lookups for table questions, and computed cases for every ask that needs a tool (``items`` = coverage items)."""
        caps = self.req.capabilities_needed
        directed = [i for i in (items or []) if i.kind == "ask_capability"]
        if "structured_lookup" in caps and self.corpus.tables:
            lookup_asks = [i.id for i in directed if i.capability == "structured_lookup"]
            self._lookups(max(1, math.ceil(n / 2)) if "arithmetic" in caps else n, covers=lookup_asks)
        for item in directed:
            self.computed(item.capability, item.text, [item.id], COMPUTE_ATTEMPTS)
        if not directed and "arithmetic" in caps:  # nothing in the brief points at a tool, but the requirements allow one
            self.computed("arithmetic", "", [], COMPUTE_ATTEMPTS)

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

    def _lookups(self, n: int, covers: list[str] | None = None) -> None:
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
                covers=list(covers or []), provenance=self._prov("evals/lookup_question"),
                validation={"expected_value_computed_by_code": True, "source": "table cell", "column": column})
            self.out.candidates.append(Candidate(case, [], group=f"lookup:{table.doc}"))

    TEMPLATES_FOR = {
        "arithmetic": ("discounted_product", "product", "sum", "difference"),
        "temporal": ("date_plus_days",),
        "structured_lookup": ("extreme",),
    }

    def computed(self, capability: str, focus: str, covers: list[str], attempts: int, salt: int = 600) -> None:
        """Computed cases: the model picks a template and binds its slots to menu facts; code verifies and computes."""
        names = [n for n in self.TEMPLATES_FOR.get(capability, ()) if n in TEMPLATES]
        if not self.corpus.tables:
            names = [n for n in names if n not in ("extreme", "date_plus_days")] if capability != "temporal" else []
        if not names:
            return
        today = "temporal" in self.req.capabilities_needed
        for i in range(attempts):
            menu = build_menu(self.corpus, self.rng, with_row=bool(self.corpus.tables), today=today)
            choice = self._call("evals/compute_template",
                                {"templates": template_help(names), "menu": menu.text, "focus": focus or "any calculation"},
                                TemplateChoice, salt + i)
            if choice.template not in names:  # type: ignore[attr-defined]
                self.out.reject("tool_required", choice.question, "template is not one of the offered templates",  # type: ignore[attr-defined]
                                f"{choice.template!r} not in {names}")  # type: ignore[attr-defined]
                continue
            try:
                result = instantiate(choice, menu, self.corpus)  # type: ignore[arg-type]
            except PlanRejected as exc:
                self.out.reject("tool_required", choice.question, "computed case failed verification", str(exc))  # type: ignore[attr-defined]
                continue
            kind = {"number": "number", "date": "date", "phrase": "phrase", "relative_date": "relative_date"}[result.answer_kind]
            case = EvalCase(
                id="", type="tool_required", question=result.question, expected_status="answered",
                expected_sources=result.docs, key_facts=[KeyFact(text=result.answer_text, kind=kind)],  # type: ignore[arg-type]
                seed_spans=result.spans, expected_tools=result.tools, origin="computed", covers=list(covers),
                provenance=self._prov("evals/compute_template") | {"template": result.template, "formula": result.formula,
                                                                   "values": result.values},
                validation={"expected_value_computed_by_code": True, "template": result.template,
                            "cross_source": result.cross_source, "needs_current_date": result.needs_current_date})
            self.out.candidates.append(Candidate(case, [], group=f"computed:{result.template}"))


def normalise_question(text: str) -> str:
    return normalize(text).lower()
