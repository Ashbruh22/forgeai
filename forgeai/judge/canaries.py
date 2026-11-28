"""Judge canaries (spec §6.2): gold answers built IN CODE from verified seed spans, then perturbed IN CODE.

No model writes a canary. A gold answer is the text of the case's seed spans, cited with a verbatim quote of the
chunk that overlaps each span. Each gold answer is perturbed six ways; the judge must catch the four that are wrong
and must not penalise the two that are right (a longer and a shorter way to say the same correct thing):

=================  ===========================================================  ===========
perturbation       what changes                                                  should be
=================  ===========================================================  ===========
number_swap        one number of the answer is replaced by one not in context    detected
negation           the statement is turned into its opposite                     detected
wrong_document     the answer is unchanged but cites another document            detected
fabricated_clause  an extra sentence the documents do not support                detected
padded_correct     filler plus true sentences from the cited chunk added         not flagged
concise_correct    only the key facts                                            not flagged
=================  ===========================================================  ===========

Detection is reported twice: for the whole pipeline (layer 1 + layer 2) and for layer 2 alone (is any claim of the
answer rated unsupported), because layer 1 is the same code for every judge and only layer 2 differs between them.
A wrong-document citation is a layer-1 check (the answer text is unchanged), so layer 2 is not scored on it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from forgeai.config import ModelSpec
from forgeai.evals.matching import NUMBER, numbers_in
from forgeai.evals.models import EvalCase
from forgeai.ingest.models import Chunk
from forgeai.ingest.spans import SeedSpan
from forgeai.judge.layer1 import _overlaps
from forgeai.judge.models import CaseRun, CitedChunk, RetrievedChunk
from forgeai.judge.score import score_case
from forgeai.runner.contract import Answer, Citation
from forgeai.textmatch import is_verbatim, normalize

PERTURBATIONS = ("number_swap", "number_swap_given", "negation", "wrong_document", "fabricated_clause", "padded_correct", "concise_correct")
SHOULD_BE_DETECTED = {"number_swap", "number_swap_given", "negation", "wrong_document", "fabricated_clause"}
LAYER2_SCORED = ("number_swap", "number_swap_given", "negation", "fabricated_clause", "padded_correct", "concise_correct")
GOLD = "gold"
FABRICATION = "In addition, a 37% surcharge applies to requests made after the first quarter of 2019, according to an internal addendum."
PADDING_OPEN = "Thank you for your question. Based on the documents I was given, here is what I found."
PADDING_CLOSE = "Please let me know if you would like more detail or a different comparison."
_AUXILIARIES = ("is", "are", "may", "must", "can", "will", "should", "applies", "apply")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_LABEL = re.compile(r"^\s*([A-Za-z_][\w ]*?)\s*:\s*(.+)$")


@dataclass
class Canary:
    id: str
    demo: str
    kind: str  # GOLD or one of PERTURBATIONS
    case: EvalCase
    run: CaseRun
    should_pass: bool


def _rank(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()


def _best_chunk(span: SeedSpan, chunks: list[Chunk]) -> Chunk | None:
    over = [c for c in chunks if _overlaps(c.doc, c.char_start, c.char_end, span)]
    return max(over, key=lambda c: min(c.char_end, span.char_end) - max(c.char_start, span.char_start), default=None)  # type: ignore[type-var,operator]


def _quote(span: SeedSpan, chunk: Chunk) -> str:
    """A verbatim quote of ``chunk`` that carries the span's content."""
    if is_verbatim(span.span_text, chunk.text):
        return normalize(span.span_text)[:300]
    for sentence in _SENTENCE.split(chunk.text):
        if is_verbatim(sentence, chunk.text) and normalize(sentence) in normalize(span.span_text):
            return normalize(sentence)
    return normalize(chunk.text)[:200]


def _as_chunk(chunk: Chunk) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk.chunk_id, doc=chunk.doc, char_start=chunk.char_start, char_end=chunk.char_end, text=chunk.text)


def _answering_text(case: EvalCase, span: SeedSpan) -> str:
    """The span as a gold answer would use it. A table row ("a: 1 | b: 2 | ...") is cut to the cell(s) that carry the
    key facts plus the row's first cell (which names the row), because a whole row is not an answer to a question."""
    text = normalize(span.span_text)
    fields = [f.strip() for f in text.split(" | ")]
    if len(fields) < 3 or not all(_LABEL.match(f) for f in fields):
        return text
    values = [(_LABEL.match(f).group(2) or "").lower() for f in fields]  # type: ignore[union-attr]
    wanted = [fact.text.lower().split(": ")[-1] for fact in case.key_facts]
    keep = [f for f, v in zip(fields, values, strict=True) if any(w and (w in v or v in w) for w in wanted)]
    return " | ".join([fields[0], *[f for f in keep if f != fields[0]]]) if keep else " | ".join(fields[:3])


def build_gold(case: EvalCase, chunks: list[Chunk]) -> tuple[CaseRun, list[Chunk]] | None:
    """The gold answer of ``case``: its seed-span text, cited with verbatim quotes. None if a span has no chunk."""
    gold_chunks: list[Chunk] = []
    citations: list[Citation] = []
    pieces: list[str] = []
    for span in case.seed_spans:
        chunk = _best_chunk(span, chunks)
        if chunk is None:
            return None
        gold_chunks.append(chunk)
        citations.append(Citation(doc=chunk.doc, chunk_id=chunk.chunk_id, quote=_quote(span, chunk)))
        pieces.append(_answering_text(case, span))
    if not pieces:
        return None
    distractors = sorted((c for c in chunks if c.chunk_id not in {g.chunk_id for g in gold_chunks} and c.kind != "schema"),
                         key=lambda c: _rank(case.id, c.chunk_id))[:2]
    answer = Answer(status="answered", answer=" ".join(pieces)[:900], citations=citations)
    run = CaseRun(case_id=case.id, answer=answer, retrieved=[_as_chunk(c) for c in [*gold_chunks, *distractors]],
                  cited=[CitedChunk(chunk_id=c.chunk_id, doc=c.doc, char_start=c.char_start, char_end=c.char_end) for c in gold_chunks])
    return run, gold_chunks


def _context_numbers(run: CaseRun) -> set[float]:
    return {n for chunk in run.retrieved for n in numbers_in(chunk.text)}


def number_swap(case: EvalCase, gold: CaseRun) -> CaseRun | None:
    """Replace one number of the answer, preferring a number that is a key fact, by one that is nowhere in context."""
    text = gold.answer.answer
    wanted = {n for f in case.key_facts if f.kind == "number" for n in numbers_in(f.text)}
    tokens = [m for m in NUMBER.finditer(text) if m.group(0).replace(",", "").replace(".", "", 1).isdigit()]
    tokens.sort(key=lambda m: float(m.group(0).replace(",", "")) not in wanted)
    seen = _context_numbers(gold)
    for match in tokens:
        value = float(match.group(0).replace(",", ""))
        for candidate in (value + 7, value + 13, value * 2 + 1, value + 101):
            if candidate not in seen and candidate != value:
                shown = f"{candidate:g}" if "." not in match.group(0) else f"{candidate:.{len(match.group(0).split('.')[1])}f}"
                swapped = text[: match.start()] + shown + text[match.end():]
                return gold.model_copy(update={"answer": gold.answer.model_copy(update={"answer": swapped})})
    return None


def number_swap_given(case: EvalCase, gold: CaseRun) -> CaseRun | None:
    """Replace a number of the answer by a DIFFERENT number the question contains (a code or a quantity the user wrote).
    Counting the question's own numbers as given must not let this through: the number is in the wrong relation."""
    given = [float(m.group(0).replace(",", "")) for m in NUMBER.finditer(case.question)]  # codes' digits included
    text = gold.answer.answer
    wanted = {n for f in case.key_facts if f.kind == "number" for n in numbers_in(f.text)}
    tokens = [m for m in NUMBER.finditer(text) if m.group(0).replace(",", "").replace(".", "", 1).isdigit()]
    tokens.sort(key=lambda m: float(m.group(0).replace(",", "")) not in wanted)
    for match in tokens:
        value = float(match.group(0).replace(",", ""))
        for candidate in given:
            if candidate != value:
                swapped = text[: match.start()] + f"{candidate:g}" + text[match.end():]
                return gold.model_copy(update={"answer": gold.answer.model_copy(update={"answer": swapped})})
    return None


def negation(gold: CaseRun) -> CaseRun:
    text = gold.answer.answer
    for aux in _AUXILIARIES:
        flipped, count = re.subn(rf"(?<!\w){aux}(?!\w)", f"{aux} not", text, count=1)
        if count:
            return gold.model_copy(update={"answer": gold.answer.model_copy(update={"answer": flipped})})
    return gold.model_copy(update={"answer": gold.answer.model_copy(update={"answer": "It is not true that " + text})})


def fabricated_clause(gold: CaseRun) -> CaseRun:
    return gold.model_copy(update={"answer": gold.answer.model_copy(update={"answer": f"{gold.answer.answer} {FABRICATION}"})})


def wrong_document(case: EvalCase, gold: CaseRun, chunks: list[Chunk]) -> CaseRun | None:
    expected = {s.doc for s in case.seed_spans}
    others = sorted((c for c in chunks if c.doc not in expected and c.kind == "prose" and len(c.text.split()) >= 12),
                    key=lambda c: _rank(case.id, c.chunk_id))
    if not others:
        return None
    chunk = others[0]
    quote = normalize(chunk.text)[:120].rsplit(" ", 1)[0]
    answer = gold.answer.model_copy(update={"citations": [Citation(doc=chunk.doc, chunk_id=chunk.chunk_id, quote=quote)]})
    return gold.model_copy(update={
        "answer": answer, "retrieved": [*gold.retrieved, _as_chunk(chunk)],
        "cited": [CitedChunk(chunk_id=chunk.chunk_id, doc=chunk.doc, char_start=chunk.char_start, char_end=chunk.char_end)]})


def padded_correct(gold: CaseRun, gold_chunks: list[Chunk]) -> CaseRun:
    """Filler around the gold answer plus up to two TRUE sentences of the cited chunks that it does not already say."""
    have = normalize(gold.answer.answer)
    extra: list[str] = []
    for chunk in gold_chunks:
        for sentence in _SENTENCE.split(normalize(chunk.text)):
            if len(sentence.split()) >= 5 and sentence not in have and sentence not in extra and len(extra) < 2:
                extra.append(sentence)
    text = " ".join([PADDING_OPEN, gold.answer.answer, *extra, PADDING_CLOSE])
    return gold.model_copy(update={"answer": gold.answer.model_copy(update={"answer": text})})


def concise_correct(case: EvalCase, gold: CaseRun) -> CaseRun | None:
    values = []
    for fact in case.key_facts:
        match = _LABEL.match(fact.text)
        values.append(match.group(2) if match and "_" in match.group(1) else fact.text)
    if not values:
        return None
    return gold.model_copy(update={"answer": gold.answer.model_copy(update={"answer": "; ".join(values)})})


def pick_golds(cases: list[EvalCase], per_demo: int) -> list[EvalCase]:
    """Dev cases with a number in their span (so every perturbation applies): one multi-document case, then spread."""
    pool = [c for c in cases if c.split == "dev" and c.type in ("answerable", "multidoc") and c.key_facts and c.seed_spans
            and any(numbers_in(s.span_text) for s in c.seed_spans) and all(len(s.span_text) < 600 for s in c.seed_spans)]
    chosen = [c for c in pool if c.type == "multidoc"][:1]
    rest = [c for c in pool if c not in chosen]
    step = max(1, len(rest) // max(per_demo - len(chosen), 1))
    return (chosen + rest[::step])[:per_demo]


def build_canaries(demo: str, cases: list[EvalCase], chunks: list[Chunk], per_demo: int = 5) -> list[Canary]:
    """Every gold answer of this demo plus its perturbations (a perturbation that does not apply is skipped)."""
    out: list[Canary] = []
    for case in pick_golds(cases, per_demo):
        built = build_gold(case, chunks)
        if built is None:
            continue
        gold, gold_chunks = built
        variants = {
            GOLD: gold, "number_swap": number_swap(case, gold), "number_swap_given": number_swap_given(case, gold), "negation": negation(gold),
            "wrong_document": wrong_document(case, gold, chunks), "fabricated_clause": fabricated_clause(gold),
            "padded_correct": padded_correct(gold, gold_chunks), "concise_correct": concise_correct(case, gold)}
        for kind, run in variants.items():
            if run is not None:
                out.append(Canary(f"{demo}/{case.id}/{kind}", demo, kind, case, run, should_pass=kind not in SHOULD_BE_DETECTED))
    return out


def run_canaries(canaries: list[Canary], judge: ModelSpec, *, today: date, fuzzy_min: int = 85, nli: Any = None,
                 nli_min: float = 0.7) -> dict[str, Any]:
    """Score every canary with ``judge`` and tabulate detection accuracy per perturbation type."""
    items: list[dict[str, Any]] = []
    for canary in canaries:
        score = score_case(canary.case, canary.run, judge, today=today, fuzzy_min=fuzzy_min, nli=nli, nli_min=nli_min)
        clean = score.layer2.unsupported == 0 and not score.layer2.refusal_claims
        items.append({"id": canary.id, "kind": canary.kind, "should_pass": canary.should_pass,
                      "pipeline_ok": score.passed == canary.should_pass,
                      "layer2_ok": (clean == canary.should_pass) if canary.kind in (*LAYER2_SCORED, GOLD) else None,
                      "passed": score.passed, "reasons": score.reasons, "answer": canary.run.answer.answer,
                      "claims": [c.model_dump() for c in score.layer2.claims]})
    table: dict[str, Any] = {}
    for kind in (GOLD, *PERTURBATIONS):
        group = [i for i in items if i["kind"] == kind]
        scored2 = [i for i in group if i["layer2_ok"] is not None]
        table[kind] = {
            "n": len(group), "pipeline_accuracy": round(sum(i["pipeline_ok"] for i in group) / len(group), 3) if group else None,
            "layer2_accuracy": round(sum(i["layer2_ok"] for i in scored2) / len(scored2), 3) if scored2 else None}
    return {"judge": judge.name, "nli": bool(nli), "table": table, "items": items}


def layer2_mean(result: dict[str, Any]) -> float:
    """Macro-average of the layer-2 accuracy over the perturbation types layer 2 is responsible for."""
    values = [result["table"][k]["layer2_accuracy"] for k in LAYER2_SCORED if result["table"][k]["layer2_accuracy"] is not None]
    return sum(values) / len(values) if values else 0.0


def low_confidence_types(result: dict[str, Any], minimum: float) -> list[str]:
    """Perturbation types the judge detects below ``minimum`` accuracy: its layer-2 verdicts on them are low confidence."""
    return [k for k in LAYER2_SCORED if (result["table"][k]["layer2_accuracy"] or 0.0) < minimum]


def choose_judge(results: dict[str, dict[str, Any]], minimum: float) -> dict[str, Any]:
    """The judge with the best macro layer-2 accuracy; ties go to the one with fewer low-confidence types, then to the
    better whole-pipeline accuracy. Returns the choice and the numbers it rests on."""
    def key(name: str) -> tuple[float, int, float]:
        res = results[name]
        pipeline = [v["pipeline_accuracy"] for v in res["table"].values() if v["pipeline_accuracy"] is not None]
        return (round(layer2_mean(res), 3), -len(low_confidence_types(res, minimum)), sum(pipeline) / len(pipeline))

    ranked = sorted(results, key=key, reverse=True)
    chosen = ranked[0]
    return {"chosen": chosen, "ranking": [{"judge": n, "layer2_macro_accuracy": round(layer2_mean(results[n]), 3),
                                           "low_confidence_types": low_confidence_types(results[n], minimum)} for n in ranked],
            "minimum": minimum, "low_confidence_types": low_confidence_types(results[chosen], minimum)}
