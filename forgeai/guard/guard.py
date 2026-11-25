"""Groundability guard (spec §4): can the corpus actually support each must-answer topic?

For every topic the guard asks the model for 3 probe questions, retrieves chunks for each and
demands a *verbatim* supporting span. A probe counts as supported only if TWO independent
checks pass:

1. **Code check**: the span is a verbatim substring of a retrieved chunk, is long enough, and
   shares a content term with the topic (a drifting probe never counts as evidence).
2. **Judge check**: a model from a different family says whether the span actually answers the
   question, and must quote a phrase from the span; that quote is verified in code too.

A topic is SUPPORTED if at least 2 of its 3 probes pass both checks. Topics that are not fully
supported are split into narrower aspects, so the supported part is kept and only the
unsupported part becomes a gap. How often the two checks disagree is reported.

The same probes drive the ``min_score`` calibration (spec §4.4). Only guard probes are
used there, never eval cases.

Calls are batched by stage (all agent-model calls, then all judge-model calls) so that an
8 GB GPU swaps models a handful of times instead of once per probe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from forgeai.config import ForgeConfig
from forgeai.ingest.models import Chunk
from forgeai.intake.schema import Coverage
from forgeai.llm import call_structured
from forgeai.prompts import load_prompt
from forgeai.runner.retrieval import Hit, HybridIndex
from forgeai.textmatch import content_terms, is_verbatim, normalize

PROBES_PER_TOPIC = 3
NEAR_MISS_PER_TOPIC = 2
MAX_ASPECTS = 3
GUARD_TOP_K = 5
MIN_SPAN_WORDS = 3  # prose; table rows are compact, so they only need MIN_ROW_SPAN_WORDS
MIN_ROW_SPAN_WORDS = 2
SAMPLE_CHUNKS = 3
SAMPLE_CHARS = 350
DEFAULT_MIN_SCORE = 0.25
BOUND_MARGIN = 0.10
OVERLAP_MARGIN = 0.05
_DEFAULTS = yaml.safe_load((Path(__file__).parents[1] / "intake" / "defaults.yaml").read_text(encoding="utf-8"))


class Probes(BaseModel):
    questions: list[str] = Field(default_factory=list)


class Support(BaseModel):
    chunk_id: str = ""
    span: str = "NONE"


class JudgeSupport(BaseModel):
    # Field order matters for constrained decoding: reason, then the evidence, then the verdict.
    reason: str = ""
    quote: str = ""
    answers: bool = False


class Aspect(BaseModel):
    aspect: str
    needs_live_values: bool = False


class TopicSplit(BaseModel):
    aspects: list[Aspect] = Field(default_factory=list)


@dataclass
class ProbeResult:
    question: str
    supported: bool = False  # both checks passed
    chunk_id: str = ""
    span: str = ""
    max_dense: float = 0.0  # best similarity of the question to ANY chunk (for near-miss calibration)
    evidence_dense: float = 0.0  # similarity of the chunk that held the span (for in-scope calibration)
    reason: str = ""
    code_ok: bool = False  # check 1: verbatim substring, long enough, shares a topic term
    substring_ok: bool = False
    judge_ok: bool | None = None  # check 2; None when there was no verbatim span to judge
    judge_quote: str = ""
    context: str = ""  # the full chunk the span comes from; shown to the judge so a table fragment is not read in isolation


@dataclass
class AspectCoverage:
    aspect: str
    verdict: Coverage | None  # None for aspects that need live values (never probed)
    needs_live_values: bool
    probes: list[ProbeResult] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


@dataclass
class TopicCoverage:
    topic_id: str
    topic: str
    verdict: Coverage
    probes: list[ProbeResult] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    aspects: list[AspectCoverage] = field(default_factory=list)

    @property
    def supported_aspects(self) -> list[AspectCoverage]:
        return [a for a in self.aspects if a.verdict == "SUPPORTED"]

    @property
    def effective_verdict(self) -> Coverage:
        """The topic's verdict after splitting: a supported part rescues an unsupported whole."""
        if not self.supported_aspects:
            return self.verdict
        whole = all(a.verdict == "SUPPORTED" and not a.needs_live_values for a in self.aspects)
        return "SUPPORTED" if self.verdict == "SUPPORTED" and whole else "PARTIAL"


@dataclass
class Calibration:
    threshold: float
    bounds: tuple[float, float]
    separable: bool
    calibrated: bool
    in_scope_maxima: list[float]
    near_miss_maxima: list[float]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_score": round(self.threshold, 4), "min_score_bounds": [round(b, 4) for b in self.bounds],
            "separable": self.separable, "calibrated": self.calibrated, "note": self.note,
            "in_scope_probe_max_similarity": [round(x, 4) for x in self.in_scope_maxima],
            "near_miss_probe_max_similarity": [round(x, 4) for x in self.near_miss_maxima],
        }


@dataclass
class GuardResult:
    coverage: list[TopicCoverage]
    calibration: Calibration
    agreement: dict[str, Any]


# ---- check 1: the code check ---------------------------------------------------------------


def span_is_relevant(topic: str, span: str, chunk: Chunk) -> bool:
    """The verified span (or its heading) must mention a content term of the TOPIC.

    The probe is deliberately not used here: a probe that drifted to another subject would
    share terms with whatever span it retrieved and make unrelated text look like support
    (found on the first grounded-probe run, see docs/DECISIONS.md). A table row is one atomic
    record, so for rows the whole row counts, not just the quoted fragment.
    """
    wanted = content_terms(topic)
    haystack = span + " " + " ".join(chunk.heading_path)
    if chunk.kind == "row":
        haystack += " " + chunk.text
    return not wanted or bool(wanted & content_terms(haystack))


def _format_excerpts(hits: list[Hit]) -> str:
    return "\n\n".join(f"[chunk_id: {h.chunk.chunk_id}]\n{h.chunk.text}" for h in hits)


def extract_span(question: str, topic: str, index: HybridIndex, config: ForgeConfig) -> ProbeResult:
    """Agent-model extraction plus the code check. The model's claim is verified, never trusted."""
    scores = index.dense_scores(question)
    max_dense = float(scores.max()) if len(scores) else 0.0
    hits = index.search(question, GUARD_TOP_K, 0.6, 0.4, min_score=0.0)
    if not hits:
        return ProbeResult(question, max_dense=max_dense, reason="nothing retrieved")
    model = config.agent_model
    support = call_structured(
        "guard.extract", "guard/extract_support",
        {"question": question, "excerpts": _format_excerpts(hits)},
        Support, model.name, 0.0, seed=model.seed, fallback=Support(),
    )
    span = support.span.strip()
    if not span or span.upper() == "NONE":
        return ProbeResult(question, max_dense=max_dense, reason="model found no supporting span")
    chunk = next((h.chunk for h in hits if h.chunk.chunk_id == support.chunk_id), None)
    if chunk is None:  # model named no (or an unretrieved) chunk: try to locate the span itself
        chunk = next((h.chunk for h in hits if is_verbatim(span, h.chunk.text)), None)
    if chunk is None or not is_verbatim(span, chunk.text):
        return ProbeResult(question, chunk_id=support.chunk_id, span=span, max_dense=max_dense,
                           reason="span is not a verbatim quote of a retrieved chunk")
    result = ProbeResult(question, chunk_id=chunk.chunk_id, span=normalize(span), max_dense=max_dense,
                         evidence_dense=index.dense_of(scores, chunk.chunk_id), substring_ok=True, context=chunk.text)
    minimum = MIN_ROW_SPAN_WORDS if chunk.kind in ("row", "schema") else MIN_SPAN_WORDS
    if len(re.findall(r"\w+", span)) < minimum:
        result.reason = "span too short to be an answer"
    elif not span_is_relevant(topic, span, chunk):
        result.reason = "span shares no content term with the topic"
    else:
        result.code_ok = True
        result.reason = "code check passed"
    return result


# ---- check 2: the judge check --------------------------------------------------------------


def judge_spans(results: list[ProbeResult], config: ForgeConfig) -> None:
    """Judge-model check on every verbatim span (all of them, so disagreement is measurable)."""
    model = config.judge_model
    for result in results:
        if not result.substring_ok:
            continue
        verdict = call_structured(
            "guard.judge", "guard/judge_support",
            {"question": result.question, "span": result.span, "context": result.context},
            JudgeSupport, model.name, model.temperature, seed=model.seed, fallback=JudgeSupport(),
        )
        quote_ok = bool(verdict.quote.strip()) and is_verbatim(verdict.quote, result.span)
        result.judge_ok = verdict.answers and quote_ok
        result.judge_quote = verdict.quote.strip()
        if result.code_ok and not result.judge_ok:
            result.reason = ("judge says the span does not answer the question" if not verdict.answers
                             else "judge's supporting quote is not a verbatim part of the span")
        result.supported = result.code_ok and bool(result.judge_ok)
        if result.supported:
            result.reason = "verified by both checks"


def evaluate_probes(pairs: list[tuple[str, str]], index: HybridIndex, config: ForgeConfig) -> list[ProbeResult]:
    """Run ``(topic, question)`` pairs: every extraction first (agent model), then every judgement."""
    results = [extract_span(question, topic, index, config) for topic, question in pairs]
    judge_spans(results, config)
    return results


def check_probe_pair(topic: str, question: str, index: HybridIndex, config: ForgeConfig) -> ProbeResult:
    """Both checks for a single ``(topic, question)``."""
    return evaluate_probes([(topic, question)], index, config)[0]


def agreement_stats(results: list[ProbeResult]) -> dict[str, Any]:
    """How often the code check and the judge check disagree, over probes that had a real span."""
    judged = [r for r in results if r.substring_ok and r.judge_ok is not None]
    both = sum(1 for r in judged if r.code_ok and r.judge_ok)
    code_only = sum(1 for r in judged if r.code_ok and not r.judge_ok)
    judge_only = sum(1 for r in judged if r.judge_ok and not r.code_ok)
    neither = len(judged) - both - code_only - judge_only
    return {
        "probes_total": len(results), "probes_with_verbatim_span": len(judged),
        "both_pass": both, "code_check_only": code_only, "judge_only": judge_only, "neither": neither,
        "disagreement_rate": round((code_only + judge_only) / len(judged), 3) if judged else None,
    }


# ---- probes, verdicts, splitting -----------------------------------------------------------


def verdict_for(verified: int) -> Coverage:
    return "SUPPORTED" if verified >= 2 else "PARTIAL" if verified == 1 else "UNSUPPORTED"


def _samples(topic: str, index: HybridIndex) -> str:
    """A few real excerpts that mention the topic, so probes use the corpus's own names and codes."""
    hits = index.search(topic, SAMPLE_CHUNKS, 0.6, 0.4, min_score=0.0)
    return "\n".join(f"- {h.chunk.text[:SAMPLE_CHARS]}" for h in hits) or "(none)"


def on_topic(questions: list[str], topic: str) -> list[str]:
    """Keep only probes that share a content term with the topic (no drifting to other subjects)."""
    wanted = content_terms(topic)
    return [q for q in questions if not wanted or wanted & content_terms(q)]


def probe_questions(topic: str, purpose: str, index: HybridIndex, config: ForgeConfig) -> list[str]:
    model = config.agent_model
    fallback = Probes(questions=[t.format(topic=topic) for t in _DEFAULTS["probe_templates"]])
    result = call_structured(
        "guard.probes", "guard/probes",
        {"purpose": purpose, "topic": topic, "samples": _samples(topic, index)},
        Probes, model.name, model.temperature, seed=model.seed, fallback=fallback,
    )
    questions = on_topic([q.strip() for q in result.questions if q.strip()], topic)[:PROBES_PER_TOPIC]
    for template in _DEFAULTS["probe_templates"]:  # top up deterministically if too few were on topic
        if len(questions) >= PROBES_PER_TOPIC:
            break
        questions.append(template.format(topic=topic))
    return questions


def near_miss_questions(topic: str, purpose: str, index: HybridIndex, config: ForgeConfig) -> list[str]:
    model = config.agent_model
    result = call_structured(
        "guard.near_miss", "guard/near_miss",
        {"purpose": purpose, "topic": topic, "samples": _samples(topic, index)},
        Probes, model.name, model.temperature, seed=model.seed, fallback=Probes(),
    )
    return [q.strip() for q in result.questions if q.strip()][:NEAR_MISS_PER_TOPIC]


def split_topic(
    topic: str, purpose: str, index: HybridIndex, config: ForgeConfig, live: bool = False
) -> list[Aspect]:
    """Ask the model to split a topic into narrower aspects (generic: no domain knowledge).

    For a topic tied to a live-data gap the model is told to separate what the documents RECORD
    from the live value, which is the split the snapshot rule needs.
    """
    model = config.agent_model
    instruction = load_prompt("guard/split_live_instruction").user if live else "(none)"
    result = call_structured(
        "guard.split", "guard/split_topic",
        {"purpose": purpose, "topic": topic, "samples": _samples(topic, index), "instruction": instruction},
        TopicSplit, model.name, model.temperature, seed=model.seed, fallback=TopicSplit(),
    )
    seen: dict[str, Aspect] = {}
    for aspect in result.aspects:
        text = aspect.aspect.strip()
        if text and text.lower() != topic.lower():
            seen.setdefault(text.lower(), Aspect(aspect=text, needs_live_values=aspect.needs_live_values))
    return list(seen.values())[:MAX_ASPECTS]


def on_topic_aspects(aspects: list[Aspect], topic: str) -> list[Aspect]:
    """Drop aspects that share no content term with the parent topic (the model drifted)."""
    wanted = content_terms(topic)
    return [a for a in aspects if not wanted or wanted & content_terms(a.aspect)]


def probe_topics(
    topics: list[tuple[str, str]], purpose: str, index: HybridIndex, config: ForgeConfig,
    with_near_miss: bool = True, anchors: dict[str, str] | None = None,
) -> tuple[list[TopicCoverage], list[float], list[float], list[ProbeResult]]:
    """Probe each ``(id, topic)``. Returns coverage, in-scope maxima, near-miss maxima, all results.

    ``anchors`` maps an id to the text its spans must be relevant to. A split aspect is anchored to
    its PARENT topic, so an aspect the model invented that drifted off the parent cannot be
    "supported" by text about something else.
    """
    anchors = anchors or {}
    plan: list[tuple[str, str, str, bool]] = []  # (topic_id, topic, question, is_near_miss)
    for topic_id, topic in topics:  # agent model: all probe generation first
        plan += [(topic_id, topic, q, False) for q in probe_questions(topic, purpose, index, config)]
        if with_near_miss:
            plan += [(topic_id, topic, q, True) for q in near_miss_questions(topic, purpose, index, config)]
    results = evaluate_probes([(anchors.get(tid, topic), question) for tid, topic, question, _ in plan], index, config)

    coverage: list[TopicCoverage] = []
    in_scope: list[float] = []
    near_miss: list[float] = []
    for topic_id, topic in topics:
        own = [r for (tid, _, _, miss), r in zip(plan, results, strict=True) if tid == topic_id and not miss]
        misses = [r for (tid, _, _, miss), r in zip(plan, results, strict=True) if tid == topic_id and miss]
        verified = [r for r in own if r.supported]
        verdict = verdict_for(len(verified))
        coverage.append(TopicCoverage(topic_id, topic, verdict, own,
                                      list(dict.fromkeys(r.chunk_id for r in verified))))
        if verdict == "UNSUPPORTED":
            near_miss.extend(r.max_dense for r in own)  # verified-unsupported probes
        else:
            in_scope.extend(r.evidence_dense for r in verified)
            near_miss.extend(r.max_dense for r in misses if not r.supported)
    return coverage, in_scope, near_miss, results


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def calibrate_min_score(in_scope: list[float], near_miss: list[float]) -> Calibration:
    """Place ``min_score`` between in-scope and near-miss max similarities (spec §4.4)."""
    if not in_scope or not near_miss:
        threshold = DEFAULT_MIN_SCORE
        note = "uncalibrated: needs at least one in-scope and one near-miss probe; default kept"
        separable, calibrated = False, False
    else:
        low_in, high_out = percentile(in_scope, 0.10), percentile(near_miss, 0.90)
        if low_in > high_out:
            threshold, separable = (low_in + high_out) / 2, True
            note = f"separable: in-scope p10={low_in:.3f} > near-miss p90={high_out:.3f}"
        else:
            # Same-subject questions the corpus cannot answer look just like answerable ones to a
            # dense embedder, so no threshold separates them. Keep recall: never gate out the best
            # chunk of the answerable probes, and leave unanswerable ones to the refusal policy.
            threshold, separable = max(0.05, low_in - OVERLAP_MARGIN), False
            note = (f"LOW CONFIDENCE: distributions overlap (in-scope p10={low_in:.3f}, "
                    f"near-miss p90={high_out:.3f}); threshold set {OVERLAP_MARGIN} below the in-scope "
                    f"p10 to preserve recall; unanswerable near-misses rely on the refusal policy")
        calibrated = True
    bounds = (max(0.05, threshold - BOUND_MARGIN), min(0.95, threshold + BOUND_MARGIN))
    return Calibration(threshold, bounds, separable, calibrated, in_scope, near_miss, note)


def split_unsupported_topics(
    coverage: list[TopicCoverage], purpose: str, index: HybridIndex, config: ForgeConfig,
    live_topic_ids: frozenset[str] = frozenset(),
) -> list[ProbeResult]:
    """Split topics into aspects and probe the non-live ones.

    Every topic that is not fully supported is split. A topic tied to a live-data gap
    (``live_topic_ids``, found by rules from the brief) is split even if its probes were
    verified, because a static snapshot supports the recorded part of it, never the live part.
    The model's ``needs_live_values`` flag is honoured only for those live-tied topics; on any
    other topic it is ignored, so a loose label cannot turn a missing document into "live data".
    """
    pending: list[tuple[TopicCoverage, Aspect]] = []
    for topic in coverage:
        if topic.verdict == "SUPPORTED" and topic.topic_id not in live_topic_ids:
            continue
        live = topic.topic_id in live_topic_ids
        for aspect in on_topic_aspects(split_topic(topic.topic, purpose, index, config, live), topic.topic):
            if topic.topic_id not in live_topic_ids:
                aspect = aspect.model_copy(update={"needs_live_values": False})
            pending.append((topic, aspect))
    probed = [(t, a) for t, a in pending if not a.needs_live_values]
    aspect_cov, _, _, results = probe_topics(
        [(f"{t.topic_id}/{i}", a.aspect) for i, (t, a) in enumerate(probed)], purpose, index, config,
        with_near_miss=False, anchors={f"{t.topic_id}/{i}": t.topic for i, (t, _) in enumerate(probed)},
    )
    by_key = {c.topic_id: c for c in aspect_cov}
    for i, (topic, aspect) in enumerate(probed):
        cov = by_key[f"{topic.topic_id}/{i}"]
        topic.aspects.append(AspectCoverage(aspect.aspect, cov.verdict, False, cov.probes, cov.evidence))
    for topic, aspect in pending:
        if aspect.needs_live_values:
            topic.aspects.append(AspectCoverage(aspect.aspect, None, True))
    return results


def run_guard(
    topics: list[tuple[str, str]], purpose: str, index: HybridIndex, config: ForgeConfig,
    live_topic_ids: frozenset[str] = frozenset(),
) -> GuardResult:
    """Probe every ``(topic_id, topic)``, split the unsupported ones, calibrate ``min_score``."""
    coverage, in_scope, near_miss, results = probe_topics(topics, purpose, index, config)
    results += split_unsupported_topics(coverage, purpose, index, config, live_topic_ids)
    return GuardResult(coverage, calibrate_min_score(in_scope, near_miss), agreement_stats(results))
