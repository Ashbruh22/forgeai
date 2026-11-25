"""Groundability guard (spec §4): can the corpus actually support each must-answer topic?

For every topic the guard asks the model for 3 probe questions, retrieves chunks for each,
and demands a *verbatim* supporting span. The span is then verified in code: it must be a
substring of a retrieved chunk and share a content term with the topic. A model claim
without a verified span counts as unsupported.

The same probes drive the ``min_score`` calibration (spec §4.4). Only guard probes are
used there, never eval cases.
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
from forgeai.runner.retrieval import Hit, HybridIndex
from forgeai.textmatch import content_terms, is_verbatim, normalize

PROBES_PER_TOPIC = 3
NEAR_MISS_PER_TOPIC = 2
GUARD_TOP_K = 5
MIN_SPAN_WORDS = 3  # prose; table rows are compact, so they only need MIN_ROW_SPAN_WORDS
MIN_ROW_SPAN_WORDS = 2
SAMPLE_CHUNKS = 3
SAMPLE_CHARS = 350
OVERLAP_MARGIN = 0.05
DEFAULT_MIN_SCORE = 0.25
BOUND_MARGIN = 0.10
_DEFAULTS = yaml.safe_load((Path(__file__).parents[1] / "intake" / "defaults.yaml").read_text(encoding="utf-8"))


class Probes(BaseModel):
    questions: list[str] = Field(default_factory=list)


class Support(BaseModel):
    chunk_id: str = ""
    span: str = "NONE"


@dataclass
class ProbeResult:
    question: str
    supported: bool
    chunk_id: str = ""
    span: str = ""
    max_dense: float = 0.0
    reason: str = ""


@dataclass
class TopicCoverage:
    topic_id: str
    topic: str
    verdict: Coverage
    probes: list[ProbeResult] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


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


def check_probe(
    question: str, topic: str, index: HybridIndex, config: ForgeConfig
) -> ProbeResult:
    """Extractive support check for one question; the model's claim is verified in code."""
    max_dense = index.max_dense(question)
    hits = index.search(question, GUARD_TOP_K, 0.6, 0.4, min_score=0.0)
    if not hits:
        return ProbeResult(question, False, max_dense=max_dense, reason="nothing retrieved")
    model = config.agent_model
    support = call_structured(
        "guard.extract", "guard/extract_support",
        {"question": question, "excerpts": _format_excerpts(hits)},
        Support, model.name, 0.0, seed=model.seed, fallback=Support(),
    )
    span = support.span.strip()
    if not span or span.upper() == "NONE":
        return ProbeResult(question, False, max_dense=max_dense, reason="model found no supporting span")
    chunk = next((h.chunk for h in hits if h.chunk.chunk_id == support.chunk_id), None)
    if chunk is None:  # model named no (or an unretrieved) chunk: try to locate the span itself
        chunk = next((h.chunk for h in hits if is_verbatim(span, h.chunk.text)), None)
    if chunk is None or not is_verbatim(span, chunk.text):
        return ProbeResult(question, False, support.chunk_id, span, max_dense, "span is not a verbatim quote of a retrieved chunk")
    minimum = MIN_ROW_SPAN_WORDS if chunk.kind in ("row", "schema") else MIN_SPAN_WORDS
    if len(re.findall(r"\w+", span)) < minimum:
        return ProbeResult(question, False, chunk.chunk_id, span, max_dense, "span too short to be an answer")
    if not span_is_relevant(topic, span, chunk):
        return ProbeResult(question, False, chunk.chunk_id, span, max_dense, "span shares no content term with the topic")
    return ProbeResult(question, True, chunk.chunk_id, normalize(span), max_dense, "verified")


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


def run_guard(
    topics: list[tuple[str, str]], purpose: str, index: HybridIndex, config: ForgeConfig
) -> tuple[list[TopicCoverage], Calibration]:
    """Probe every ``(topic_id, topic)``; return per-topic coverage and the calibration."""
    coverage: list[TopicCoverage] = []
    in_scope: list[float] = []
    near_miss: list[float] = []
    for topic_id, topic in topics:
        results = [check_probe(q, topic, index, config) for q in probe_questions(topic, purpose, index, config)]
        verified = [r for r in results if r.supported]
        verdict = verdict_for(len(verified))
        evidence = list(dict.fromkeys(r.chunk_id for r in verified))
        coverage.append(TopicCoverage(topic_id, topic, verdict, results, evidence))
        if verdict == "UNSUPPORTED":
            near_miss.extend(r.max_dense for r in results)  # verified-unsupported probes
        else:
            in_scope.extend(r.max_dense for r in verified)
            for question in near_miss_questions(topic, purpose, index, config):
                result = check_probe(question, topic, index, config)
                if not result.supported:  # keep only questions the corpus verifiably cannot answer
                    near_miss.append(result.max_dense)
    return coverage, calibrate_min_score(in_scope, near_miss)
