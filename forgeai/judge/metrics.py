"""Scorecard metrics (spec §6.4), computed per split from the per-case scores."""

from __future__ import annotations

import math
import statistics
from typing import Any

from forgeai.judge.models import CaseScore


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval of a pass rate (0, 0 for no cases)."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _rate(successes: int, n: int) -> dict[str, Any]:
    low, high = wilson(successes, n)
    return {"passes": successes, "n": n, "rate": round(successes / n, 3) if n else None,
            "wilson95": [round(low, 3), round(high, 3)]}


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def scorecard(scores: list[CaseScore], latencies_ms: list[float], tokens: dict[str, float] | None = None,
              low_confidence_types: list[str] | None = None, k: int | None = None) -> dict[str, Any]:
    """Pass rate with its Wilson interval, groundedness, key-fact recall, refusal precision/recall, citation
    validity, hit@k, tool accuracy, latency. ``scores`` are the cases of ONE split."""
    n = len(scores)
    facts = [f for s in scores for f in s.layer1.facts]
    answered = [s for s in scores if s.layer2.ran and s.answer_status == "answered"]
    refused_actual = [s for s in scores if s.answer_status == "refused"]
    refused_expected = [s for s in scores if s.expected_status == "refused"]
    true_refusals = [s for s in refused_actual if s.expected_status == "refused"]
    cited = sum(s.layer1.citations_total for s in scores)
    valid = sum(s.layer1.citations_valid for s in scores)
    hits = [s for s in scores if s.layer1.retrieval_hit is not None]
    tools = [s for s in scores if s.layer1.tool_ok is not None]
    ranks = [s.layer1.retrieval_rank for s in hits if s.layer1.retrieval_rank]
    by_type: dict[str, Any] = {}
    for kind in sorted({s.type for s in scores}):
        group = [s for s in scores if s.type == kind]
        by_type[kind] = _rate(sum(s.passed for s in group), len(group))
    card: dict[str, Any] = {
        "cases": n,
        "pass_rate": _rate(sum(s.passed for s in scores), n),
        "pass_rate_without_layer2_upgrades": _rate(sum(s.passed_without_layer2_upgrades for s in scores), n),
        "pass_rate_by_type": by_type,
        "groundedness": {**_rate(sum(1 for s in answered if not s.layer2.unsupported), len(answered)),
                         "note": "answered cases whose every claim is supported by the retrieved context (layer 2)"},
        "key_fact_recall": {"matched": sum(f.matched for f in facts), "total": len(facts),
                            "rate": _ratio(sum(f.matched for f in facts), len(facts)),
                            "layer1_only_rate": _ratio(sum(f.matched and not f.upgraded_by_judge for f in facts), len(facts))},
        "refusal": {"precision": _ratio(len(true_refusals), len(refused_actual)), "recall": _ratio(len(true_refusals), len(refused_expected)),
                    "refused_by_agent": len(refused_actual), "refusal_expected": len(refused_expected),
                    "answered_when_refusal_expected": sum(1 for s in refused_expected if s.answer_status != "refused")},
        "citation_validity": {"valid": valid, "total": cited, "rate": _ratio(valid, cited)},
        "retrieval_hit_at_k": {"k": k, "hits": sum(bool(s.layer1.retrieval_hit) for s in hits), "n": len(hits),
                               "rate": _ratio(sum(bool(s.layer1.retrieval_hit) for s in hits), len(hits)),
                               "mean_first_hit_rank": round(statistics.mean(ranks), 2) if ranks else None},
        "tool_accuracy": {"correct": sum(bool(s.layer1.tool_ok) for s in tools), "n": len(tools),
                          "rate": _ratio(sum(bool(s.layer1.tool_ok) for s in tools), len(tools))},
        "latency_ms": {"mean": round(statistics.mean(latencies_ms)) if latencies_ms else None,
                       "median": round(statistics.median(latencies_ms)) if latencies_ms else None,
                       "max": round(max(latencies_ms)) if latencies_ms else None},
        "tokens_per_case": tokens or {},
        "layer2_low_confidence_types": low_confidence_types or [],
        "low_confidence_cases": sum(s.low_confidence for s in scores),
    }
    return card
