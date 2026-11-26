"""Eval-set consistency (spec §9, P5): no case may contradict the requirements.

* No case expects a refusal on a must-answer topic, or an answer on a must-refuse topic.
* A refusal that comes from a gap on a topic that is still must-answer (a split topic) must name
  the unsupported ASPECT; otherwise it would refuse the part the documents do support.

Violating cases are dropped with a reason; ``assert_cases_consistent`` then fails loudly if any
remain, because that would be a bug in the generator, not a data problem.
"""

from __future__ import annotations

from forgeai.evals.models import EvalCase
from forgeai.intake.schema import Requirements
from forgeai.textmatch import content_terms


class EvalConsistencyError(Exception):
    """An eval case contradicts the requirements."""


def violation(case: EvalCase, req: Requirements) -> str | None:
    """Why ``case`` contradicts ``req``, or None."""
    answer_ids = {t.id for t in req.must_answer_topics}
    refuse_ids = {r.id for r in req.must_refuse_topics}
    gaps = {g.id: g for g in req.gaps}
    terms = content_terms(case.question)
    if case.expected_status == "refused":
        if case.topic_ref in answer_ids:
            return f"refusal expected on must-answer topic {case.topic_ref}"
        gap = gaps.get(case.topic_ref)
        if gap is not None and gap.topic_id in answer_ids and not gap.aspect:
            return f"refusal from gap {gap.id} would refuse the whole must-answer topic {gap.topic_id}"
        if case.origin in ("must_refuse", "negative_tool", "gap") and not (gap and gap.aspect):
            for topic in req.must_answer_topics:
                topic_terms = content_terms(topic.topic)
                if topic_terms and topic_terms <= terms:
                    return f"refusal question restates must-answer topic {topic.id} ({topic.topic!r})"
        return None
    if case.topic_ref in refuse_ids:
        return f"answer expected on must-refuse topic {case.topic_ref}"
    gap = gaps.get(case.topic_ref)
    if gap is not None and not gap.aspect:
        return f"answer expected on gap {gap.id}, which the documents do not cover"
    for refuse in req.must_refuse_topics:
        refuse_terms = content_terms(refuse.topic)
        if refuse_terms and refuse_terms <= terms and case.origin in ("topic", "snapshot"):
            return f"answerable question restates must-refuse topic {refuse.id} ({refuse.topic!r})"
    return None


def assert_cases_consistent(cases: list[EvalCase], req: Requirements) -> None:
    bad = [(c.id, v) for c in cases if (v := violation(c, req))]
    if bad:
        raise EvalConsistencyError("; ".join(f"{cid}: {why}" for cid, why in bad))
