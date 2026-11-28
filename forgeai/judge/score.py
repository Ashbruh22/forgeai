"""Per-case verdicts (spec §6.3): layer 1 and layer 2 combined into one pass/fail with the checks that failed.

The reasons name the checks that failed. They are facts, not a failure classification: classifying a failure
(retrieval miss, generation error, ...) is the decision tree of P7.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from forgeai.config import ModelSpec
from forgeai.evals.models import EvalCase
from forgeai.judge.layer1 import check_layer1
from forgeai.judge.layer2 import run_layer2
from forgeai.judge.models import CaseRun, CaseScore, Layer1, Layer2


def decide(case: EvalCase, l1: Layer1, l2: Layer2) -> list[str]:
    """The checks of the §6.3 pass rule that fail for this case (empty list = pass)."""
    failed: list[str] = []
    if case.type == "out_of_scope":
        if not l1.status_ok:
            failed.append(f"status is {l1.status}, expected refused")
        if not l1.contract_valid:
            failed.append("contract invalid: " + "; ".join(l1.contract_issues))
        if l2.refusal_claims:
            failed.append(f"the refusal states factual claims: {l2.refusal_claims}")
        return failed

    if not l1.contract_valid:
        failed.append("contract invalid: " + "; ".join(l1.contract_issues))
    if case.type == "ambiguous":
        if l1.status == "needs_clarification":
            return failed
        if l1.status != "answered":
            return [*failed, f"status is {l1.status}, expected needs_clarification or an answer covering every reading"]
        if l1.citation_issues:
            failed.append("citation invalid: " + "; ".join(l1.citation_issues))
        if l1.interpretations_covered != l1.interpretations_total:
            failed.append(f"answer covers {l1.interpretations_covered} of {l1.interpretations_total} readings")
        if l2.unsupported:
            failed.append(f"{l2.unsupported} unsupported claim(s)")
        return failed

    if not l1.status_ok:
        failed.append(f"status is {l1.status}, expected {case.expected_status}")
    if l1.citation_issues:
        failed.append("citation invalid: " + "; ".join(l1.citation_issues))
    if case.type in ("answerable", "multidoc") and l1.source_ok is False:
        failed.append(f"expected source not cited: {l1.sources_missing}")
    missing = [f.fact for f in l1.facts if not f.matched]
    if missing:
        failed.append(f"key facts missing: {missing}")
    if case.type == "tool_required" and l1.tool_ok is False:
        failed.append("tool check failed: " + "; ".join(l1.tool_notes))
    if l2.unsupported:
        failed.append(f"{l2.unsupported} unsupported claim(s)")
    return failed


def _disagreements(case: EvalCase, before: Layer1, after: Layer1, l2: Layer2) -> list[dict]:
    found: list[dict] = []
    for log in l2.upgrades:
        if log["upgraded"]:
            found.append({"kind": "fact_upgraded", "layer1": "key fact not matched", "layer2": "stated in the answer", **log})
        elif log["judge_says_expressed"]:
            found.append({"kind": "upgrade_refused", "layer1": "key fact not matched", "layer2": "judge claims it is stated", **log})
    clean = (before.status_ok and before.contract_valid and not before.citation_issues and before.source_ok is not False
             and all(f.matched for f in before.facts))
    if clean and l2.unsupported:
        found.append({"kind": "layer1_passes_layer2_ungrounded", "layer1": "every deterministic check passed",
                      "layer2": [c.claim for c in l2.claims if not c.supported]})
    if after.status == "refused" and l2.refusal_claims:
        found.append({"kind": "refused_but_states_claims", "layer1": "status refused as expected", "layer2": l2.refusal_claims})
    if l2.claims and not l2.unsupported and any(not f.matched for f in after.facts):
        found.append({"kind": "claims_supported_key_fact_missing", "layer1": [f.fact for f in after.facts if not f.matched],
                      "layer2": "every claim is supported by the retrieved context"})
    return found


def score_case(case: EvalCase, run: CaseRun, judge: ModelSpec, *, today: date, fuzzy_min: int = 85,
               low_confidence_types: Iterable[str] = ()) -> CaseScore:
    layer1 = check_layer1(case, run, today=today, fuzzy_min=fuzzy_min)
    layer2, upgraded = run_layer2(case, run, layer1, judge)
    reasons = decide(case, upgraded, layer2)
    without_upgrades = decide(case, layer1, layer2)
    contributed = bool(layer2.unsupported or layer2.refusal_claims or any(u["upgraded"] for u in layer2.upgrades))
    return CaseScore(
        case_id=case.id, type=case.type, split=case.split, passed=not reasons, reasons=reasons, layer1=upgraded, layer2=layer2,
        passed_without_layer2_upgrades=not without_upgrades, expected_status=case.expected_status,
        low_confidence=bool(list(low_confidence_types)) and contributed, question=case.question,
        answer_status=run.answer.status, answer_text=run.answer.answer or (run.answer.clarifying_question or ""),
        disagreements=_disagreements(case, layer1, upgraded, layer2),
    )
