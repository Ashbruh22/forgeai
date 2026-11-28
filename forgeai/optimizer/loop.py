"""The critic-optimizer loop (spec §6.6): at most three cycles after the baseline, each one a measured experiment.

One cycle: classify every dev failure, pick the category with the most failures (ties: hallucination > wrong refusal >
retrieval miss > prompt defect), apply ONE patch set, re-run the whole dev split, and keep the patch only if the gate
(``gate.py``) says the gain beats the noise floor with at most one regression; otherwise revert it. The loop stops when
the target is met, nothing patchable is left, or a cycle produced no accepted patch (plateau).

This module and everything it imports see DEV results only. Test scores are computed by a separate step the loop
cannot read (``forgeai/iterate.py``), which the loop reaches only through the ``after_iteration`` hook it is given.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

from forgeai.config import ForgeConfig
from forgeai.evals.models import EvalCase
from forgeai.judge import store
from forgeai.judge.metrics import scorecard
from forgeai.judge.models import CaseRun, CaseScore
from forgeai.optimizer import critic as critic_module
from forgeai.optimizer.classify import RETRIEVAL_MISS, Failure, classify_all
from forgeai.optimizer.gate import acceptance_gate, confirm_gate
from forgeai.optimizer.patches import PatchSet, apply_ops, candidates, with_text
from forgeai.runner.agent_def import AgentDef

TARGET_DEV_PASS_RATE = 0.90


@dataclass
class DevResult:
    scores: dict[str, CaseScore]
    runs: dict[str, CaseRun] = field(default_factory=dict)

    @property
    def passed(self) -> dict[str, bool]:
        return {cid: s.passed for cid, s in self.scores.items()}


class Evaluator(Protocol):
    def dev(self, agent: AgentDef, iteration: int) -> DevResult:
        """Run ``agent`` on every dev case and judge the answers."""

    def confirm(self, agent: AgentDef, case_ids: list[str], iteration: int) -> dict[str, CaseScore]:
        """Run ``agent`` on just these dev cases again, with NO cache (fresh agent answers, fresh judge verdicts)."""


AfterIteration = Callable[[int, AgentDef, bool, dict[str, Any]], None]


@dataclass
class IterationRecord:
    iteration: int
    change: dict[str, Any]
    accepted: bool


@dataclass
class LoopResult:
    final: AgentDef
    iterations: list[IterationRecord]
    stop_reason: str
    final_dev: DevResult


def pass_rate(result: DevResult) -> float:
    return sum(result.passed.values()) / max(len(result.passed), 1)


def fabricated_on_refusals(result: DevResult, cases: dict[str, EvalCase]) -> int:
    """Must-refuse cases the agent ANSWERED (spec: max_fabricated_on_refusals is 0)."""
    return sum(1 for cid, s in result.scores.items() if cases[cid].expected_status == "refused" and s.answer_status == "answered")


def target_met(result: DevResult, cases: dict[str, EvalCase]) -> bool:
    return pass_rate(result) >= TARGET_DEV_PASS_RATE and fabricated_on_refusals(result, cases) == 0


def saved_agent(agent: AgentDef, directory: Path, corpus_dir: Path) -> dict[str, Any]:
    """The agent as written into an iteration folder: its corpus path made relative to that folder so it loads from there."""
    data = agent.model_dump(mode="json")
    data["corpus"]["path"] = os.path.relpath(corpus_dir, directory).replace("\\", "/")
    return data


def write_agent(agent: AgentDef, directory: Path, corpus_dir: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "agent.yaml").write_text(
        yaml.safe_dump(saved_agent(agent, directory, corpus_dir), sort_keys=False, allow_unicode=True, width=100), encoding="utf-8")


def write_dev(directory: Path, result: DevResult, cases: dict[str, EvalCase], k: int) -> dict[str, Any]:
    ordered = [cid for cid in cases if cid in result.scores]
    scores = [result.scores[c] for c in ordered]
    runs = [result.runs[c] for c in ordered if c in result.runs]
    card = scorecard(scores, [r.latency_ms for r in runs], {}, [], k)
    store.write_split(directory / "dev", "", runs, scores, card)
    return card


def _crowded_out(failures: list[Failure], cases: dict[str, EvalCase], result: DevResult) -> bool:
    """Did a retrieval miss on a prose document come with a retrieved set made of table rows only?"""
    for f in failures:
        if f.category != RETRIEVAL_MISS or f.case_id not in result.runs:
            continue
        wanted = {s.doc for s in cases[f.case_id].seed_spans}
        got = {c.doc for c in result.runs[f.case_id].retrieved}
        if got and all(d.lower().endswith(".csv") for d in got) and any(not d.lower().endswith(".csv") for d in wanted):
            return True
    return False


def _line(result: DevResult) -> dict[str, Any]:
    return {"passes": sum(result.passed.values()), "n": len(result.passed), "rate": round(pass_rate(result), 3)}


def choose_patch(current: AgentDef, failures: list[Failure], cases: dict[str, EvalCase], result: DevResult, category_order: list[str],
                 rejected: frozenset[str], config: ForgeConfig, vocabulary: str) -> tuple[PatchSet | None, AgentDef | None, list[dict], str]:
    """The first usable patch set: the target category's candidates in order, then the next category's. A text patch is
    drafted by the critic and checked by the guard; a draft the guard refuses is logged and the next candidate is tried."""
    drafts_rejected: list[dict] = []
    for category in category_order:
        mine = [f for f in failures if f.category == category]
        questions = [cases[f.case_id].question for f in mine]
        for patch in candidates(category, current, failures, questions, crowded_out=_crowded_out(failures, cases, result), exclude=rejected):
            if patch.needs_text:
                text, log = critic_module.draft_text(patch, current, mine, cases, config, vocabulary)
                drafts_rejected += log
                if text is None:
                    continue
                patch = with_text(patch, text)
            try:
                return patch, apply_ops(current, patch.ops), drafts_rejected, category
            except ValueError as exc:
                drafts_rejected.append({"patch": patch.id, "reasons": [f"out of bounds: {exc}"]})
    return None, None, drafts_rejected, ""


def optimize(*, agent0: AgentDef, cases: dict[str, EvalCase], baseline: DevResult, evaluator: Evaluator, config: ForgeConfig,
             out: Path, corpus_dir: Path, noise_floor: int, vocabulary: str = "", after_iteration: AfterIteration | None = None,
             max_cycles: int = 3) -> LoopResult:
    """Run the loop. ``cases`` are the DEV cases (id -> case); ``baseline`` is iteration 0."""
    from forgeai.optimizer.classify import CATEGORIES

    cap = min(max_cycles, config.limits.max_fix_cycles)  # the hard limit lives in Limits; it cannot be raised from here
    current, current_result = agent0, baseline
    rejected: set[str] = set()
    records: list[IterationRecord] = []
    stop = f"the cycle limit ({cap}) was reached"
    k = agent0.retrieval.top_k
    for n in range(1, cap + 1):
        if target_met(current_result, cases):
            stop = "target met: dev pass rate >= 0.90 and no must-refuse question answered"
            break
        failures = classify_all(cases, current_result.scores)
        if not failures:
            stop = "no failing dev case is left"
            break
        solid = [f for f in failures if not f.low_confidence] or failures
        counts: dict[str, int] = {}
        for f in solid:
            counts[f.category] = counts.get(f.category, 0) + 1
        order = sorted(counts, key=lambda c: (-counts[c], CATEGORIES.index(c)))
        order += [c for c in CATEGORIES if c not in order and any(f.category == c for f in failures)]
        patch, candidate, drafts, category = choose_patch(current, failures, cases, current_result, order, frozenset(rejected), config, vocabulary)
        folder = out / "04_iterations" / f"iter_{n}"
        if patch is None or candidate is None:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "rejected_patches.json").write_text(json.dumps(drafts, indent=2), encoding="utf-8")
            stop = "no patchable failure: every candidate patch is out of bounds, already rejected, or failed the guard"
            break
        targets = sorted(f.case_id for f in failures if f.category == category)
        result = evaluator.dev(candidate, n)
        first = acceptance_gate(current_result.passed, result.passed, noise_floor)
        gate, confirmation = first, None
        if first.accepted:  # confirmation rerun: only the fixed and regressed cases, uncached; fixes must hold on both runs
            again = evaluator.confirm(candidate, sorted(first.fixed + first.regressed), n)
            gate = confirm_gate(first, {c: s.passed for c, s in again.items()}, noise_floor)
            confirmation = {"rerun_cases": sorted(again), "fixes_first_run": first.fixed, "fixes_held": gate.fixed,
                            "fixes_dropped": sorted(set(first.fixed) - set(gate.fixed)),
                            "regressed_still_failing": [c for c in first.regressed if not again[c].passed],
                            "regressed_passed_on_rerun": [c for c in first.regressed if again[c].passed],
                            "first_run_gate": first.reason}
            for cid in first.fixed:  # a pass counts only if it held on both runs
                if not again[cid].passed:
                    result.scores[cid] = again[cid]
        write_agent(candidate, folder, corpus_dir)
        card = write_dev(folder, result, cases, k)
        change = {
            "iteration": n, "category": category, "category_counts": counts, "target_cases": targets,
            "failures": {f.case_id: {"category": f.category, "section": f.section, "low_confidence": f.low_confidence, "reason": f.reason[:200]} for f in failures},
            "patch_id": patch.id, "patch": patch.ops, "rationale": patch.rationale, "noise_floor": noise_floor,
            "score_before": {"dev": _line(current_result)}, "score_after": {"dev": _line(result)},
            "fixed": gate.fixed, "regressed": gate.regressed, "still_failing": gate.still_failing,
            "net_gain": gate.net_gain, "accepted": gate.accepted, "gate_reason": gate.reason, "confirmation": confirmation,
            "status": "accepted" if gate.accepted else ("rejected: unconfirmed" if first.accepted else "rejected: gate"),
            "rejected_drafts": len(drafts), "dev_scorecard": {"pass_rate": card["pass_rate"], "groundedness": card["groundedness"]["rate"]},
        }
        (folder / "rejected_patches.json").write_text(json.dumps(drafts, indent=2), encoding="utf-8")
        (folder / "change.json").write_text(json.dumps(change, indent=2), encoding="utf-8")
        records.append(IterationRecord(n, change, gate.accepted))
        if after_iteration is not None:
            after_iteration(n, candidate, gate.accepted, change)
        if gate.accepted:
            current, current_result = candidate, result
        else:
            rejected.add(patch.id)  # reverted: the agent stays as it was; the patch is logged as evidence
            stop = "plateau: the last cycle produced no accepted patch"
            break
    else:
        stop = "target met: dev pass rate >= 0.90 and no must-refuse question answered" if target_met(current_result, cases)             else f"the cycle limit ({cap}) was reached"
    return LoopResult(current, records, stop, current_result)
