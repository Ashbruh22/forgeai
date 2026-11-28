"""Phase 7 driver: the noise floor, the optimizer loop, and the sealed test step.

This module is the only place where the optimizer's dev results and the test split meet, and it keeps them apart: the
loop (``forgeai/optimizer``) gets a dev evaluator and an ``after_iteration`` hook; the hook scores the candidate agent
on the TEST split and writes it under ``sealed/``. Nothing the loop imports can read those files.
"""

from __future__ import annotations

import difflib
import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from forgeai.config import ForgeConfig
from forgeai.embeddings import Embedder
from forgeai.evals.build import load_eval_set
from forgeai.evals.freeze import assert_eval_set_frozen
from forgeai.evals.models import EvalCase
from forgeai.judge import sealed, store
from forgeai.judge.baseline import to_case_run
from forgeai.judge.fingerprint import assert_judge_frozen
from forgeai.judge.metrics import scorecard
from forgeai.judge.models import CaseRun, CaseScore
from forgeai.judge.score import score_case
from forgeai.llm import LLMClient, get_client, set_client
from forgeai.optimizer.gate import noise_flips
from forgeai.optimizer.loop import DevResult, LoopResult, optimize, saved_agent, write_agent
from forgeai.runner.agent_def import AgentDef, load_agent
from forgeai.runner.runner import AgentRunner


class Ruler:
    """Everything needed to run an agent on a list of cases and judge the answers, batched by stage."""

    def __init__(self, run_dir: Path, config: ForgeConfig, embedder: Embedder, low_confidence_types: list[str], today: date | None = None,
                 progress=print) -> None:  # noqa: ANN001
        self.config, self.embedder, self.today = config, embedder, today or date.today()
        self.run_dir, self.fresh_cache = run_dir, run_dir / ".cache_confirm"
        self.base_dir = run_dir / "02_agent"  # the corpus path of agent.v0.yaml is relative to this folder
        self.low, self.progress = low_confidence_types, progress
        self.repairs: dict[str, list[dict[str, str]]] = {}
        self.nli = None
        if config.judge.use_nli:
            from forgeai.judge.nli import NliChecker

            self.nli = NliChecker(config.judge.nli_model, config.judge.nli_revision)

    def run(self, agent: AgentDef, cases: list[EvalCase], label: str) -> tuple[dict[str, CaseRun], dict[str, CaseScore]]:
        runner = AgentRunner(agent, self.base_dir, self.config, self.embedder)
        runs: dict[str, CaseRun] = {}
        self.repairs = {}  # case id -> citation repairs of the last run (product robustness, logged)
        for position, case in enumerate(cases, 1):  # stage 1: every answer from the agent model
            result = runner.ask(case.question)
            runs[case.id] = to_case_run(case, result, runner)
            if result.citation_repairs:
                self.repairs[case.id] = result.citation_repairs
            self.progress(f"{label}: agent {position}/{len(cases)}")
        scores: dict[str, CaseScore] = {}
        for position, case in enumerate(cases, 1):  # stage 2: every answer judged (one model swap)
            scores[case.id] = score_case(case, runs[case.id], self.config.judge_model, today=self.today,
                                         fuzzy_min=self.config.thresholds.fact_fuzzy_score, low_confidence_types=self.low,
                                         nli=self.nli, nli_min=self.config.judge.nli_contradiction_min)
            self.progress(f"{label}: judge {position}/{len(cases)} {'pass' if scores[case.id].passed else 'FAIL'}")
        return runs, scores


class DevEvaluator:
    def __init__(self, ruler: Ruler, dev_cases: list[EvalCase]) -> None:
        self.ruler, self.cases = ruler, dev_cases

    def dev(self, agent: AgentDef, iteration: int) -> DevResult:
        runs, scores = self.ruler.run(agent, self.cases, f"iter {iteration} dev")
        return DevResult(scores, runs, dict(self.ruler.repairs))


    def confirm(self, agent: AgentDef, case_ids: list[str], iteration: int) -> dict[str, CaseScore]:
        """Re-run these dev cases with a fresh, empty cache (agent answers and judge verdicts are sampled again)."""
        subset = [c for c in self.cases if c.id in set(case_ids)]
        previous = get_client()
        fresh = self.ruler.config.model_copy(update={"cache_dir": self.ruler.fresh_cache / f"confirm_{iteration}"})
        set_client(LLMClient(fresh, trace_path=previous.trace_path))
        try:
            ruler = Ruler(self.ruler.run_dir, fresh, self.ruler.embedder, self.ruler.low, today=self.ruler.today, progress=self.ruler.progress)
            _, scores = ruler.run(agent, subset, f"iter {iteration} confirm")
        finally:
            set_client(previous)
        return scores


def leakage_check_for(embedder: Embedder, dev_questions: list[str]):  # noqa: ANN201
    """Spec 5.5 for a few-shot patch: the highest cosine similarity between the example's question and any DEV question
    (the optimizer must not see the test split, so the dev questions stand for the eval questions here)."""
    import numpy as np

    vectors = np.asarray(embedder.encode(dev_questions))

    def check(shot) -> float:  # noqa: ANN001
        one = np.asarray(embedder.encode([shot.question]))
        return float((one @ vectors.T).max()) if len(dev_questions) else 0.0

    return check


def check_frozen(run_dir: Path, config: ForgeConfig) -> None:
    """The ruler must not change: the eval-set hash and the judge fingerprint recorded in the manifest must still match."""
    assert_eval_set_frozen(run_dir)
    assert_judge_frozen(run_dir, config)


def measure_noise(run_dir: Path, out: Path, config: ForgeConfig, embedder: Embedder, baseline_passed: dict[str, bool],
                  low_types: list[str], progress=print) -> dict[str, Any]:  # noqa: ANN001
    """Re-run agent v0 on the dev split with NO cache (a fresh cache folder: the agent answers and the judge verdicts are
    sampled again) and count the cases whose pass/fail differs from the baseline. That count is the noise floor."""
    check_frozen(run_dir, config)
    fresh = config.model_copy(update={"cache_dir": out / ".cache_noise"})
    set_client(LLMClient(fresh, trace_path=out / "trace.jsonl"))
    cases = [c for c in load_eval_set(run_dir / "03_evals" / "eval_set.jsonl") if c.split == "dev"]
    agent0 = load_agent(run_dir / "02_agent" / "agent.v0.yaml")
    ruler = Ruler(run_dir, fresh, embedder, low_types, progress=progress)
    runs, scores = ruler.run(agent0, cases, "noise")
    again = {cid: s.passed for cid, s in scores.items()}
    flipped = noise_flips(baseline_passed, again)
    baseline_runs = {r.case_id: r for r in _read_baseline_runs(out)}
    changed = sorted(cid for cid, r in runs.items() if cid in baseline_runs and r.answer.answer != baseline_runs[cid].answer.answer)
    report = {"dev_cases": len(cases), "noise_floor": len(flipped), "flipped_cases": flipped,
              "baseline_passes": sum(baseline_passed.values()), "rerun_passes": sum(again.values()),
              "answers_that_differ_in_text": len(changed), "method": "agent v0 on dev, fresh cache for agent and judge, flips vs the baseline"}
    (out / "noise_floor.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _read_baseline_runs(out: Path) -> list[CaseRun]:
    path = store.dev_dir(out) / "answers.jsonl"
    return [CaseRun.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def agent_diff(agent0_path: Path, final: AgentDef, corpus_dir: Path) -> str:
    """Unified diff of agent.v0.yaml and the final agent (the corpus path is made the same so only real changes show)."""
    before = yaml.safe_dump(load_agent(agent0_path).model_dump(mode="json"), sort_keys=False, allow_unicode=True, width=100)
    data = final.model_dump(mode="json")
    data["corpus"]["path"] = load_agent(agent0_path).corpus.path
    after = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
    return "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), "agent.v0.yaml", "05_final/agent.yaml"))


def run_optimizer(run_dir: Path, out: Path, config: ForgeConfig, embedder: Embedder, low_types: list[str], vocabulary: str,
                  progress=print) -> LoopResult:  # noqa: ANN001
    """Iteration 0 is the baseline in ``out/05_baseline``; this runs up to three cycles and writes ``04_iterations`` and ``05_final``."""
    check_frozen(run_dir, config)
    noise = json.loads((out / "noise_floor.json").read_text(encoding="utf-8"))  # the loop refuses to run without a measured floor
    set_client(LLMClient(config, trace_path=out / "trace.jsonl"))
    all_cases = load_eval_set(run_dir / "03_evals" / "eval_set.jsonl")
    dev = {c.id: c for c in all_cases if c.split == "dev"}
    test_cases = [c for c in all_cases if c.split == "test"]
    agent0_path = run_dir / "02_agent" / "agent.v0.yaml"
    agent0 = load_agent(agent0_path)
    corpus_dir = agent0.corpus_dir(agent0_path.parent)
    base_runs = {r.case_id: r for r in _read_baseline_runs(out)}
    baseline = DevResult({s.case_id: s for s in store.read_dev_scores(out)}, base_runs)
    ruler = Ruler(run_dir, config, embedder, low_types, progress=progress)
    write_agent(agent0, out / "04_iterations" / "iter_0", corpus_dir)

    previous_test = sealed.read_test_scorecard(out)["pass_rate"]  # iteration 0 is the baseline's test score
    state = {"before": previous_test}

    def after_iteration(n: int, candidate: AgentDef, accepted: bool, change: dict[str, Any]) -> None:
        """The sealed step: the candidate's TEST score, written where the optimizer cannot read it."""
        check_frozen(run_dir, config)  # the ruler is checked between cycles too
        runs, scores = ruler.run(candidate, test_cases, f"iter {n} test")
        ordered = [c.id for c in test_cases]
        card = scorecard([scores[c] for c in ordered], [runs[c].latency_ms for c in ordered], {}, low_types, candidate.retrieval.top_k)
        card.update({"split": "test", "iteration": n})
        after = card["pass_rate"]
        sealed.write_iteration(out, n, [runs[c] for c in ordered], [scores[c] for c in ordered], card,
                               {"iteration": n, "accepted": accepted, "score_before": {"test": state["before"]}, "score_after": {"test": after},
                                "fixed": [c for c in ordered if scores[c].passed and not _was_passing(out, n, c)]})
        if accepted:
            state["before"] = after

    result = optimize(agent0=agent0, cases=dev, baseline=baseline, evaluator=DevEvaluator(ruler, list(dev.values())), config=config, out=out,
                      corpus_dir=corpus_dir, noise_floor=int(noise["noise_floor"]), vocabulary=vocabulary, after_iteration=after_iteration,
                      leakage_check=leakage_check_for(embedder, [c.question for c in dev.values()]))
    final_dir = out / "05_final"
    write_agent(result.final, final_dir, corpus_dir)
    (final_dir / "agent_diff.txt").write_text(agent_diff(agent0_path, result.final, corpus_dir), encoding="utf-8")
    summary = {"stop_reason": result.stop_reason, "noise_floor": noise["noise_floor"],
               "iterations": [{"iteration": r.iteration, "accepted": r.accepted, "status": r.change["status"], "category": r.change["category"],
                               "patch": r.change["patch_id"], "candidates_considered": r.change["candidates_considered"],
                               "citation_repairs_dev": r.change["citation_repairs_dev"],
                               "dev_before": r.change["score_before"]["dev"], "dev_after": r.change["score_after"]["dev"],
                               "fixed": r.change["fixed"], "regressed": r.change["regressed"], "gate": r.change["gate_reason"]} for r in result.iterations]}
    (out / "04_iterations" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return result


def _was_passing(out: Path, n: int, case_id: str) -> bool:
    """Was the case passing in the previous accepted state of the test split? (for the sealed change log only)"""
    previous = n - 1
    try:
        scores = sealed.read_test_scores(out) if previous == 0 else sealed.read_iteration(out, previous)["scores"]
    except OSError:
        return False
    return any(s.case_id == case_id and s.passed for s in scores)
