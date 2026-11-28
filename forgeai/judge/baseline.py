"""Iteration 0 (spec §6.4): run the v0 agent on every eval case, then judge every answer, batched by stage.

Stage 1 puts every question to the agent (one model loaded). Stage 2 judges every answer (one model swap). Dev
results go to ``store`` (readable by the optimizer); test results go to ``sealed`` (not).
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from forgeai.config import ForgeConfig
from forgeai.embeddings import Embedder
from forgeai.evals.build import load_eval_set
from forgeai.evals.models import EvalCase
from forgeai.judge import sealed, store
from forgeai.judge.metrics import scorecard
from forgeai.judge.models import CaseRun, CaseScore, CitedChunk, RetrievedChunk
from forgeai.judge.score import score_case
from forgeai.runner.agent_def import load_agent
from forgeai.runner.runner import AgentRunner, RunResult
from forgeai.timing import _trace_slice

Progress = Callable[[str], None]


def to_case_run(case: EvalCase, result: RunResult, runner: AgentRunner) -> CaseRun:
    """The agent's run as the judge reads it. Citation ranges come from the runner's index, never from the model."""
    cited = []
    for citation in result.answer.citations:
        chunk = runner.index.by_id.get(citation.chunk_id)
        if chunk is not None:
            cited.append(CitedChunk(chunk_id=chunk.chunk_id, doc=chunk.doc, char_start=chunk.char_start, char_end=chunk.char_end))
    retrieved = [RetrievedChunk(chunk_id=h.chunk.chunk_id, doc=h.chunk.doc, char_start=h.chunk.char_start, char_end=h.chunk.char_end,
                                text=h.chunk.text, dense=h.dense) for h in result.retrieved]
    return CaseRun(case_id=case.id, answer=result.answer, retrieved=retrieved, cited=cited, issues=result.issues,
                   latency_ms=result.latency_ms, search_query=result.search_query)


def _tokens(trace: Path, offset: int, cases: int) -> dict[str, float]:
    rows = [r for r in _trace_slice(trace, offset) if r.get("role") == "runner.agent" and not r.get("cache_hit")]
    if not rows or not cases:
        return {}
    return {"agent_in": round(sum(r.get("tokens_in", 0) for r in rows) / cases), "agent_out": round(sum(r.get("tokens_out", 0) for r in rows) / cases),
            "note": "live calls only; cached answers carry no token counts"}


def run_baseline(run_dir: Path, out: Path, config: ForgeConfig, embedder: Embedder, *, canaries: dict[str, Any],
                 today: date | None = None, progress: Progress = print) -> dict[str, Any]:
    """Run and judge the v0 agent of ``run_dir``; write dev files under ``out/05_baseline/dev`` and test files under ``sealed``."""
    today = today or date.today()
    agent_path = run_dir / "02_agent" / "agent.v0.yaml"
    runner = AgentRunner(load_agent(agent_path), agent_path.parent, config, embedder)
    cases = load_eval_set(run_dir / "03_evals" / "eval_set.jsonl")
    trace = out / "trace.jsonl"

    offset = trace.stat().st_size if trace.is_file() else 0
    runs: dict[str, CaseRun] = {}
    for position, case in enumerate(cases, 1):  # stage 1: every answer from the agent model
        runs[case.id] = to_case_run(case, runner.ask(case.question), runner)
        progress(f"agent {position}/{len(cases)} {case.type}")
    tokens = _tokens(trace, offset, len(cases))

    low = list(canaries.get("low_confidence_types", []))
    scores: dict[str, CaseScore] = {}
    for position, case in enumerate(cases, 1):  # stage 2: every answer judged
        scores[case.id] = score_case(case, runs[case.id], config.judge_model, today=today,
                                     fuzzy_min=config.thresholds.fact_fuzzy_score, low_confidence_types=low)
        progress(f"judge {position}/{len(cases)} {'pass' if scores[case.id].passed else 'FAIL'}")

    k = runner.agent.retrieval.top_k
    summary: dict[str, Any] = {"judge": config.judge_model.name, "scored_on": today.isoformat(), "top_k": k,
                               "layer2_low_confidence_types": low}
    for split in ("dev", "test"):
        part = [c for c in cases if c.split == split]
        split_scores = [scores[c.id] for c in part]
        split_runs = [runs[c.id] for c in part]
        card = scorecard(split_scores, [r.latency_ms for r in split_runs], tokens, low, k)
        card.update({"split": split, "judge": config.judge_model.name, "scored_on": today.isoformat()})
        if split == "dev":
            store.write_split(store.dev_dir(out), "", split_runs, split_scores, card)
        else:
            sealed.write_test(out, split_runs, split_scores, card)
        summary[split] = {"cases": len(part), "pass_rate": card["pass_rate"]}
    visible = {k: v for k, v in summary.items() if k != "test"}  # the file the optimizer may read says nothing about test
    visible["mean_agent_latency_ms"] = round(statistics.mean(r.latency_ms for r in runs.values()))
    (out / "05_baseline" / "summary.json").write_text(json.dumps(visible, indent=2), encoding="utf-8")
    return summary
