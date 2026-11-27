"""Measure what one evaluation pass costs, so the build time of P6 and P7 can be projected from facts.

Usage: python scripts/time_eval_pass.py <run_dir> [sample_size] [--config path]

Runs a sample of the run's dev eval cases through the generated agent (every answer is a real model call),
then asks the judge model one answer-versus-fact question per sample answer (the same kind and size of call the
P6 judge makes). Batched by stage, like the real pass: all agent answers first, then all judge calls. Writes
``<run_dir>/timing_probe.json`` with seconds per agent answer, seconds per judge call and model load times. It
changes nothing else.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel  # noqa: E402

from forgeai.config import load_config  # noqa: E402
from forgeai.embeddings import SentenceTransformerEmbedder  # noqa: E402
from forgeai.evals.build import load_eval_set  # noqa: E402
from forgeai.llm import LLMClient, call_structured, set_client  # noqa: E402
from forgeai.runner.agent_def import load_agent  # noqa: E402
from forgeai.runner.runner import AgentRunner  # noqa: E402
from forgeai.timing import summarize_calls  # noqa: E402


class _Answer(BaseModel):
    answer: str = ""


def main(run_dir: Path, sample: int, config_path: str) -> None:
    config = load_config(config_path)
    trace = run_dir / "timing_probe_trace.jsonl"
    trace.unlink(missing_ok=True)
    set_client(LLMClient(config, trace_path=trace))
    agent_path = run_dir / "02_agent" / "agent.v0.yaml"
    embedder = SentenceTransformerEmbedder(config.embedding_model, config.embedding_revision)
    runner = AgentRunner(load_agent(agent_path), agent_path.parent, config, embedder)
    cases = [c for c in load_eval_set(run_dir / "03_evals" / "eval_set.jsonl") if c.split == "dev"]
    step = max(1, len(cases) // sample)
    chosen = cases[::step][:sample]

    agent_seconds, answers = [], []
    for case in chosen:  # stage 1: every answer from the agent model
        started = time.perf_counter()
        result = runner.ask(case.question)
        agent_seconds.append(time.perf_counter() - started)
        answers.append((case, result.answer.answer or result.answer.clarifying_question or result.answer.status))

    judge, judge_seconds = config.judge_model, []
    for case, answer in answers:  # stage 2: the judge model rules on each answer
        facts = "; ".join(f.text for f in case.key_facts) or case.expected_status
        started = time.perf_counter()
        call_structured("probe.judge", "evals/answer_from_context",
                        {"passages": f"Expected facts: {facts}\nAgent answer: {answer}", "question": case.question},
                        _Answer, judge.name, judge.temperature, seed=judge.seed, fallback=_Answer())
        judge_seconds.append(time.perf_counter() - started)

    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    live = [r for r in rows if not r.get("cache_hit")]
    swaps = [r["load_ms"] / 1000 for r in live if r.get("swap") or r.get("load_ms", 0) > 1000]
    report = {
        "sampled_cases": len(chosen), "types": sorted({c.type for c in chosen}),
        "agent_seconds_per_answer": {"mean": round(statistics.mean(agent_seconds), 1), "median": round(statistics.median(agent_seconds), 1),
                                     "max": round(max(agent_seconds), 1)},
        "judge_seconds_per_call": {"mean": round(statistics.mean(judge_seconds), 2), "max": round(max(judge_seconds), 2)},
        "model_load_seconds_when_swapping": round(statistics.mean(swaps), 1) if swaps else 0.0,
        "llm_calls": summarize_calls(rows),
        "note": "cache hits excluded from latency; a repeated question would be answered from the response cache",
    }
    (run_dir / "timing_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cfg = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "forgeai.yaml"
    if "--config" in sys.argv:
        args.remove(cfg)
    main(Path(args[0]), int(args[1]) if len(args) > 1 else 8, cfg)
