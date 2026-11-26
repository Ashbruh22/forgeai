"""Few-shot leakage check (spec §5.5).

Few-shot examples show the output FORMAT only. They are rejected if their embedding
similarity to any eval question is above the threshold, so the agent can never be shown the
answers to its own test.

The check needs the eval set, which exists only after P5. Until then ``generate_agent``
records ``fewshot_leakage_check: pending`` and calls ``validate_few_shots`` again once the
questions are known (TODO(P5), see docs/DECISIONS.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pathlib import Path

import numpy as np

from forgeai.embeddings import Embedder
from forgeai.runner.agent_def import AgentDef, FewShot


@dataclass
class LeakageReport:
    threshold: float
    kept: list[FewShot] = field(default_factory=list)
    rejected: list[tuple[FewShot, str, float]] = field(default_factory=list)  # (example, eval question, similarity)


def check_few_shot_leakage(
    few_shots: list[FewShot], eval_questions: list[str], embedder: Embedder, threshold: float
) -> LeakageReport:
    """Split ``few_shots`` into kept and rejected by similarity to any eval question."""
    report = LeakageReport(threshold)
    if not few_shots:
        return report
    if not eval_questions:
        report.kept = list(few_shots)
        return report
    shot_vectors = embedder.encode([f.question for f in few_shots])
    question_vectors = embedder.encode(eval_questions)
    similarity = shot_vectors @ question_vectors.T
    for shot, row in zip(few_shots, similarity, strict=True):
        worst = int(np.argmax(row))
        if float(row[worst]) > threshold:
            report.rejected.append((shot, eval_questions[worst], float(row[worst])))
        else:
            report.kept.append(shot)
    return report


def validate_few_shots(
    agent: AgentDef, eval_questions: list[str] | None, embedder: Embedder, threshold: float
) -> tuple[AgentDef, LeakageReport | None]:
    """Hook for P5/P7: drop leaking examples from ``agent``. ``None`` eval set means pending."""
    if eval_questions is None:
        return agent, None
    report = check_few_shot_leakage(agent.prompt_sections.few_shot, eval_questions, embedder, threshold)
    sections = agent.prompt_sections.model_copy(update={"few_shot": report.kept})
    return agent.model_copy(update={"prompt_sections": sections}), report


def run_fewshot_check(run_dir: Path, embedder: Embedder, threshold: float, eval_questions: list[str]) -> dict:
    """Run the leakage check for a run now that its eval set exists (called by the build, not by evals)."""
    import json

    from forgeai.runner.agent_def import load_agent, save_agent

    path = run_dir / "02_agent" / "agent.v0.yaml"
    agent = load_agent(path)
    checked, report = validate_few_shots(agent, eval_questions, embedder, threshold)
    result = {
        "eval_questions": len(eval_questions), "few_shot_examples_checked": len(agent.prompt_sections.few_shot),
        "kept": len(checked.prompt_sections.few_shot), "threshold": threshold,
        "rejected": [{"example": s.question, "similar_eval_question": q, "similarity": round(sim, 3)}
                     for s, q, sim in (report.rejected if report else [])],
    }
    note = (f"passed: {result['few_shot_examples_checked']} examples checked against {len(eval_questions)} eval questions"
            if not result["rejected"] else f"FAILED: {len(result['rejected'])} examples removed")
    checked.provenance["fewshot_leakage_check"] = note
    save_agent(checked, path)
    (run_dir / "02_agent" / "fewshot_leakage.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
