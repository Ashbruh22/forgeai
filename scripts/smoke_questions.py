"""Ask a generated agent a few real questions and save the full answers (no mocks).

Usage: python scripts/smoke_questions.py <run_dir> <questions.yaml>

``questions.yaml`` is a list of ``{kind: in-scope|gap|out-of-scope|snapshot, question: ...}``.
Writes ``<run_dir>/02_agent/smoke_answers.json`` and prints each answer. The output is
evidence of what the agent actually says; nothing in it is edited by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forgeai.config import load_config  # noqa: E402
from forgeai.embeddings import SentenceTransformerEmbedder  # noqa: E402
from forgeai.llm import LLMClient, set_client  # noqa: E402
from forgeai.runner.agent_def import load_agent  # noqa: E402
from forgeai.runner.runner import AgentRunner  # noqa: E402


def main(run_dir: Path, questions_file: Path) -> None:
    config = load_config()
    set_client(LLMClient(config, trace_path=run_dir / "trace.jsonl"))
    agent_path = run_dir / "02_agent" / "agent.v0.yaml"
    runner = AgentRunner(
        load_agent(agent_path), agent_path.parent, config,
        SentenceTransformerEmbedder(config.embedding_model, config.embedding_revision),
    )
    results = []
    for item in yaml.safe_load(questions_file.read_text(encoding="utf-8")):
        result = runner.ask(item["question"])
        answer = result.answer
        print(f"\n[{item['kind']}] {item['question']}\n  status: {answer.status}\n  answer: {answer.answer or answer.clarifying_question}")
        for citation in answer.citations:
            print(f"  cite: {citation.chunk_id}: {citation.quote[:90]}")
        for call in answer.tool_calls:
            print(f"  tool: {call.tool}({call.args}) -> {call.result[:80]}")
        if result.issues:
            print(f"  CONTRACT ISSUES: {result.issues}")
        results.append({
            "kind": item["kind"], "question": item["question"], "answer": answer.model_dump(mode="json"),
            "contract_issues": result.issues, "retrieved": [h.chunk.chunk_id for h in result.retrieved],
            "latency_ms": round(result.latency_ms),
        })
    (agent_path.parent / "smoke_answers.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
