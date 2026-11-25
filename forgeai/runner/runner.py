"""The generic chat runner (spec §8): retrieve, compile, call, use tools, validate.

This is the single code path for both users (``forgeai chat``) and the eval harness.
It knows nothing about any particular agent or domain: everything comes from ``AgentDef``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from forgeai.config import ForgeConfig
from forgeai.embeddings import Embedder
from forgeai.llm import call_structured
from forgeai.runner.agent_def import EMPTY, AgentDef, compile_system_prompt
from forgeai.runner.contract import Answer, Citation, ToolCallRecord, contract_issues
from forgeai.runner.retrieval import Hit, HybridIndex, load_corpus_chunks
from forgeai.tools import ToolContext, ToolError, build_tools

History = list[tuple[str, str]]
MODEL_OUTPUT_INVALID = "model_output_invalid"
NO_FINAL_ANSWER = "no_final_answer"


class AgentStep(BaseModel):
    """One model step: either a tool call or the final answer (flat for small models)."""

    action: Literal["tool_call", "final"]
    tool: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    status: Literal["answered", "refused", "needs_clarification"] = "answered"
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    clarifying_question: str | None = None


class RewrittenQuery(BaseModel):
    query: str


@dataclass
class RunResult:
    question: str
    answer: Answer
    retrieved: list[Hit]
    issues: list[str]  # deterministic contract problems; empty means the contract is valid
    steps: list[dict[str, Any]] = field(default_factory=list)  # raw per-step trace for /debug
    search_query: str = ""
    latency_ms: float = 0.0

    @property
    def contract_valid(self) -> bool:
        return not self.issues


def format_context(hits: list[Hit]) -> str:
    if not hits:
        return "No relevant context was retrieved."
    blocks = []
    for hit in hits:
        where = " > ".join([hit.chunk.doc, *hit.chunk.heading_path])
        blocks.append(f"[chunk_id: {hit.chunk.chunk_id}] ({where})\n{hit.chunk.text}")
    return "\n\n".join(blocks)


def format_history(history: History) -> str:
    return "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history) or EMPTY


class AgentRunner:
    def __init__(
        self,
        agent: AgentDef,
        base_dir: Path,
        config: ForgeConfig,
        embedder: Embedder,
        tool_context: ToolContext | None = None,
    ) -> None:
        self.agent = agent
        self.config = config
        docs_dir = agent.corpus_dir(base_dir)
        chunking = agent.retrieval.chunking
        chunks, self.skipped_files = load_corpus_chunks(
            docs_dir, chunking.size_tokens, chunking.overlap_tokens, agent.retrieval.csv_mode
        )
        self.index = HybridIndex.build(
            chunks, embedder, config.embedding_model,
            cache_dir=config.index_cache_dir, query_prefix=config.embedding_query_prefix,
        )
        context = tool_context or ToolContext()
        if context.corpus_dir is None:
            context.corpus_dir = docs_dir
        self.tools = build_tools(agent.tools, context)
        self.system_prompt = compile_system_prompt(agent)

    # -- public ---------------------------------------------------------------

    def retrieve(self, question: str, history: History | None = None) -> tuple[str, list[Hit]]:
        params = self.agent.retrieval
        query = question
        if params.query_rewrite:
            model = self.agent.model
            query = call_structured(
                "runner.query_rewrite", "runner/query_rewrite",
                {"history": format_history(history or []), "question": question},
                RewrittenQuery, model.name, model.temperature, seed=model.seed,
                fallback=RewrittenQuery(query=question),
            ).query
        hits = self.index.search(
            query, params.top_k, params.hybrid.dense_weight, params.hybrid.bm25_weight, params.min_score
        )
        return query, hits

    def ask(self, question: str, history: History | None = None) -> RunResult:
        started = time.perf_counter()
        history = history or []
        query, hits = self.retrieve(question, history)
        context = format_context(hits)
        records: list[ToolCallRecord] = []
        trace: list[dict[str, Any]] = []
        feedback = EMPTY
        repaired = False
        max_tools = self.config.limits.max_tool_calls_per_turn

        step_budget = max_tools + 3
        for _ in range(step_budget):
            step, fell_back = self._step(question, history, context, records, feedback)
            trace.append({"step": step.model_dump(mode="json"), "model_output_invalid": fell_back})
            if fell_back:
                return self._finish(question, query, hits, step, records, trace, started,
                                    extra=[MODEL_OUTPUT_INVALID])
            if step.action == "tool_call":
                if len(records) >= max_tools:
                    feedback = f"Tool call limit ({max_tools}) reached. Give your final answer now."
                    continue
                records.append(self._run_tool(step))
                feedback = EMPTY
                continue
            result = self._finish(question, query, hits, step, records, trace, started)
            if result.issues and not repaired:  # one repair retry, with the problems fed back
                repaired = True
                feedback = "\n".join(f"- {issue}" for issue in result.issues)
                continue
            return result
        return self._finish(
            question, query, hits, AgentStep(action="final", status="refused"), records, trace,
            started, extra=[NO_FINAL_ANSWER],
        )

    # -- internals ------------------------------------------------------------

    def _step(
        self, question: str, history: History, context: str,
        records: list[ToolCallRecord], feedback: str,
    ) -> tuple[AgentStep, bool]:
        model = self.agent.model
        tool_results = (
            "\n".join(f"{i}. {r.tool}({r.args}) -> {r.result}" for i, r in enumerate(records, 1))
            or EMPTY
        )
        sentinel = AgentStep(action="final", status="refused")
        step = call_structured(
            "runner.agent", "runner/agent_turn",
            {
                "system_prompt": self.system_prompt, "history": format_history(history),
                "context": context, "tool_results": tool_results,
                "feedback": feedback, "question": question,
            },
            AgentStep, model.name, model.temperature, seed=model.seed, fallback=sentinel,
        )
        return step, step is sentinel

    def _run_tool(self, step: AgentStep) -> ToolCallRecord:
        tool = self.tools.get(step.tool)
        if tool is None:
            attached = sorted(self.tools) or "none"
            return ToolCallRecord(
                tool=step.tool, args=step.args,
                result=f"ERROR: tool {step.tool!r} is not available. Attached tools: {attached}",
            )
        try:
            result = tool.run(step.args)
        except ToolError as exc:
            result = f"ERROR: {exc}"
        return ToolCallRecord(tool=step.tool, args=step.args, result=result)

    def _finish(
        self, question: str, query: str, hits: list[Hit], step: AgentStep,
        records: list[ToolCallRecord], trace: list[dict[str, Any]], started: float,
        extra: list[str] | None = None,
    ) -> RunResult:
        # tool_calls come from what was actually executed, never from model claims.
        answer = Answer(
            status=step.status, answer=step.answer, citations=step.citations,
            tool_calls=list(records), clarifying_question=step.clarifying_question,
        )
        issues = list(extra or [])
        if not issues:
            issues = contract_issues(answer, self.index.by_id)
        return RunResult(
            question=question, answer=answer, retrieved=hits, issues=issues, steps=list(trace),
            search_query=query, latency_ms=(time.perf_counter() - started) * 1000,
        )
