"""The patch library (spec §6.6): bounded JSON-Patch operations on ``agent.yaml``, grouped by failure category.

A *patch set* is what one optimizer cycle applies: a few operations aimed at one category. Retrieval patches are
numbers (bounded here); prompt patches name ONE section whose text a critic drafts afterwards (``critic.py``) and
the anti-overfitting guard checks (``guard.py``). ``min_score`` may only move inside the agent's own
``retrieval.min_score_bounds`` (spec §4.4).

Not implemented (listed in docs/DECISIONS.md section K): the format-only few-shot patch and the explicit gap-refusal
patch of the spec's table.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from forgeai.optimizer.classify import HALLUCINATION, PROMPT_DEFECT, RETRIEVAL_MISS, WRONG_REFUSAL, Failure
from forgeai.runner.agent_def import AgentDef

PROMPT_SECTIONS = ("scope", "citation_rules", "ambiguity_policy", "output_contract", "tool_policy", "grounding_rules", "refusal_policy")
TOP_K_MAX = 10
CHUNK_SIZE_BOUNDS = (120, 800)
MIN_SCORE_STEP = 0.05
BM25_WEIGHT_MAX = 0.8
_CODE = re.compile(r"(?<![\w-])[A-Za-z]{1,6}[-_]?\d[\w-]*")


@dataclass
class PatchSet:
    id: str  # stable name, e.g. "retrieval_miss/top_k+2"
    category: str
    ops: list[dict[str, Any]] = field(default_factory=list)  # JSON-Patch operations (empty while a text patch awaits its text)
    rationale: str = ""
    section: str = ""  # a text patch: the one prompt section to rewrite
    instruction: str = ""  # what the rewrite must achieve (general, never case specific)

    @property
    def needs_text(self) -> bool:
        return bool(self.section)


def to_dict(agent: AgentDef) -> dict[str, Any]:
    return agent.model_dump(mode="json")


def apply_ops(agent: AgentDef, ops: list[dict[str, Any]]) -> AgentDef:
    """A copy of ``agent`` with the JSON-Patch operations (replace/add) applied; validated by the schema afterwards."""
    data = copy.deepcopy(to_dict(agent))
    for op in ops:
        if op["op"] not in ("replace", "add"):
            raise ValueError(f"unsupported patch op {op['op']!r}")
        parts = [p for p in op["path"].split("/") if p]
        target = data
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = op["value"]
    patched = AgentDef.model_validate(data)
    check_bounds(agent, patched)
    return patched


def check_bounds(before: AgentDef, after: AgentDef) -> None:
    """Raise if a patch leaves the allowed range (the patch library never produces these; this is the backstop)."""
    r0, r1 = before.retrieval, after.retrieval
    low, high = r0.min_score_bounds
    if not low - 1e-9 <= r1.min_score <= high + 1e-9:
        raise ValueError(f"min_score {r1.min_score} is outside its bounds {r0.min_score_bounds}")
    if not 1 <= r1.top_k <= TOP_K_MAX:
        raise ValueError(f"top_k {r1.top_k} is outside 1..{TOP_K_MAX}")
    if not CHUNK_SIZE_BOUNDS[0] <= r1.chunking.size_tokens <= CHUNK_SIZE_BOUNDS[1]:
        raise ValueError(f"chunk size {r1.chunking.size_tokens} is outside {CHUNK_SIZE_BOUNDS}")
    if r1.chunking.overlap_tokens > r1.chunking.size_tokens * 0.4:
        raise ValueError("chunk overlap above 40% of the chunk size")
    if not 0 <= r1.hybrid.bm25_weight <= BM25_WEIGHT_MAX or abs(r1.hybrid.bm25_weight + r1.hybrid.dense_weight - 1.0) > 1e-6:
        raise ValueError("hybrid weights out of bounds or not summing to 1")
    if not 0 <= r1.reserve_prose <= r1.top_k // 2:
        raise ValueError("reserve_prose above half of top_k")
    if not 0.0 <= after.model.temperature <= before.model.temperature + 1e-9:
        raise ValueError("a patch may only lower the temperature")


def _replace(path: str, value: Any) -> dict[str, Any]:
    return {"op": "replace", "path": path, "value": value}


def _has_exact_terms(questions: list[str]) -> bool:
    return any(_CODE.search(q) for q in questions)


def _retrieval_candidates(agent: AgentDef, failures: list[Failure], questions: list[str], crowded_out: bool) -> list[PatchSet]:
    r = agent.retrieval
    out: list[PatchSet] = []
    cat = RETRIEVAL_MISS
    if crowded_out and r.reserve_prose < r.top_k // 2:
        keep = min(max(r.reserve_prose, 0) + 1, r.top_k // 2)
        out.append(PatchSet(f"{cat}/reserve_prose={keep}", cat, [_replace("/retrieval/reserve_prose", keep)],
                            "the failing cases' documents are prose but the retrieved set held table rows only: keep a slot for prose (source diversity)"))
    if r.top_k + 2 <= TOP_K_MAX:
        ops = [_replace("/retrieval/top_k", r.top_k + 2)]
        why = "the gold chunk was not among the top-k chunks: widen the window by two"
        if _has_exact_terms(questions) and r.hybrid.bm25_weight + 0.2 <= BM25_WEIGHT_MAX:
            bm25 = round(r.hybrid.bm25_weight + 0.2, 2)
            ops += [_replace("/retrieval/hybrid/bm25_weight", bm25), _replace("/retrieval/hybrid/dense_weight", round(1 - bm25, 2))]
            why += "; the questions carry exact codes, so shift the hybrid weight toward BM25"
        out.append(PatchSet(f"{cat}/top_k+2", cat, ops, why))
    if not r.query_rewrite:
        out.append(PatchSet(f"{cat}/query_rewrite", cat, [_replace("/retrieval/query_rewrite", True)],
                            "rewrite the question into a standalone search query before retrieving"))
    for factor, label in ((1.3, "+30%"), (0.7, "-30%")):
        size = round(r.chunking.size_tokens * factor)
        if CHUNK_SIZE_BOUNDS[0] <= size <= CHUNK_SIZE_BOUNDS[1]:
            out.append(PatchSet(f"{cat}/chunk_size{label}", cat, [_replace("/retrieval/chunking/size_tokens", size)],
                                f"change the chunk size {label} so a passage is not split from its context (or a chunk dilutes it)"))
    low, _ = r.min_score_bounds
    if r.min_score - MIN_SCORE_STEP >= low - 1e-9:
        out.append(PatchSet(f"{cat}/min_score-{MIN_SCORE_STEP}", cat, [_replace("/retrieval/min_score", round(r.min_score - MIN_SCORE_STEP, 4))],
                            "lower the evidence threshold so a borderline chunk is not cut"))
    if r.chunking.overlap_tokens + 20 <= r.chunking.size_tokens * 0.4:
        out.append(PatchSet(f"{cat}/overlap+20", cat, [_replace("/retrieval/chunking/overlap_tokens", r.chunking.overlap_tokens + 20)],
                            "more overlap between neighbouring chunks"))
    return out


def _grounding_candidates(agent: AgentDef) -> list[PatchSet]:
    cat = HALLUCINATION
    out = [PatchSet(f"{cat}/grounding_rules", cat, section="grounding_rules", rationale="strengthen the grounding rules: quote before you claim",
                    instruction="Make the rules stricter about stating only what a retrieved passage says: require the agent to find the passage "
                                "first, state only claims that passage supports, and say so plainly when the passages do not contain the answer.")]
    high = agent.retrieval.min_score_bounds[1]
    if agent.retrieval.min_score + MIN_SCORE_STEP <= high + 1e-9:
        out.append(PatchSet(f"{cat}/min_score+{MIN_SCORE_STEP}", cat, [_replace("/retrieval/min_score", round(agent.retrieval.min_score + MIN_SCORE_STEP, 4))],
                            "raise the evidence threshold so weakly related chunks are not treated as evidence"))
    if agent.model.temperature > 0:
        out.append(PatchSet(f"{cat}/temperature=0", cat, [_replace("/model/temperature", 0.0)], "sample deterministically"))
    return out


def _refusal_candidates(agent: AgentDef) -> list[PatchSet]:
    cat = WRONG_REFUSAL
    out = [PatchSet(f"{cat}/refusal_policy", cat, section="refusal_policy", rationale="relax the refusal wording for in-scope questions",
                    instruction="Make the refusal policy refuse only what it lists or what the passages clearly do not cover, and answer from the "
                                "passages whenever they contain the answer. Keep EVERY listed refusal item exactly as it is.")]
    low, _ = agent.retrieval.min_score_bounds
    if agent.retrieval.min_score - MIN_SCORE_STEP >= low - 1e-9:
        out.append(PatchSet(f"{cat}/min_score-{MIN_SCORE_STEP}", cat, [_replace("/retrieval/min_score", round(agent.retrieval.min_score - MIN_SCORE_STEP, 4))],
                            "lower the evidence threshold so relevant chunks are not cut"))
    return out


_SECTION_INSTRUCTIONS = {
    "scope": "State what an answer must contain: the specific value asked for AND any context the data itself carries that a reader needs "
             "(for example a date a table row is valid on), and keep the listed topics and exclusions as they are.",
    "citation_rules": "Require every answer to cite, for each document it relies on, the chunk that contains the fact, with a verbatim quote.",
    "ambiguity_policy": "When one question has two or more readings that the documents answer differently, ask one clarifying question instead of choosing; "
                        "when it has one answer, answer it.",
    "output_contract": "Make the required reply format clearer: every field, the allowed status values, and that citations quote the passage.",
    "tool_policy": "State when to call each attached tool, and that a value read from a table or computed must come from the tool result.",
}


def _prompt_candidates(agent: AgentDef, failures: list[Failure]) -> list[PatchSet]:
    sections = [s for s, _ in sorted(((f.section, 0) for f in failures if f.section in _SECTION_INSTRUCTIONS), key=lambda p: p[0])]
    counts = {s: sections.count(s) for s in set(sections)}
    return [PatchSet(f"{PROMPT_DEFECT}/{s}", PROMPT_DEFECT, section=s, instruction=_SECTION_INSTRUCTIONS[s],
                     rationale=f"{counts[s]} failing case(s) point at the {s} section")
            for s in sorted(counts, key=lambda s: (-counts[s], s))]


def candidates(category: str, agent: AgentDef, failures: list[Failure], questions: list[str], *, crowded_out: bool = False,
               exclude: frozenset[str] = frozenset()) -> list[PatchSet]:
    """The patch sets of ``category`` that are still inside their bounds, most promising first, without those in ``exclude``."""
    mine = [f for f in failures if f.category == category]
    if category == RETRIEVAL_MISS:
        found = _retrieval_candidates(agent, mine, questions, crowded_out)
    elif category == HALLUCINATION:
        found = _grounding_candidates(agent)
    elif category == WRONG_REFUSAL:
        found = _refusal_candidates(agent)
    else:
        found = _prompt_candidates(agent, mine)
    return [p for p in found if p.id not in exclude]


def with_text(patch: PatchSet, text: str) -> PatchSet:
    """The text patch with its drafted section text turned into the JSON-Patch operation."""
    return PatchSet(patch.id, patch.category, [_replace(f"/prompt_sections/{patch.section}", text)], patch.rationale, patch.section, patch.instruction)
