"""The patch library (spec §6.6): bounded JSON-Patch operations on ``agent.yaml``, grouped by failure category.

A *patch set* is what one optimizer cycle applies: a few operations aimed at one category. Retrieval patches are
numbers (bounded here); prompt patches name ONE section whose text a critic drafts afterwards (``critic.py``) and
the anti-overfitting guard checks (``guard.py``). ``min_score`` may only move inside the agent's own
``retrieval.min_score_bounds`` (spec §4.4).

Every patch is generic: text patches name a section and a goal (never a case), the few-shot patch is a fixed
domain-free format example, and a patch whose parameters are out of bounds is simply not offered.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from forgeai.optimizer.classify import HALLUCINATION, PROMPT_DEFECT, RETRIEVAL_MISS, WRONG_REFUSAL, Failure
from forgeai.runner.agent_def import AgentDef, FewShot

PROMPT_SECTIONS = ("scope", "citation_rules", "ambiguity_policy", "output_contract", "tool_policy", "grounding_rules", "refusal_policy")
TOP_K_MAX = 10
FEW_SHOT_MAX = 2
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
    if len(after.prompt_sections.few_shot) > max(FEW_SHOT_MAX, len(before.prompt_sections.few_shot)):
        raise ValueError(f"more than {FEW_SHOT_MAX} few-shot examples")
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


def _grounding_candidates(agent: AgentDef, failures: list[Failure]) -> list[PatchSet]:
    cat = HALLUCINATION
    out: list[PatchSet] = []
    if any("expected_refused" in f.tags for f in failures):  # a gap or live-data question was answered
        out.append(PatchSet(f"{cat}/gap_refusal", cat, section="refusal_policy",
                            rationale="a question the documents cannot answer (a gap, live or current data, or an excluded topic) was answered: strengthen the refusal for that class",
                            instruction="Add a general rule: when a question asks for information the documents do not contain (live or real-time values, "
                                        "figures the documents mark as unavailable, or topics the policy excludes), refuse with status refused, state no "
                                        "figure, and say where the user can look. Keep EVERY listed refusal item exactly as it is."))
    out += [PatchSet(f"{cat}/grounding_rules", cat, section="grounding_rules", rationale="strengthen the grounding rules: quote before you claim",
                    instruction="Make the rules stricter about stating only what a retrieved passage says: require the agent to find the passage "
                                "first, state only claims that passage supports, and say so plainly when the passages do not contain the answer.")]
    high = agent.retrieval.min_score_bounds[1]
    if agent.retrieval.min_score + MIN_SCORE_STEP <= high + 1e-9:
        out.append(PatchSet(f"{cat}/min_score+{MIN_SCORE_STEP}", cat, [_replace("/retrieval/min_score", round(agent.retrieval.min_score + MIN_SCORE_STEP, 4))],
                            "raise the evidence threshold so weakly related chunks are not treated as evidence"))
    if agent.model.temperature > 0:
        out.append(PatchSet(f"{cat}/temperature=0", cat, [_replace("/model/temperature", 0.0)], "sample deterministically"))
    return out


def _refusal_candidates(agent: AgentDef, failures: list[Failure]) -> list[PatchSet]:
    cat = WRONG_REFUSAL
    out: list[PatchSet] = []
    if any("ambiguous" in f.tags for f in failures):
        out.append(PatchSet(f"{cat}/ambiguity_policy", cat, section="ambiguity_policy",
                            rationale="an ambiguous question was refused although the passages support more than one reading",
                            instruction="When the passages support two or more readings of a question, do not refuse: ask ONE clarifying question that names "
                                        "the readings (status needs_clarification), or, if the question allows it, answer each reading briefly. Refuse only "
                                        "when no passage bears on the question."))
    out += [PatchSet(f"{cat}/refusal_policy", cat, section="refusal_policy", rationale="relax the refusal wording for in-scope questions",
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
_SNAPSHOT = ("When an answer comes from a table or file that is a snapshot (it carries a date such as an as-of, valid-on or last-updated field), state that "
             "date in the answer in the form 'as of <date>', next to the value, and say the value may have changed since. Keep the listed topics and exclusions.")
_CITATION_FORMAT = ("Copy each citation's chunk_id exactly as it is written in the [chunk_id: ...] header of the passage you quote (letters, digits and symbols "
                    "unchanged), set doc to that passage's document, and make quote a verbatim span of that same passage. Cite one passage per claim.")


def format_example() -> FewShot:
    """A fixed, domain-free example of the reply format (a made-up passage about nothing in any corpus)."""
    quote = "The sample widget is blue."
    return FewShot(question="(format example only) What colour is the sample widget in the example passage?",
                   reply={"status": "answered", "answer": "The sample widget is blue.",
                          "citations": [{"doc": "example.md", "chunk_id": "example.md#c1", "quote": quote}],
                          "tool_calls": [], "clarifying_question": None})


def _prompt_candidates(agent: AgentDef, failures: list[Failure]) -> list[PatchSet]:
    """One patch per prompt section the failures point at (most failures first); the specific patches (snapshot dates,
    citation format) come before the generic rewrite of their section, and the format-only few-shot comes last."""
    sections = [f.section for f in failures if f.section in _SECTION_INSTRUCTIONS or f.section == "output_contract"]
    counts = {s: sections.count(s) for s in set(sections)}
    tags = {t for f in failures for t in f.tags}
    out: list[PatchSet] = []
    cat = PROMPT_DEFECT
    for section in sorted(counts, key=lambda s: (-counts[s], s)):
        why = f"{counts[section]} failing case(s) point at the {section} section"
        if section == "scope" and "snapshot_date" in tags:
            out.append(PatchSet(f"{cat}/snapshot_dates", cat, section="scope", instruction=_SNAPSHOT, rationale=why + ": a snapshot answer lacks its 'as of' date"))
        if section in ("output_contract", "citation_rules") and ("citation_invalid" in tags or "citation_source" in tags):
            out.append(PatchSet(f"{cat}/citation_format", cat, section="citation_rules", instruction=_CITATION_FORMAT, rationale=why + ": citations name the wrong chunk or source"))
        if section in _SECTION_INSTRUCTIONS:
            out.append(PatchSet(f"{cat}/{section}", cat, section=section, instruction=_SECTION_INSTRUCTIONS[section], rationale=why))
    if tags & {"contract_invalid", "citation_invalid"} and len(agent.prompt_sections.few_shot) < FEW_SHOT_MAX:
        shots = [*agent.prompt_sections.few_shot, format_example()]
        out.append(PatchSet(f"{cat}/few_shot_format", cat, [_replace("/prompt_sections/few_shot", [x.model_dump(mode="json") for x in shots])],
                            "answers break the reply format: add one format-only example (a made-up passage, no corpus content)"))
    seen: set[str] = set()
    return [p for p in out if not (p.id in seen or seen.add(p.id))]


def candidates(category: str, agent: AgentDef, failures: list[Failure], questions: list[str], *, crowded_out: bool = False,
               exclude: frozenset[str] = frozenset()) -> list[PatchSet]:
    """The patch sets of ``category`` that are still inside their bounds, most promising first, without those in ``exclude``."""
    mine = [f for f in failures if f.category == category]
    if category == RETRIEVAL_MISS:
        found = _retrieval_candidates(agent, mine, questions, crowded_out)
    elif category == HALLUCINATION:
        found = _grounding_candidates(agent, mine)
    elif category == WRONG_REFUSAL:
        found = _refusal_candidates(agent, mine)
    else:
        found = _prompt_candidates(agent, mine)
    return [p for p in found if p.id not in exclude]


def with_text(patch: PatchSet, text: str) -> PatchSet:
    """The text patch with its drafted section text turned into the JSON-Patch operation."""
    return PatchSet(patch.id, patch.category, [_replace(f"/prompt_sections/{patch.section}", text)], patch.rationale, patch.section, patch.instruction)
