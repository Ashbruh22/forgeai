"""Stage 5: build the eval set (spec §6.1).

Over-generate about 2x, validate, filter, de-duplicate, then select toward the composition target
(32; hard floor 20) and split 70/30 into dev and test. The generator reads only the
requirements and the documents. Every rejection is kept, with its reason.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel

from forgeai.config import ForgeConfig
from forgeai.embeddings import Embedder
from forgeai.evals.consistency import assert_cases_consistent, violation
from forgeai.evals.corpus import EvalCorpus
from forgeai.evals.coverage import Item, build_items, build_matrix, case_covers, uncovered_items
from forgeai.evals.generators import Candidate, Generator
from forgeai.evals.matching import fact_in_text, quantity_match
from forgeai.evals.models import EvalCase, KeyFact, RejectedCase
from forgeai.guard.guard import evaluate_probes
from forgeai.intake.schema import Requirements
from forgeai.llm import call_structured
from forgeai.prompts import PROMPTS_DIR
from forgeai.textmatch import content_terms

TARGET_TOTAL = 32
BASE_TARGETS = {"answerable": 12, "multidoc": 5, "ambiguous": 4, "out_of_scope": 7, "tool_required": 4}
OVERGENERATION = 2
TEST_FRACTION = 0.3
TYPE_CODE = {"answerable": "A", "multidoc": "M", "ambiguous": "X", "out_of_scope": "O", "tool_required": "T"}


class AnswerFromContext(BaseModel):
    answer: str = ""


class AnswerBearing(BaseModel):
    reason: str = ""  # first, so the verdict follows the reasoning
    needed: bool = False


@dataclass
class EvalSetResult:
    cases: list[EvalCase]
    rejected: list[RejectedCase]
    stats: dict[str, Any]
    matrix: dict[str, Any] = field(default_factory=dict)


def targets_for(req: Requirements, has_tables: bool) -> dict[str, int]:
    """Composition target; with no tool capability the tool slots go to answerable and multi-doc (spec §6.1)."""
    caps = set(req.capabilities_needed)
    can_tool = ("arithmetic" in caps) or ("structured_lookup" in caps and has_tables)
    targets = dict(BASE_TARGETS)
    if not can_tool:
        targets["answerable"] += 2
        targets["multidoc"] += 2
        targets["tool_required"] = 0
    return targets


REF_PREFIX = {"gap": "gap", "must_refuse": "refuse", "negative_tool": "negtool"}


def ref_of(case: EvalCase) -> str:
    """The requirement item a case was written for, so a rejection can be attributed to it."""
    if case.covers:
        return case.covers[-1] if case.origin == "snapshot" else case.covers[0]
    if case.expected_status == "refused" and case.origin in REF_PREFIX:
        return f"{REF_PREFIX[case.origin]}:{case.topic_ref}"
    return ""


def _reject(cand: Candidate, reason: str, detail: str) -> RejectedCase:
    return RejectedCase(case_type=cand.case.type, question=cand.case.question, reason=reason, detail=detail,
                        ref=ref_of(cand.case))


def restates_the_question(fact: str, question: str) -> bool:
    """A fact most of whose words the question already says is context, not new information to give."""
    if re.search(r"\d", fact):  # a number or a date is the information itself, never a restatement
        return False
    terms = content_terms(fact)
    return bool(terms) and len(terms & content_terms(question)) / len(terms) >= 0.8


def prune_facts(candidates: list[Candidate], config: ForgeConfig) -> tuple[list[Candidate], list[RejectedCase]]:
    """Keep only ANSWER-BEARING key facts (P5.1 correction 5): an answer without the fact would be wrong or incomplete.

    Code first drops facts that merely restate the question; the judge model then rules on each remaining
    fact (yes or no, one sentence of reasoning). A case left with no fact is rejected. The pass threshold of the
    answerability check is NOT relaxed: detail facts are removed instead.
    """
    judge = config.judge_model
    kept: list[Candidate] = []
    rejected: list[RejectedCase] = []
    for cand in candidates:
        case = cand.case
        if not cand.passages or case.type != "answerable":
            kept.append(cand)
            continue
        dropped: list[dict[str, str]] = []
        keep: list[KeyFact] = []
        for position, fact in enumerate(case.key_facts):
            if fact.kind == "snapshot_date":
                keep.append(fact)
            elif restates_the_question(fact.text, case.question):
                dropped.append({"fact": fact.text, "why": "restates the question"})
            elif position == 0:
                keep.append(fact)  # the first fact IS the answer the question was written for; the judge rules on the extras
            else:
                span = next((s.span_text for s in case.seed_spans if fact_in_text(fact.kind, fact.text, s.span_text, strict=False)), "")
                verdict = call_structured(
                    "evals.answer_bearing", "evals/answer_bearing",
                    {"question": case.question, "fact": fact.text, "span": span or cand.passages[0]},
                    AnswerBearing, judge.name, judge.temperature, seed=judge.seed, fallback=AnswerBearing(needed=True))
                if verdict.needed:
                    keep.append(fact)
                else:
                    dropped.append({"fact": fact.text, "why": verdict.reason.strip()[:160]})
        if not [f for f in keep if f.kind != "snapshot_date"]:
            rejected.append(_reject(cand, "no answer-bearing key fact",
                                    f"every fact was detail: {[d['fact'] for d in dropped]}"))
            continue
        case.key_facts = keep
        case.validation["dropped_detail_facts"] = dropped
        kept.append(cand)
    return kept, rejected


def judge_answerability(candidates: list[Candidate], config: ForgeConfig) -> tuple[list[Candidate], list[RejectedCase]]:
    """Independent answerability: the judge model answers from the seed passage(s) alone and must recover the facts."""
    judge = config.judge_model
    kept: list[Candidate] = []
    rejected: list[RejectedCase] = []
    for cand in candidates:
        if not cand.passages:
            kept.append(cand)
            continue
        answer = call_structured(
            "evals.answerability", "evals/answer_from_context",
            {"passages": "\n\n".join(cand.passages), "question": cand.case.question},
            AnswerFromContext, judge.name, judge.temperature, seed=judge.seed, fallback=AnswerFromContext(),
        ).answer
        facts = [f for f in cand.case.key_facts if f.kind not in ("snapshot_date", "relative_date")]
        recovered = [f for f in facts if fact_in_text(f.kind, f.text, answer, strict=False)
                     or (f.kind == "phrase" and quantity_match(f.text, answer))]
        # Every key fact is answer-bearing by now (``prune_facts``), so the PRIMARY fact must be recovered
        # plus at least half of all of them. The threshold is unchanged from the first P5 build.
        ok = bool(facts) and facts[0] in recovered and len(recovered) * 2 >= len(facts)
        if cand.fact_groups:  # multi-document: the answer must draw on EVERY document, or one document was enough
            ok = ok and all(any(f in recovered for f in group) for group in cand.fact_groups)
        cand.case.validation["independent_answerable"] = ok
        cand.case.validation["facts_recovered_by_judge"] = f"{len(recovered)}/{len(facts)}"
        if ok:
            kept.append(cand)
        else:
            rejected.append(_reject(
                cand, "independent answerability failed",
                f"judge recovered {len(recovered)} of {len(facts)} key facts from the seed alone; facts {[f.text for f in facts]}; judge answered: {answer[:160]!r}"))
    return kept, rejected


def probe_must_fail(candidates: list[Candidate], corpus: EvalCorpus, config: ForgeConfig) -> tuple[list[Candidate], list[RejectedCase]]:
    """Refusal cases the corpus must NOT be able to answer: the guard's two-check probe has to fail."""
    probed = [c for c in candidates if c.probe]
    results = evaluate_probes([c.probe for c in probed], corpus.index, config) if probed else []  # type: ignore[misc]
    failing = {id(c): r for c, r in zip(probed, results, strict=True)}
    kept: list[Candidate] = []
    rejected: list[RejectedCase] = []
    for cand in candidates:
        result = failing.get(id(cand))
        if result is None:
            kept.append(cand)
        elif result.supported:
            rejected.append(_reject(cand, "the corpus can answer it", f"{result.chunk_id}: {result.span[:100]}"))
        else:
            cand.case.validation["guard_probe_failed"] = True
            cand.case.validation["max_dense_similarity"] = round(result.max_dense, 3)
            kept.append(cand)
    return kept, rejected


def dedupe(candidates: list[Candidate], embedder: Embedder, threshold: float) -> tuple[list[Candidate], list[RejectedCase]]:
    """Embedding similarity above ``threshold`` keeps only the first of a pair (a merged question and its parents excepted)."""
    if not candidates:
        return [], []
    vectors = embedder.encode([c.case.question for c in candidates])
    kept: list[int] = []
    rejected: list[RejectedCase] = []

    def related(a: EvalCase, b: EvalCase) -> bool:
        return a.question in b.provenance.get("composed_from", []) or b.question in a.provenance.get("composed_from", [])

    for i, cand in enumerate(candidates):
        twin = None if cand.case.aspect else next((j for j in kept if float(vectors[i] @ vectors[j]) > threshold
                     and not related(cand.case, candidates[j].case)), None)
        if twin is None:
            kept.append(i)
        else:
            rejected.append(_reject(cand, "duplicate (embedding similarity)", f"similar to {candidates[twin].case.question!r}"))
    return [candidates[i] for i in kept], rejected


def select(candidates: list[Candidate], target: int) -> list[Candidate]:
    """Up to ``target`` candidates, taking turns across groups (documents, origins) for diversity."""
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for cand in candidates:
        groups[cand.group].append(cand)
    chosen: list[Candidate] = []
    while len(chosen) < target and any(groups.values()):
        for key in list(groups):
            if groups[key] and len(chosen) < target:
                chosen.append(groups[key].pop(0))
    return chosen


def assign_split(cases: list[EvalCase], items: list[Item] | None = None,
                 evidence: dict[str, list[tuple[str, int, int]]] | None = None) -> dict[str, list[str]]:
    """Stratified by type: about 30% of each type goes to test, picked by a stable hash of the id.

    The optimizer only ever sees dev, so every requirement item must also be covered by a DEV case: where a
    stratified split would leave an item with only test cases, one covering case moves to dev (and a dev case of the
    same type that no item depends on moves to test, to keep the proportions).
    """
    by_type: dict[str, list[EvalCase]] = defaultdict(list)
    for case in cases:
        by_type[case.type].append(case)
    for group in by_type.values():
        ranked = sorted(group, key=lambda c: hashlib.sha1(c.id.encode()).hexdigest())
        n_test = round(len(group) * TEST_FRACTION) if len(group) >= 3 else 0
        for position, case in enumerate(ranked):
            case.split = "test" if position < n_test else "dev"
    for item in [i for i in items or [] if i.kind != "ask"]:
        covering = [c for c in cases if case_covers(item, c, evidence or {})]
        if covering and not any(c.split == "dev" for c in covering):
            moved = covering[0]
            moved.split = "dev"
            sacrifice = next((c for c in by_type[moved.type] if c.split == "dev" and c is not moved
                              and not any(case_covers(i, c, evidence or {}) and sum(case_covers(i, o, evidence or {}) and o.split == "dev"
                                                                                    for o in cases) == 1
                                          for i in items or [] if i.kind != "ask")), None)
            if sacrifice is not None:
                sacrifice.split = "test"
    return {"dev": [c.id for c in cases if c.split == "dev"], "test": [c.id for c in cases if c.split == "test"]}


MAX_ROUNDS = 6  # round 0 over-generates 2x; later rounds target what is still missing
ENOUGH = 28  # stop topping up types once this many validated cases exist AND every requirement item is covered


def _evidence_ranges(req: Requirements, corpus: EvalCorpus) -> dict[str, list[tuple[str, int, int]]]:
    by_id = {c.chunk_id: c for c in corpus.chunks}
    ranges: dict[str, list[tuple[str, int, int]]] = {}
    for topic in req.must_answer_topics:
        ranges[topic.id] = [(by_id[e].doc, by_id[e].char_start, by_id[e].char_end) for e in topic.evidence
                            if e in by_id and by_id[e].char_start is not None and by_id[e].char_end is not None]
    return ranges


def _judge_stage(cands: list[Candidate], corpus: EvalCorpus, config: ForgeConfig) -> tuple[list[Candidate], list[RejectedCase]]:
    """Everything that needs the judge model, batched so the model loads once: facts, answerability, refusals."""
    rejected: list[RejectedCase] = []
    with_passages = [c for c in cands if c.passages]
    others = [c for c in cands if not c.passages]
    pruned, rej = prune_facts(with_passages, config)
    rejected += rej
    kept_answerable, rej = judge_answerability(pruned, config)
    rejected += rej
    kept_refusals, rej = probe_must_fail([c for c in others if c.case.type == "out_of_scope"], corpus, config)
    rejected += rej
    return kept_answerable + kept_refusals + [c for c in others if c.case.type != "out_of_scope"], rejected


def _round(req: Requirements, corpus: EvalCorpus, config: ForgeConfig, plan: dict[str, Any], round_index: int,
           items: list[Item], singles: list[Candidate], skip_pairs: set[frozenset[str]],
           ) -> tuple[list[Candidate], list[RejectedCase], Counter[str], int]:
    """Generate and validate candidates for ``plan``: undirected counts per type plus directed requirement items."""
    salt = 7000 * round_index
    gen = Generator(req, corpus, config, salt_base=salt)
    if plan.get("answerable"):
        gen.answerable(plan["answerable"])
    for topic_id in plan.get("topics", []):
        gen.out.ref = f"topic:{topic_id}"
        gen.for_topic(topic_id, 3, salt=900 + 10 * round_index)
    gen.out.ref = ""
    if plan.get("aspects") is not None and (round_index == 0 or plan["aspects"]):
        gen.snapshot_cases(only=plan["aspects"] or None, salt=500 + 20 * round_index)
    if plan.get("ambiguous"):
        gen.ambiguous(plan["ambiguous"])
    if "refusals" in plan:
        only = plan["refusals"]  # None: every subject; a set: only these requirement items
        gen.refusals(7, only=only, per_subject=2 if only is None else 3, salt=300 + 20 * round_index)
    if plan.get("tools"):
        gen.tool_cases(plan["tools"], items=[i for i in items if i.id in plan["tool_items"]] if plan.get("tool_items") is not None else items)
    for item in plan.get("computed_items", []):
        gen.out.ref = item.id
        gen.computed(item.capability, item.text, [item.id], 4, salt=600 + 40 * round_index)
    gen.out.ref = ""
    candidates, rejected = gen.out.candidates, list(gen.out.rejected)
    generated = Counter(c.case.type for c in candidates)
    attempts = len(candidates) + len(gen.out.rejected)  # every question the model wrote, kept or not
    kept, rej = _judge_stage(candidates, corpus, config)  # agent calls are done; judge model next
    rejected += rej

    composed: list[Candidate] = []
    if plan.get("multidoc"):  # merge validated single-document cases: the agent model writes, the judge model validates
        pool = singles + [c for c in kept if c.case.type == "answerable"]
        mgen = Generator(req, corpus, config, salt_base=salt + 3000)
        mgen.skip_pairs = skip_pairs
        mgen.multidoc_from(pool, plan["multidoc"])
        made = Counter(c.case.type for c in mgen.out.candidates)
        generated.update(made)
        attempts += len(mgen.out.candidates) + len(mgen.out.rejected)
        rejected += mgen.out.rejected
        composed, rej = judge_answerability(mgen.out.candidates, config)
        rejected += rej
    return kept + composed, rejected, generated, attempts


def _pick_for_coverage(items: list[Item], pool: list[Candidate], evidence: dict[str, list[tuple[str, int, int]]]) -> list[Candidate]:
    """One case for every item that has any, preferring a case that covers several items (fewer cases, more checks)."""
    chosen: list[Candidate] = []
    for item in [i for i in items if i.kind != "ask"]:
        if any(case_covers(item, c.case, evidence) for c in chosen):
            continue
        options = [c for c in pool if c not in chosen and case_covers(item, c.case, evidence)]
        if options:
            chosen.append(max(options, key=lambda c: sum(case_covers(i, c.case, evidence) for i in items if i.kind != "ask")))
    return chosen


def build_eval_set(
    req: Requirements, corpus: EvalCorpus, config: ForgeConfig, embedder: Embedder, prefix: str = "ev", brief: str = "",
) -> EvalSetResult:
    targets = targets_for(req, bool(corpus.tables))
    order = {t: i for i, t in enumerate(BASE_TARGETS)}
    items = build_items(req, brief, embedder)
    evidence = _evidence_ranges(req, corpus)
    validated: list[Candidate] = []
    rejected: list[RejectedCase] = []
    generated: Counter[str] = Counter()
    skip_pairs: set[frozenset[str]] = set()
    rounds = attempts = 0
    for round_index in range(MAX_ROUNDS):
        have = Counter(c.case.type for c in validated)
        deficit = {t: max(0, targets[t] - have[t]) for t in targets}
        missing = uncovered_items([i for i in items if i.kind != "ask"], [c.case for c in validated], evidence)
        if round_index and not missing and (not any(deficit.values()) or len(validated) >= ENOUGH):
            break
        if round_index == 0:
            plan: dict[str, Any] = {
                "answerable": targets["answerable"] * OVERGENERATION, "aspects": set(), "refusals": None,
                "ambiguous": targets["ambiguous"] * OVERGENERATION, "multidoc": targets["multidoc"] * OVERGENERATION,
                "tools": targets["tool_required"] * OVERGENERATION if targets["tool_required"] else 0}
        else:
            ids = {i.id for i in missing}
            plan = {
                "answerable": deficit["answerable"] * OVERGENERATION,
                "topics": [i.topic_id for i in missing if i.kind == "topic"],
                "aspects": {i.id for i in missing if i.kind == "aspect"},
                "refusals": {i.id for i in missing if i.kind in ("gap", "refuse", "negtool")} or None,
                "ambiguous": deficit["ambiguous"] * OVERGENERATION * round_index if deficit["ambiguous"] else 0,
                "multidoc": deficit["multidoc"] * OVERGENERATION if deficit["multidoc"] else 0,
                "computed_items": [i for i in missing if i.kind == "ask_capability"],
                "tools": 0}
            if not plan["refusals"]:
                plan.pop("refusals")
            if deficit["out_of_scope"] and "refusals" not in plan:
                plan["refusals"] = None  # more refusal questions for every subject (a different salt writes different ones)
            if deficit["tool_required"]:
                plan["tools"] = deficit["tool_required"] * OVERGENERATION
            if not ids and not any(plan.get(k) for k in ("answerable", "ambiguous", "multidoc", "tools"))                     and "refusals" not in plan:
                break
        singles = [c for c in validated if c.case.type == "answerable"]
        new, rej, made, tried = _round(req, corpus, config, plan, round_index, items, singles, skip_pairs)
        attempts += tried
        # Consistency with the requirements, then de-duplication against everything kept so far.
        for cand in new:
            problem = violation(cand.case, req)
            if problem:
                rejected.append(_reject(cand, "contradicts the requirements", problem))
            else:
                validated.append(cand)
        validated.sort(key=lambda c: order[c.case.type])
        validated, dupes = dedupe(validated, embedder, config.thresholds.dedup_similarity)
        rejected += rej + dupes
        generated.update(made)
        rounds += 1

    # Select: first one case for every requirement item (coverage), then toward the composition target.
    chosen: list[Candidate] = _pick_for_coverage(items, validated, evidence)
    for case_type, target in targets.items():
        have_type = sum(c.case.type == case_type for c in chosen)
        chosen += select([c for c in validated if c.case.type == case_type and c not in chosen], max(0, target - have_type))
    leftovers = [c for c in validated if c not in chosen and c.case.type in ("answerable", "multidoc", "out_of_scope")]
    chosen += select(leftovers, max(0, TARGET_TOTAL - len(chosen)))

    counters: Counter[str] = Counter()
    cases: list[EvalCase] = []
    for cand in sorted(chosen, key=lambda c: order[c.case.type]):
        counters[cand.case.type] += 1
        cand.case.id = f"{prefix}-{TYPE_CODE[cand.case.type]}{counters[cand.case.type]:02d}"
        cases.append(cand.case)
    assert_cases_consistent(cases, req)  # fail loudly: this would be a generator bug
    split = assign_split(cases, items, evidence)

    notes = _explanations(items, rejected)
    matrix = build_matrix(items, cases, evidence, rejection_notes=notes)
    for row in matrix["items"]:
        covering = [c for c in cases if c.id in row["cases"]]
        row["dev_cases"] = [c.id for c in covering if c.split == "dev"]
        row["test_cases"] = [c.id for c in covering if c.split == "test"]
    matrix["summary"]["uncovered_in_dev"] = [r["id"] for r in matrix["items"] if r["status"] == "COVERED" and not r["dev_cases"]]
    stats = {
        "target_total": TARGET_TOTAL, "floor": config.limits.min_eval_cases, "valid_cases": len(cases),
        "floor_met": len(cases) >= config.limits.min_eval_cases, "generation_rounds": rounds,
        "targets": targets, "generated_candidates": dict(generated),
        "validated_after_filters": dict(Counter(c.case.type for c in validated)),
        "selected": dict(counters), "split": {k: len(v) for k, v in split.items()},
        "rejections_by_reason": dict(Counter(r.reason for r in rejected)),
        "candidates_attempted": attempts, "rejection_rate": round(len(rejected) / max(attempts, 1), 3),
        "shortfall": {t: max(0, targets[t] - counters.get(t, 0)) for t in targets if counters.get(t, 0) < targets[t]},
        "coverage": matrix["summary"],
        "key_facts_per_case": dict(Counter(len(c.key_facts) for c in cases)),
    }
    return EvalSetResult(cases, rejected, stats, matrix)


def _explanations(items: list[Item], rejected: list[RejectedCase]) -> dict[str, str]:
    """Why an item may have no case: what happened to the candidates written for it."""
    notes: dict[str, str] = {}
    for item in items:
        reasons = Counter(r.reason for r in rejected if r.ref == item.id)
        if reasons:
            top = ", ".join(f"{reason} x{n}" for reason, n in reasons.most_common(3))
            notes[item.id] = f"{sum(reasons.values())} candidates were written for this item and all were rejected: {top}"
        else:
            notes[item.id] = "no candidate was produced for this item"
    return notes


def write_eval_set(result: EvalSetResult, out_dir: Path) -> None:
    """Write ``03_evals/``: eval_set.jsonl, rejected_cases.jsonl, split.json, coverage_matrix.json, prompts_used/, stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_set.jsonl").write_text(
        "".join(c.model_dump_json() + "\n" for c in result.cases), encoding="utf-8")
    (out_dir / "rejected_cases.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in result.rejected), encoding="utf-8")
    split = {"dev": [c.id for c in result.cases if c.split == "dev"], "test": [c.id for c in result.cases if c.split == "test"]}
    (out_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (out_dir / "generation_stats.json").write_text(json.dumps(result.stats, indent=2), encoding="utf-8")
    (out_dir / "coverage_matrix.json").write_text(json.dumps(result.matrix, indent=2), encoding="utf-8")
    prompts = out_dir / "prompts_used"
    prompts.mkdir(exist_ok=True)
    for stale in prompts.glob("*.md"):  # a prompt that is no longer used must not look as if it was
        stale.unlink()
    versions = {}
    for path in sorted((PROMPTS_DIR / "evals").glob("*.md")) + [PROMPTS_DIR / "guard" / "near_miss.md"]:
        shutil.copy(path, prompts / path.name)
        versions[f"{path.parent.name}/{path.stem}"] = int(path.read_text(encoding="utf-8").split("version:")[1].split()[0])
    (prompts / "versions.yaml").write_text(yaml.safe_dump(versions), encoding="utf-8")


def load_eval_set(path: Path) -> list[EvalCase]:
    return [EvalCase.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
