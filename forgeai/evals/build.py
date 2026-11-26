"""Stage 5: build the eval set (spec §6.1).

Over-generate about 2x, validate, filter, de-duplicate, then select toward the composition target
(32; hard floor 20) and split 70/30 into dev and test. The generator reads only the
requirements and the documents. Every rejection is kept, with its reason.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel

from forgeai.config import ForgeConfig
from forgeai.embeddings import Embedder
from forgeai.evals.consistency import assert_cases_consistent, violation
from forgeai.evals.corpus import EvalCorpus
from forgeai.evals.generators import Candidate, Generator
from forgeai.evals.matching import fact_in_text
from forgeai.evals.models import EvalCase, RejectedCase
from forgeai.guard.guard import evaluate_probes
from forgeai.intake.schema import Requirements
from forgeai.llm import call_structured
from forgeai.prompts import PROMPTS_DIR

TARGET_TOTAL = 32
BASE_TARGETS = {"answerable": 12, "multidoc": 5, "ambiguous": 4, "out_of_scope": 7, "tool_required": 4}
OVERGENERATION = 2
TEST_FRACTION = 0.3
TYPE_CODE = {"answerable": "A", "multidoc": "M", "ambiguous": "X", "out_of_scope": "O", "tool_required": "T"}


class AnswerFromContext(BaseModel):
    answer: str = ""


@dataclass
class EvalSetResult:
    cases: list[EvalCase]
    rejected: list[RejectedCase]
    stats: dict[str, Any]


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
        facts = [f for f in cand.case.key_facts if f.kind != "snapshot_date"]
        recovered = [f for f in facts if fact_in_text(f.kind, f.text, answer, strict=False)]
        # The model lists at most two facts and the second is often a detail the question does not
        # ask for, so a judge answer that omits it is not evidence of a bad question. Every fact is
        # still anchored in the seed span; here the PRIMARY fact must be recovered, plus at least half.
        ok = bool(facts) and facts[0] in recovered and len(recovered) * 2 >= len(facts)
        if cand.fact_groups:  # multi-document: the answer must draw on EVERY document, or one document was enough
            ok = ok and all(any(f in recovered for f in group) for group in cand.fact_groups)
        cand.case.validation["independent_answerable"] = ok
        cand.case.validation["facts_recovered_by_judge"] = f"{len(recovered)}/{len(facts)}"
        if ok:
            kept.append(cand)
        else:
            rejected.append(RejectedCase(
                case_type=cand.case.type, question=cand.case.question, reason="independent answerability failed",
                detail=f"judge recovered {len(recovered)} of {len(facts)} key facts from the seed alone; facts {[f.text for f in facts]}; judge answered: {answer[:160]!r}"))
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
            rejected.append(RejectedCase(
                case_type="out_of_scope", question=cand.case.question, reason="the corpus can answer it",
                detail=f"{result.chunk_id}: {result.span[:100]}"))
        else:
            cand.case.validation["guard_probe_failed"] = True
            cand.case.validation["max_dense_similarity"] = round(result.max_dense, 3)
            kept.append(cand)
    return kept, rejected


def dedupe(candidates: list[Candidate], embedder: Embedder, threshold: float) -> tuple[list[Candidate], list[RejectedCase]]:
    """Embedding similarity above ``threshold`` keeps only the first of a pair."""
    if not candidates:
        return [], []
    vectors = embedder.encode([c.case.question for c in candidates])
    kept: list[int] = []
    rejected: list[RejectedCase] = []
    for i, cand in enumerate(candidates):
        twin = next((j for j in kept if float(vectors[i] @ vectors[j]) > threshold), None)
        if twin is None:
            kept.append(i)
        else:
            rejected.append(RejectedCase(
                case_type=cand.case.type, question=cand.case.question, reason="duplicate (embedding similarity)",
                detail=f"similar to {candidates[twin].case.question!r}"))
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


def assign_split(cases: list[EvalCase]) -> dict[str, list[str]]:
    """Stratified by type: about 30% of each type goes to test, picked by a stable hash of the id."""
    split: dict[str, list[str]] = {"dev": [], "test": []}
    by_type: dict[str, list[EvalCase]] = defaultdict(list)
    for case in cases:
        by_type[case.type].append(case)
    for group in by_type.values():
        ranked = sorted(group, key=lambda c: hashlib.sha1(c.id.encode()).hexdigest())
        n_test = round(len(group) * TEST_FRACTION) if len(group) >= 3 else 0
        for position, case in enumerate(ranked):
            case.split = "test" if position < n_test else "dev"
            split[case.split].append(case.id)
    return split


MAX_ROUNDS = 6  # round 0 over-generates 2x; later rounds top up only the types that fell short
ENOUGH = 28  # stop topping up once this many validated cases exist (the floor is 20, the target 32)


def _round(req: Requirements, corpus: EvalCorpus, config: ForgeConfig, wanted: dict[str, int], round_index: int
           ) -> tuple[list[Candidate], list[RejectedCase], Counter[str], int]:
    """Generate and model-validate candidates for ``wanted`` (a count per type, already over-generated)."""
    gen = Generator(req, corpus, config, salt_base=7000 * round_index)
    if wanted.get("answerable"):
        gen.answerable(wanted["answerable"])
    if round_index == 0:
        gen.snapshot_cases()
    for name in ("multidoc", "ambiguous"):
        if wanted.get(name):
            getattr(gen, name)(wanted[name])
    if wanted.get("out_of_scope"):
        gen.refusals(wanted["out_of_scope"])
    if wanted.get("tool_required"):
        gen.tool_cases(wanted["tool_required"])
    candidates, rejected = gen.out.candidates, list(gen.out.rejected)
    generated = Counter(c.case.type for c in candidates)
    with_passages = [c for c in candidates if c.passages]
    others = [c for c in candidates if not c.passages]
    kept_answerable, rej = judge_answerability(with_passages, config)  # agent calls are done; judge model next
    rejected += rej
    kept_refusals, rej = probe_must_fail([c for c in others if c.case.type == "out_of_scope"], corpus, config)
    rejected += rej
    attempts = len(candidates) + len(gen.out.rejected)  # every question the model wrote, kept or not
    return kept_answerable + kept_refusals + [c for c in others if c.case.type != "out_of_scope"], rejected, generated, attempts


def build_eval_set(
    req: Requirements, corpus: EvalCorpus, config: ForgeConfig, embedder: Embedder, prefix: str = "ev",
) -> EvalSetResult:
    targets = targets_for(req, bool(corpus.tables))
    order = {t: i for i, t in enumerate(BASE_TARGETS)}
    validated: list[Candidate] = []
    rejected: list[RejectedCase] = []
    generated: Counter[str] = Counter()
    rounds = attempts = 0
    for round_index in range(MAX_ROUNDS):
        have = Counter(c.case.type for c in validated)
        deficit = {t: max(0, targets[t] - have[t]) for t in targets}
        if round_index and (not any(deficit.values()) or len(validated) >= ENOUGH):
            break
        wanted = {t: n * OVERGENERATION for t, n in (targets if round_index == 0 else deficit).items()}
        new, rej, made, tried = _round(req, corpus, config, wanted, round_index)
        attempts += tried
        # Consistency with the requirements, then de-duplication against everything kept so far.
        for cand in new:
            problem = violation(cand.case, req)
            if problem:
                rejected.append(RejectedCase(case_type=cand.case.type, question=cand.case.question,
                                             reason="contradicts the requirements", detail=problem))
            else:
                validated.append(cand)
        validated.sort(key=lambda c: order[c.case.type])
        validated, dupes = dedupe(validated, embedder, config.thresholds.dedup_similarity)
        rejected += rej + dupes
        generated.update(made)
        rounds += 1

    # Select toward the composition target, then top up from surplus validated cases.
    chosen: list[Candidate] = []
    for case_type, target in targets.items():
        chosen += select([c for c in validated if c.case.type == case_type], target)
    leftovers = [c for c in validated if c not in chosen and c.case.type in ("answerable", "multidoc", "out_of_scope")]
    chosen += select(leftovers, max(0, TARGET_TOTAL - len(chosen)))

    counters: Counter[str] = Counter()
    cases: list[EvalCase] = []
    for cand in sorted(chosen, key=lambda c: order[c.case.type]):
        counters[cand.case.type] += 1
        cand.case.id = f"{prefix}-{TYPE_CODE[cand.case.type]}{counters[cand.case.type]:02d}"
        cases.append(cand.case)
    assert_cases_consistent(cases, req)  # fail loudly: this would be a generator bug
    split = assign_split(cases)

    stats = {
        "target_total": TARGET_TOTAL, "floor": config.limits.min_eval_cases, "valid_cases": len(cases),
        "floor_met": len(cases) >= config.limits.min_eval_cases, "generation_rounds": rounds,
        "targets": targets, "generated_candidates": dict(generated),
        "validated_after_filters": dict(Counter(c.case.type for c in validated)),
        "selected": dict(counters), "split": {k: len(v) for k, v in split.items()},
        "rejections_by_reason": dict(Counter(r.reason for r in rejected)),
        "candidates_attempted": attempts, "rejection_rate": round(len(rejected) / max(attempts, 1), 3),
        "shortfall": {t: max(0, targets[t] - counters.get(t, 0)) for t in targets if counters.get(t, 0) < targets[t]},
    }
    return EvalSetResult(cases, rejected, stats)


def write_eval_set(result: EvalSetResult, out_dir: Path) -> None:
    """Write ``03_evals/``: eval_set.jsonl, rejected_cases.jsonl, split.json, prompts_used/, stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_set.jsonl").write_text(
        "".join(c.model_dump_json() + "\n" for c in result.cases), encoding="utf-8")
    (out_dir / "rejected_cases.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in result.rejected), encoding="utf-8")
    split = {"dev": [c.id for c in result.cases if c.split == "dev"], "test": [c.id for c in result.cases if c.split == "test"]}
    (out_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (out_dir / "generation_stats.json").write_text(json.dumps(result.stats, indent=2), encoding="utf-8")
    prompts = out_dir / "prompts_used"
    prompts.mkdir(exist_ok=True)
    versions = {}
    for path in sorted((PROMPTS_DIR / "evals").glob("*.md")) + [PROMPTS_DIR / "guard" / "near_miss.md"]:
        shutil.copy(path, prompts / path.name)
        versions[f"{path.parent.name}/{path.stem}"] = int(path.read_text(encoding="utf-8").split("version:")[1].split()[0])
    (prompts / "versions.yaml").write_text(yaml.safe_dump(versions), encoding="utf-8")


def load_eval_set(path: Path) -> list[EvalCase]:
    return [EvalCase.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
