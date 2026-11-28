"""Print scored cases in full: question, agent answer, citations, every layer-1 check and every layer-2 verdict.

Usage: python scripts/show_cases.py <phase run folder> <eval run folder> [--fail N] [--pass N] [--id CASE_ID ...]

Cases are taken from the DEV split. ``--id`` also reads the sealed test scores, for a case a person explicitly asked
to see; this is a report, not the optimizer's code path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forgeai.evals.build import load_eval_set  # noqa: E402
from forgeai.judge import sealed, store  # noqa: E402
from forgeai.judge.models import CaseRun, CaseScore  # noqa: E402


def mark(value: bool | None) -> str:
    return "n/a" if value is None else ("ok" if value else "FAIL")


def show(score: CaseScore, run: CaseRun, case) -> None:  # noqa: ANN001
    l1, l2 = score.layer1, score.layer2
    print("=" * 100)
    print(f"{score.case_id}  type={score.type}  split={score.split}  RESULT: {'PASS' if score.passed else 'FAIL'}")
    print(f"QUESTION: {score.question}")
    print(f"EXPECTED: status={case.expected_status}  sources={case.expected_sources}  tools={case.expected_tools}")
    print(f"  key facts: {[f'{k.text} ({k.kind})' for k in case.key_facts] or '-'}")
    for reading in case.interpretations:
        print(f"  reading: {reading.reading}  facts={[k.text for k in reading.key_facts]}")
    print(f"AGENT ANSWER: status={run.answer.status}  latency={run.latency_ms / 1000:.1f}s")
    print(f"  {run.answer.answer or run.answer.clarifying_question or '(empty)'}")
    for c in run.answer.citations:
        print(f"  CITATION {c.doc} [{c.chunk_id}]: \"{c.quote[:150]}\"")
    for t in run.answer.tool_calls:
        print(f"  TOOL {t.tool}({json.dumps(t.args)}) -> {t.result[:120]}")
    print(f"  retrieved ({len(run.retrieved)}): " + ", ".join(f"{c.chunk_id}" for c in run.retrieved))
    print("LAYER 1 (deterministic)")
    print(f"  contract valid: {mark(l1.contract_valid)} {l1.contract_issues or ''}")
    print(f"  status: {mark(l1.status_ok)} (got {l1.status})")
    print(f"  citations valid: {l1.citations_valid}/{l1.citations_total} {l1.citation_issues or ''}")
    print(f"  source correct (seed-span overlap): {mark(l1.source_ok)} {('missing ' + str(l1.sources_missing)) if l1.sources_missing else ''}")
    for f in l1.facts:
        print(f"  key fact {f.fact!r} [{f.kind}]: {mark(f.matched)} via {f.method} {f.note} {('quote: ' + f.quote) if f.quote else ''}")
    print(f"  tools called: {l1.tools_called}  tool check: {mark(l1.tool_ok)} {l1.tool_notes or ''}")
    print(f"  retrieval hit@k: {mark(l1.retrieval_hit)} first-hit rank={l1.retrieval_rank} docs hit={l1.retrieval_docs_hit}")
    if l1.interpretations_total is not None:
        print(f"  readings covered: {l1.interpretations_covered}/{l1.interpretations_total}")
    print(f"LAYER 2 (judge) ran={l2.ran}  low_confidence={score.low_confidence}")
    for c in l2.claims:
        print(f"  claim [{'supported' if c.supported else 'UNSUPPORTED'}] {c.claim}")
        print(f"      quote: {c.quote[:140]!r}  reason: {c.reason[:200]}")
    for u in l2.upgrades:
        print(f"  fact equivalence: {u}")
    if l2.refusal_claims:
        print(f"  refusal claims: {l2.refusal_claims}")
    print(f"WHY: {score.reasons or 'every check passed'}")
    if score.disagreements:
        print(f"LAYER 1 vs LAYER 2: {json.dumps(score.disagreements)[:600]}")


def main() -> None:
    args = sys.argv[1:]
    out, run_dir = Path(args[0]), Path(args[1])
    fails = int(args[args.index("--fail") + 1]) if "--fail" in args else 5
    passes = int(args[args.index("--pass") + 1]) if "--pass" in args else 3
    ids = [args[i + 1] for i, a in enumerate(args) if a == "--id"]
    cases = {c.id: c for c in load_eval_set(run_dir / "03_evals" / "eval_set.jsonl")}
    dev_scores = store.read_dev_scores(out)
    dev_runs = {json.loads(line)["case_id"]: CaseRun.model_validate_json(line)
                for line in (store.dev_dir(out) / "answers.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
    failing = [s for s in dev_scores if not s.passed]
    chosen, seen_types = [], set()
    for s in failing:  # a spread of case types first, then the rest
        if s.type not in seen_types and len(chosen) < fails:
            chosen.append(s)
            seen_types.add(s.type)
    chosen += [s for s in failing if s not in chosen][: max(fails - len(chosen), 0)]
    chosen += [s for s in dev_scores if s.passed][:passes]
    for case_id in ids:
        if case_id not in {s.case_id for s in chosen}:
            pool = {s.case_id: s for s in [*dev_scores, *sealed.read_test_scores(out)]}
            chosen.append(pool[case_id])
    test_runs = {json.loads(line)["case_id"]: CaseRun.model_validate_json(line)
                 for line in (sealed.sealed_dir(out) / "test_answers.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()} \
        if ids else {}
    for score in chosen:
        show(score, {**dev_runs, **test_runs}[score.case_id], cases[score.case_id])


if __name__ == "__main__":
    main()
