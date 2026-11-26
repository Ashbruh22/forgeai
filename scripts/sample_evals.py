"""Print N randomly sampled eval cases (mixed types) in full, in a readable layout, for hand-checking.

Usage: python scripts/sample_evals.py <run_dir> [n] [seed]
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forgeai.evals.build import load_eval_set  # noqa: E402


def show(case) -> str:
    lines = [f"[{case.id}] type={case.type} split={case.split} expected={case.expected_status} origin={case.origin}"
             + (f" ref={case.topic_ref}" if case.topic_ref else "") + (f" tags={case.tags}" if case.tags else ""),
             f"  Q: {case.question}"]
    if case.expected_sources:
        lines.append(f"  sources: {case.expected_sources}")
    if case.key_facts:
        lines.append("  key facts: " + "; ".join(f"{f.text} ({f.kind})" for f in case.key_facts))
    if case.expected_tools:
        lines.append(f"  expected tools: {case.expected_tools}")
    for span in case.seed_spans:
        lines.append(f"  seed span {span.doc}[{span.char_start}:{span.char_end}]: {span.span_text}")
    for reading in case.interpretations:
        facts = "; ".join(f"{f.text} ({f.kind})" for f in reading.key_facts)
        lines.append(f"  reading: {reading.reading} | discriminators {reading.discriminators} | facts {facts}")
    if "formula" in case.provenance:
        lines.append(f"  computed by code: formula `{case.provenance['formula']}` with {case.provenance['values']}")
    lines.append(f"  validation: {case.validation}")
    return "\n".join(lines)


def main(run_dir: Path, n: int = 8, seed: int = 1) -> None:
    cases = load_eval_set(run_dir / "03_evals" / "eval_set.jsonl")
    rng = random.Random(seed)
    by_type: dict[str, list] = defaultdict(list)
    for case in cases:
        by_type[case.type].append(case)
    for group in by_type.values():
        rng.shuffle(group)
    picked = []
    while len(picked) < min(n, len(cases)):  # mixed types: take turns across types
        for group in list(by_type.values()):
            if group and len(picked) < n:
                picked.append(group.pop())
    print("\n\n".join(show(c) for c in picked))


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 8, int(sys.argv[3]) if len(sys.argv) > 3 else 1)
