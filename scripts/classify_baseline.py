"""Print the iteration-0 failure classification of a run: per demo, category counts, case ids, section, low_confidence.

Usage: python scripts/classify_baseline.py RUN_DIR [RUN_DIR ...]   (each holds 03_evals/ and 05_baseline/dev/)
Reads DEV results only.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from forgeai.evals.build import load_eval_set
from forgeai.judge import store
from forgeai.optimizer.classify import CATEGORIES, classify_all


def table(run: Path) -> str:
    cases = {c.id: c for c in load_eval_set(run / "03_evals" / "eval_set.jsonl") if c.split == "dev"}
    scores = {s.case_id: s for s in store.read_dev_scores(run)}
    failures = classify_all(cases, scores)
    by_cat: dict[str, list] = defaultdict(list)
    for f in failures:
        by_cat[f.category].append(f)
    lines = [f"## {run.name}: {len(failures)} dev failures of {len(scores)}"]
    for cat in CATEGORIES:
        mine = by_cat.get(cat, [])
        lines.append(f"- {cat}: {len(mine)}")
        for f in mine:
            lines.append(f"    {f.case_id} [{cases[f.case_id].type}] section={f.section or '-'} low_confidence={f.low_confidence} :: {f.reason[:110]}")
    return "\n".join(lines)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(table(Path(arg)))
        print()
