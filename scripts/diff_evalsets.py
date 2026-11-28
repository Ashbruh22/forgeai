"""Compare two eval sets by QUESTION (ids are renumbered when a case is dropped, so ids alone mislead).

Usage: python scripts/diff_evalsets.py <old eval_set.jsonl> <new eval_set.jsonl>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(old_path: Path, new_path: Path) -> None:
    old, new = load(old_path), load(new_path)
    old_by_q, new_by_q = {r["question"]: r for r in old}, {r["question"]: r for r in new}
    kept = [q for q in new_by_q if q in old_by_q]
    retyped = [q for q in kept if old_by_q[q].get("tool_mode", "required") != new_by_q[q].get("tool_mode", "required")]
    renumbered = [q for q in kept if old_by_q[q]["id"] != new_by_q[q]["id"]]
    resplit = [q for q in kept if old_by_q[q]["split"] != new_by_q[q]["split"]]
    removed = [q for q in old_by_q if q not in new_by_q]
    added = [q for q in new_by_q if q not in old_by_q]
    print(f"old {len(old)} cases, new {len(new)} cases")
    print(f"  same question kept: {len(kept)}  (renumbered id: {len(renumbered)}, moved dev/test: {len(resplit)}, tool mode now optional: {len(retyped)})")
    print(f"  removed: {len(removed)}   added: {len(added)}")
    for q in removed:
        print(f"    - {old_by_q[q]['id']} [{old_by_q[q]['type']}] {q[:120]}")
    for q in added:
        print(f"    + {new_by_q[q]['id']} [{new_by_q[q]['type']}] {q[:120]}")
    for q in retyped:
        print(f"    ~ {new_by_q[q]['id']} tool mode required -> optional: {q[:100]}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
