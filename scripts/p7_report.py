"""Print the P7 report tables of a phase folder: dev vs test per iteration, candidate logs, citation repairs, case trajectories.

Usage: python scripts/p7_report.py RUN_ROOT [CASE_ID ...]     (RUN_ROOT holds one folder per demo)
This is a reporting script run by a person AFTER the optimizer finished; it reads files and imports nothing sealed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.is_file() else []


def passes(rows: list[dict]) -> dict[str, bool]:
    return {r["case_id"]: r["passed"] for r in rows}


def demo_report(demo: Path, watch: list[str]) -> None:
    summary = json.loads((demo / "04_iterations" / "summary.json").read_text(encoding="utf-8"))
    base = demo / "05_baseline"
    states = {0: {"dev": passes(lines(base / "dev" / "case_scores.jsonl")), "test": passes(lines(base / "sealed" / "test_case_scores.jsonl"))}}
    accepted_state = dict(states[0])
    print(f"\n## {demo.name}: stopped: {summary['stop_reason']} (noise floor {summary['noise_floor']})")
    repairs = json.loads((base / "citation_repairs.json").read_text(encoding="utf-8"))
    print(f"citation repairs in the baseline run (dev+test): {repairs['count']} in cases {sorted(repairs['repairs'])}")
    print("\n| iter | patch | status | dev | test (sealed) | fixed | regressed | candidates | dev repairs |\n|---|---|---|---|---|---|---|---|---|")
    d0, t0 = states[0]["dev"], states[0]["test"]
    print(f"| 0 | baseline | - | {sum(d0.values())}/{len(d0)} | {sum(t0.values())}/{len(t0)} | - | - | - | - |")
    for it in summary["iterations"]:
        n = it["iteration"]
        dev = passes(lines(demo / "04_iterations" / f"iter_{n}" / "dev" / "case_scores.jsonl"))
        test = passes(lines(base / "sealed" / f"iter_{n}" / "test_case_scores.jsonl"))
        states[n] = {"dev": dev, "test": test}
        print(f"| {n} | {it['patch']} | {it['status']} | {sum(dev.values())}/{len(dev)} | {sum(test.values())}/{len(test)} | {len(it['fixed'])} | {len(it['regressed'])} | "
              f"{it['candidates_considered']} | {it['citation_repairs_dev']} |")
    for it in summary["iterations"]:
        n = it["iteration"]
        change = json.loads((demo / "04_iterations" / f"iter_{n}" / "change.json").read_text(encoding="utf-8"))
        print(f"\ncycle {n}: category {change['category']} (counts {change['category_counts']}), targets {change['target_cases']}")
        print(f"  gate: {change['gate_reason']}")
        for e in change["candidates"]:
            extra = f" -- {'; '.join(e['reasons'])[:120]}" if e.get("reasons") else ""
            print(f"  - {e['category']}/{e['patch']}: {e['status']}{extra}")
    for cid in watch:
        trail = []
        for n in sorted(states):
            for split in ("dev", "test"):
                if cid in states[n][split]:
                    trail.append(f"it{n}:{split}:{'PASS' if states[n][split][cid] else 'fail'}")
        if trail:
            print(f"trajectory {cid}: " + " ".join(trail))


if __name__ == "__main__":
    root = Path(sys.argv[1])
    watch = sys.argv[2:]
    for demo in sorted(p for p in root.iterdir() if (p / "04_iterations" / "summary.json").is_file()):
        demo_report(demo, watch)
