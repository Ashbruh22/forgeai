"""Requirement coverage matrix (P5.1 correction 1).

Every requirement item must be exercised by at least one valid eval case, or the matrix says
UNCOVERED, loudly. Items come from the requirements and the brief alone:

* every must-answer topic, and every supported aspect of a split topic (snapshot aspects need a
  case that expects the "as of" date);
* every gap that is refused, every must-refuse topic, every tool the requirements do not call for;
* every distinct ask in the brief, which maps to the topics and gaps it is about, and, when it
  needs a tool (estimating a cost needs arithmetic, "when will it arrive" needs the date), a case
  that really exercises that tool. A topic answered only by rate lookups can therefore never hide
  that "cost with discounts" is untested.

Coverage is decided by code from the cases themselves; nothing here asks a model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from forgeai.embeddings import Embedder
from forgeai.evals.models import EvalCase
from forgeai.intake.asks import specific_asks
from forgeai.intake.schema import Gap, Requirements
from forgeai.textmatch import content_terms
from forgeai.tools import REGISTRY

ASK_TARGET_MIN_SIMILARITY = 0.5  # an ask is mapped to the closest requirement item if at least this similar
LEXICAL_TARGET_SHARE = 0.5  # ... and to every item that contains at least this share of the ask's content terms

# Words in an ask that call for a tool. A capability applies only if the requirements list it.
CAPABILITY_SIGNALS: dict[str, tuple[str, ...]] = {
    "arithmetic": ("cost", "total", "estimate", "estimating", "calculate", "calculating", "discount", "sum", "price"),
    "temporal": ("when", "arrive", "arrival", "deadline", "days until", "eta", "how long until", "due date"),
    "structured_lookup": ("find", "finding", "list", "filter", "compare", "comparing", "which", "lookup", "search"),
}
TOOLS_FOR = {"arithmetic": ("calculator",), "temporal": ("current_date",), "structured_lookup": ("csv_lookup",)}


@dataclass
class Item:
    id: str
    kind: str  # topic | aspect | gap | refuse | negtool | ask | ask_capability
    text: str
    expects: str  # answered | refused
    topic_id: str = ""
    aspect: str = ""
    capability: str = ""
    targets: list[str] = field(default_factory=list)  # ask: the items it maps to
    note: str = ""


def gap_subject(gap: Gap) -> str:
    if gap.aspect:
        return gap.aspect
    if gap.kind == "unsupported_topic" and "'" in gap.description:
        return gap.description.split("'")[1]
    return gap.evidence[0] if gap.evidence else gap.description


def unattached_tools(req: Requirements) -> list[str]:
    needed = set(req.capabilities_needed)
    return [name for name, spec in REGISTRY.items() if spec.capability not in needed]


def ask_capabilities(ask: str, req: Requirements) -> list[str]:
    terms = content_terms(ask)
    lowered = ask.lower()
    found = []
    for capability, signals in CAPABILITY_SIGNALS.items():
        if capability in req.capabilities_needed and any(
                (content_terms(s) and content_terms(s) <= terms) or (" " in s and s in lowered) for s in signals):
            found.append(capability)
    return found


def build_items(req: Requirements, brief: str, embedder: Embedder) -> list[Item]:
    items: list[Item] = []
    for topic in req.must_answer_topics:
        items.append(Item(f"topic:{topic.id}", "topic", topic.topic, "answered", topic_id=topic.id))
        if topic.snapshot_as_of:
            for aspect in topic.supported_aspects:
                items.append(Item(f"aspect:{topic.id}/{aspect}", "aspect", f"{aspect} (as of {topic.snapshot_as_of})",
                                  "answered", topic_id=topic.id, aspect=aspect))
    for gap in req.gaps:
        if gap.recommended_action != "warn":
            items.append(Item(f"gap:{gap.id}", "gap", gap_subject(gap), "refused", topic_id=gap.topic_id or ""))
    for refuse in req.must_refuse_topics:
        items.append(Item(f"refuse:{refuse.id}", "refuse", refuse.topic, "refused"))
    for tool in unattached_tools(req):
        items.append(Item(f"negtool:{tool}", "negtool", f"{tool} is not needed, so the assistant must decline it", "refused"))

    targets = [i for i in items if i.kind in ("topic", "aspect", "gap")]
    # A gap that was raised for a sentence of the brief carries that sentence: an ask inside it is about that gap.
    quoted = {f"gap:{g.id}": " ".join([g.description, *[e for e in g.evidence if " " in e]]) for g in req.gaps}
    asks = specific_asks(brief)
    if asks and targets:
        vectors = embedder.encode(asks + [t.text for t in targets])
        ask_vectors, target_vectors = vectors[: len(asks)], vectors[len(asks):]
    for n, ask in enumerate(asks, 1):
        item = Item(f"ask:A{n}", "ask", ask, "answered")
        if targets:
            terms = content_terms(ask)
            sims = target_vectors @ ask_vectors[n - 1]
            best = int(np.argmax(sims))
            mapped = {targets[best].id} if sims[best] >= ASK_TARGET_MIN_SIMILARITY else set()
            mapped |= {t.id for t in targets if terms and len(terms & content_terms(t.text)) / len(terms) >= LEXICAL_TARGET_SHARE}
            mapped |= {gid for gid, text in quoted.items() if terms and len(terms & content_terms(text)) / len(terms) >= 0.8}
            item.targets = sorted(mapped)
            item.note = f"best match {targets[best].id} (cosine {sims[best]:.2f})"
        items.append(item)
        for capability in ask_capabilities(ask, req):
            items.append(Item(f"ask:A{n}/{capability}", "ask_capability", f"{ask} [needs {capability}]", "answered",
                              capability=capability))
    return items


def case_covers(item: Item, case: EvalCase, evidence: dict[str, list[tuple[str, int, int]]]) -> bool:
    """Does this valid case exercise the requirement item?"""
    if item.kind == "topic":
        if case.expected_status != "answered":
            return False
        if item.id in case.covers or case.topic_ref == item.topic_id and case.origin != "snapshot":
            return True
        return any(s.doc == d and s.char_start < e and b < s.char_end
                   for s in case.seed_spans for d, b, e in evidence.get(item.topic_id, []))
    if item.kind == "aspect":
        return (case.expected_status == "answered" and case.aspect == item.aspect and case.topic_ref == item.topic_id
                and any(f.kind == "snapshot_date" for f in case.key_facts))
    if item.kind in ("gap", "refuse"):
        return case.expected_status == "refused" and case.topic_ref == item.id.split(":", 1)[1]
    if item.kind == "negtool":
        return case.expected_status == "refused" and case.origin == "negative_tool" and case.topic_ref == item.id.split(":", 1)[1]
    if item.kind == "ask_capability":
        return (item.id in case.covers and case.expected_status == "answered"
                and bool(set(case.expected_tools) & set(TOOLS_FOR[item.capability])))
    return False


def build_matrix(
    items: list[Item], cases: list[EvalCase], evidence: dict[str, list[tuple[str, int, int]]],
    embedder: Embedder | None = None, rejection_notes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The matrix: each item with the ids of the cases covering it, and UNCOVERED where there are none."""
    by_id: dict[str, list[str]] = {}
    for item in items:
        if item.kind != "ask":
            by_id[item.id] = [c.id for c in cases if case_covers(item, c, evidence)]
    rows: list[dict[str, Any]] = []
    for item in items:
        if item.kind == "ask":
            parts = {t: by_id.get(t, []) for t in item.targets}
            ids = sorted({cid for v in parts.values() for cid in v})
            covered = bool(item.targets) and all(parts.values())
            if not item.targets:
                note = "no topic or gap in the requirements matches this ask: the intake did not carry it over"
            else:
                missing = [t for t, v in parts.items() if not v]
                note = f"maps to {item.targets}" + (f"; no case for {missing}" if missing else "")
        else:
            ids = by_id[item.id]
            covered = bool(ids)
            note = item.note
        row = {"id": item.id, "kind": item.kind, "requirement": item.text, "expects": item.expects,
               "status": "COVERED" if covered else "UNCOVERED", "cases": ids, "note": note}
        if not covered and rejection_notes and item.id in rejection_notes:
            row["explanation"] = rejection_notes[item.id]
        rows.append(row)
    uncovered = [r["id"] for r in rows if r["status"] == "UNCOVERED"]
    return {"summary": {"items": len(rows), "covered": len(rows) - len(uncovered), "uncovered": uncovered,
                        "cases": len(cases)},
            "items": rows}


def uncovered_items(items: list[Item], cases: list[EvalCase], evidence: dict[str, list[tuple[str, int, int]]]) -> list[Item]:
    matrix = build_matrix(items, cases, evidence)
    missing = set(matrix["summary"]["uncovered"])
    return [i for i in items if i.id in missing]


def write_matrix(matrix: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coverage_matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")


class CoverageError(Exception):
    """Requirement items have no eval case. Raised by the CLI after the artifacts are written."""
