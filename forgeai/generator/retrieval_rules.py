"""Initial retrieval parameters, set by RULES from the document profile (spec §5.2).

Rules are more reliable than a small model on numeric choices, and the optimizer tunes the
values later. Every choice is returned with its rationale so the report can show it.
"""

from __future__ import annotations

from typing import Any

from forgeai.ingest.models import DocRecord
from forgeai.runner.agent_def import Chunking, Hybrid, Retrieval

SHORT_SECTION_WORDS = 90   # short policy clauses: small chunks keep one rule per chunk
LONG_SECTION_WORDS = 220   # long technical prose: larger chunks keep an argument together
DEFAULT_BOUNDS = (0.05, 0.50)


def prose_section_words(records: list[DocRecord]) -> float | None:
    """Average words per section over prose documents, weighted by document length."""
    prose = [r.structure for r in records if r.status == "ok" and r.structure and r.structure.kind == "prose"]
    total = sum(s.total_words for s in prose)
    return sum(s.avg_section_words * s.total_words for s in prose) / total if total else None


def retrieval_from_profile(
    records: list[DocRecord], calibration: dict[str, Any] | None = None
) -> tuple[Retrieval, dict[str, str]]:
    """Return the retrieval config and a ``{parameter: why}`` rationale."""
    why: dict[str, str] = {}
    section_words = prose_section_words(records)
    chunking = Chunking()
    if section_words is None:
        why["chunking"] = "no prose documents; chunk size unused (tables use one chunk per row)"
    elif section_words < SHORT_SECTION_WORDS:
        chunking = Chunking(size_tokens=200, overlap_tokens=40)
        why["chunking"] = f"short sections (avg {section_words:.0f} words): 200-token chunks"
    elif section_words > LONG_SECTION_WORDS:
        chunking = Chunking(size_tokens=500, overlap_tokens=80)
        why["chunking"] = f"long sections (avg {section_words:.0f} words): 500-token chunks"
    else:
        why["chunking"] = f"medium sections (avg {section_words:.0f} words): default 350-token chunks"

    has_table = any(r.status == "ok" and r.structure and r.structure.kind == "tabular" for r in records)
    if has_table:
        hybrid, top_k = Hybrid(dense_weight=0.5, bm25_weight=0.5), 6
        why["hybrid"] = "tabular data: codes and names need exact matching, so BM25 weighs as much as dense"
        why["top_k"] = "tabular data: more rows are needed to compare items, so top_k 6"
    else:
        hybrid, top_k = Hybrid(), 5
        why["hybrid"] = "prose only: default 0.6 dense / 0.4 BM25"
        why["top_k"] = "prose only: default top_k 5"

    min_score, bounds = 0.25, DEFAULT_BOUNDS
    if calibration and calibration.get("calibrated"):
        min_score = float(calibration["min_score"])
        bounds = (float(calibration["min_score_bounds"][0]), float(calibration["min_score_bounds"][1]))
        why["min_score"] = f"calibrated from guard probes: {calibration['note']}"
    else:
        why["min_score"] = "no calibration available; default 0.25 with bounds 0.05 to 0.50"
    retrieval = Retrieval(
        chunking=chunking, csv_mode="row_as_record", hybrid=hybrid, top_k=top_k,
        min_score=round(min_score, 4), min_score_bounds=(round(bounds[0], 4), round(bounds[1], 4)),
        query_rewrite=False,
    )
    why["query_rewrite"] = "off by default; the optimizer may enable it for retrieval misses"
    return retrieval, why
