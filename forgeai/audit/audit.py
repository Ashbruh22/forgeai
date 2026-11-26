"""Eval-set audit against the hidden planted-facts file (spec §6.1, "measure the measurer").

This is the ONLY module allowed to read that file. It runs after the eval set exists and never
feeds anything back into generation. It reports two numbers:

* coverage: how many planted facts are reached by the eval set, i.e. every string of the fact
  appears in the seed spans of that document taken together (and how many are partly reached);
* key-fact agreement: for each key fact, look at the seed spans that contain it; if those spans
  hold planted facts, the key fact must agree with them (its numbers appear in the planted text,
  or it contains a planted string). A key fact that contradicts a planted number is a CONFLICT; a true
  detail with no number that was not planted is counted separately, and facts in spans with no planted
  fact cannot be checked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from forgeai.evals.matching import fact_in_text, numbers_in
from forgeai.evals.models import EvalCase
from forgeai.textmatch import normalize


def present(needle: str, text: str) -> bool:
    """``needle`` occurs in ``text`` as a whole token (so "5%" is not found inside "15%")."""
    pattern = r"(?<![\w.,])" + re.escape(normalize(needle)) + r"(?![\w])"
    return re.search(pattern, normalize(text), re.IGNORECASE) is not None


def fraction_present(strings: list[str], text: str) -> float:
    return sum(present(s, text) for s in strings) / len(strings) if strings else 0.0


def audit_eval_set(cases: list[EvalCase], planted_facts: Path) -> dict[str, Any]:
    sheet = json.loads(planted_facts.read_text(encoding="utf-8"))
    planted = [f for f in sheet["facts"] if f.get("must_contain") and f.get("doc")]
    spans = [s for case in cases for s in case.seed_spans]

    covered, partial, uncovered = [], [], []
    for fact in planted:
        doc_text = " ".join(s.span_text for s in spans if s.doc in fact["doc"])
        share = fraction_present(fact["must_contain"], doc_text)
        (covered if share == 1 else partial if share > 0 else uncovered).append(fact["id"])

    agree = detail = neutral = 0
    conflicts: list[dict[str, str]] = []
    for case in cases:
        facts = [*case.key_facts, *[kf for i in case.interpretations for kf in i.key_facts]]
        for key_fact in facts:
            if key_fact.kind == "snapshot_date":
                continue
            holding = [s for s in case.seed_spans if fact_in_text(key_fact.kind, key_fact.text, s.span_text, strict=False)]
            owners = [p for p in planted for s in holding
                      if s.doc in p["doc"] and fraction_present(p["must_contain"], s.span_text) >= 0.5]
            if not owners:
                neutral += 1  # the key fact sits in text where no planted fact lives: nothing to compare with
                continue
            reference = " ".join(" ".join([p["statement"], *p["must_contain"]]) for p in owners)
            key_numbers = numbers_in(key_fact.text)
            if key_numbers and not all(n in set(numbers_in(reference)) for n in key_numbers):
                conflicts.append({"case": case.id, "key_fact": key_fact.text[:120],
                                  "planted": ", ".join(p["id"] for p in owners)})
            elif key_numbers or any(present(ps, key_fact.text) for p in owners for ps in p["must_contain"]):
                agree += 1
            else:
                detail += 1  # a true detail from the same span that was simply not planted
    checked = agree + len(conflicts)
    total = len(planted)
    return {
        "planted_facts_file": sheet["demo"],
        "coverage": {"planted_facts": total, "fully_covered": len(covered), "partly_covered": len(partial),
                     "not_reached": len(uncovered),
                     "fraction_fully_covered": round(len(covered) / total, 3) if total else None,
                     "fraction_reached": round((len(covered) + len(partial)) / total, 3) if total else None,
                     "not_reached_ids": uncovered},
        "key_fact_agreement": {"checked": checked, "agree": agree, "conflicts": len(conflicts),
                               "fraction": round(agree / checked, 3) if checked else None,
                               "unplanted_detail_in_a_planted_span": detail,
                               "not_checkable_no_planted_fact_in_span": neutral,
                               "disagreements": conflicts},
    }
