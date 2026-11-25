"""Human-readable renderings: requirements.md and gaps.md (pure formatting of data)."""

from __future__ import annotations

from typing import Any

from forgeai.intake.schema import Requirements


def render_requirements(req: Requirements) -> str:
    lines = [f"# Requirements: {req.agent_name}", "", f"**Purpose.** {req.purpose}", "",
             f"**Audience.** {req.audience}", "", f"**Tone.** {req.tone}", "", "## Scope", ""]
    lines += [f"- in: {item}" for item in req.scope.in_] or ["- in: (not restricted)"]
    lines += [f"- out: {item}" for item in req.scope.out] or ["- out: (nothing excluded)"]
    lines += ["", "## Must answer", ""]
    lines += [f"- **{t.id}** {t.topic} (priority {t.priority}, coverage {t.coverage or 'not checked'})"
              + (f", evidence: {', '.join(t.evidence)}" if t.evidence else "") for t in req.must_answer_topics] or ["- (none)"]
    lines += ["", "## Must refuse", ""]
    lines += [f"- **{t.id}** {t.topic}: {t.reason} Redirect: {t.redirect}" for t in req.must_refuse_topics] or ["- (none)"]
    lines += ["", f"**Capabilities needed.** {', '.join(req.capabilities_needed)}", "", "## Gaps", ""]
    lines += [f"- **{g.id}** [{g.kind}] {g.description} Action: `{g.recommended_action}`" for g in req.gaps] or ["- (none)"]
    lines += ["", "## Assumptions", ""]
    lines += [f"- **{a.field}**: {a.value} ({a.source})" for a in req.assumptions] or ["- (none: every field was resolved)"]
    return "\n".join(lines) + "\n"


def render_gaps(req: Requirements, coverage: list[dict[str, Any]], calibration: dict[str, Any]) -> str:
    lines = ["# Groundability gaps", ""]
    if not req.gaps:
        lines += ["No gaps found: every must-answer topic is supported by the documents.", ""]
    for gap in req.gaps:
        lines += [f"## {gap.id}: {gap.kind}", "", gap.description, "",
                  f"- recommended action: `{gap.recommended_action}`"]
        if gap.redirect:
            lines.append(f"- redirect text: {gap.redirect}")
        if gap.evidence:
            lines.append(f"- evidence: {'; '.join(gap.evidence)}")
        lines.append("")
    lines += ["## Topic coverage", "", "| topic | verdict | verified probes | evidence |", "|---|---|---|---|"]
    for entry in coverage:
        verified = sum(1 for p in entry["probes"] if p["supported"])
        lines.append(f"| {entry['topic']} | {entry['verdict']} | {verified}/{len(entry['probes'])} | {', '.join(entry['evidence']) or '-'} |")
    lines += ["", "## Retrieval threshold calibration", "", f"- `min_score` = {calibration['min_score']} "
              f"(bounds {calibration['min_score_bounds']})", f"- {calibration['note']}", ""]
    return "\n".join(lines)
