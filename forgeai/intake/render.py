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
    for t in req.must_answer_topics:
        line = f"- **{t.id}** {t.topic} (priority {t.priority}, coverage {t.coverage or 'not checked'})"
        if t.evidence:
            line += f", evidence: {', '.join(t.evidence)}"
        lines.append(line)
        if t.supported_aspects:
            as_of = f", as of {t.snapshot_as_of}" if t.snapshot_as_of else ""
            lines.append(f"  - supported part: {'; '.join(t.supported_aspects)}{as_of}")
        elif t.snapshot_as_of:
            lines.append(f"  - answers come from a snapshot dated {t.snapshot_as_of}")
    if not req.must_answer_topics:
        lines.append("- (none)")
    lines += ["", "## Must refuse", ""]
    lines += [f"- **{t.id}** {t.topic}: {t.reason} Redirect: {t.redirect}" for t in req.must_refuse_topics] or ["- (none)"]
    lines += ["", f"**Capabilities needed.** {', '.join(req.capabilities_needed)}", "", "## Gaps", ""]
    lines += [f"- **{g.id}** [{g.kind}] {g.description} Action: `{g.recommended_action}`" for g in req.gaps] or ["- (none)"]
    lines += ["", "## Assumptions", ""]
    lines += [f"- **{a.field}**: {a.value} ({a.source})" for a in req.assumptions] or ["- (none: every field was resolved)"]
    lines += ["", "## Assumptions rejected (brief > corpus > assumptions)", ""]
    lines += [f"- **{r.field}**: '{r.item}' conflicts with the {r.conflict_with}: {r.detail}"
              + (f" Evidence: {' | '.join(r.evidence[:2])}" if r.evidence else "") for r in req.assumptions_rejected] or ["- (none)"]
    return "\n".join(lines) + "\n"


def render_gaps(req: Requirements, coverage: list[dict[str, Any]], calibration: dict[str, Any],
                agreement: dict[str, Any]) -> str:
    lines = ["# Groundability gaps", ""]
    if not req.gaps:
        lines += ["No gaps found: every must-answer topic is supported by the documents.", ""]
    for gap in req.gaps:
        lines += [f"## {gap.id}: {gap.kind}", "", gap.description, "",
                  f"- recommended action: `{gap.recommended_action}`"]
        if gap.aspect:
            lines.append(f"- unsupported part: {gap.aspect}")
        if gap.redirect:
            lines.append(f"- redirect text: {gap.redirect}")
        if gap.evidence:
            lines.append(f"- evidence: {'; '.join(gap.evidence)}")
        lines.append("")
    lines += ["## Topic coverage", "", "| topic | verdict | verified probes | supported part | evidence |", "|---|---|---|---|---|"]
    for entry in coverage:
        verified = sum(1 for p in entry["probes"] if p["supported"])
        parts = "; ".join(a["aspect"] for a in entry["aspects"] if a["verdict"] in ("SUPPORTED", "PARTIAL")) or "-"
        lines.append(f"| {entry['topic']} | {entry['effective_verdict']} | {verified}/{len(entry['probes'])} | {parts} | {', '.join(entry['evidence']) or '-'} |")
    lines += ["", "## Agreement between the two support checks", "",
              f"- probes with a verbatim span: {agreement['probes_with_verbatim_span']} of {agreement['probes_total']}",
              f"- both pass: {agreement['both_pass']}; code check only: {agreement['code_check_only']}; "
              f"judge only: {agreement['judge_only']}; neither: {agreement['neither']}",
              f"- disagreement rate: {agreement['disagreement_rate']}", "",
              "## Retrieval threshold calibration", "", f"- `min_score` = {calibration['min_score']} "
              f"(bounds {calibration['min_score_bounds']})", f"- {calibration['note']}", ""]
    return "\n".join(lines)
