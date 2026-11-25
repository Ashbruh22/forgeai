"""Prompt sections (spec §5.1): a fixed skeleton, filled in from the requirements.

The model writes only the identity and audience/tone sections, and even those are checked
in code (they must mention the assistant name and the audience, else the template is used).
Everything that carries a *rule* is rendered deterministically from requirement data and
versioned templates, so that, for example, the refusal policy provably lists every
must-refuse topic and every guard gap with its redirect.
"""

from __future__ import annotations

from pydantic import BaseModel

from forgeai.config import ForgeConfig
from forgeai.generator.tool_selector import ToolSelection
from forgeai.intake.interviewer import DEFAULTS
from forgeai.intake.schema import Gap, Requirements
from forgeai.llm import call_structured
from forgeai.prompts import load_prompt
from forgeai.runner.agent_def import PromptSections
from forgeai.textmatch import content_terms
from forgeai.tools import REGISTRY, tool_description

NONE = "(none)"


class OpeningSections(BaseModel):
    identity: str = ""
    audience_tone: str = ""


def _text(prompt_id: str, **variables: str) -> str:
    return load_prompt(prompt_id).render(variables)[1].strip()


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else NONE


def opening_sections(req: Requirements, config: ForgeConfig) -> tuple[str, str]:
    """Identity and audience/tone: model-written, verified, with a deterministic fallback."""
    agent = config.agent_model
    fallback = OpeningSections(
        identity=_text("generator/identity_fallback", agent_name=req.agent_name, purpose=req.purpose),
        audience_tone=_text("generator/audience_tone_fallback", audience=req.audience, tone=req.tone),
    )
    result = call_structured(
        "generator.identity", "generator/identity",
        {"agent_name": req.agent_name, "purpose": req.purpose, "audience": req.audience, "tone": req.tone},
        OpeningSections, agent.name, agent.temperature, seed=agent.seed, fallback=fallback,
    )
    identity = result.identity.strip()
    if req.agent_name.lower() not in identity.lower():
        identity = fallback.identity  # the model dropped the name: use the template
    audience_tone = result.audience_tone.strip()
    if not content_terms(req.audience) & content_terms(audience_tone):
        audience_tone = fallback.audience_tone
    return identity, audience_tone


def refusal_items(req: Requirements) -> list[str]:
    """One line per must-refuse topic and per refusal-worthy guard gap, each with its redirect."""
    items = [
        _text("generator/refusal_item", subject=t.topic, reason=t.reason.rstrip("."), redirect=t.redirect)
        for t in req.must_refuse_topics
    ]
    topics = {t.id: t for t in req.must_answer_topics}
    for gap in req.gaps:
        if gap.recommended_action == "warn":
            continue
        redirect = gap.redirect or DEFAULTS["gap_redirect"]
        tied = topics.get(gap.topic_id or "")
        if gap.kind == "live_data" and tied is not None and tied.supported_aspects:
            # The recorded part of this topic is answerable from a snapshot; only the live value is refused.
            items.append(_text(
                "generator/refusal_item_live", subject=gap_subject(gap), reason=_gap_reason(gap),
                recorded="; ".join(tied.supported_aspects), as_of=tied.snapshot_as_of or "the snapshot date",
                redirect=redirect))
        else:
            items.append(_text("generator/refusal_item", subject=gap_subject(gap), reason=_gap_reason(gap),
                               redirect=redirect))
    return items


def gap_subject(gap: Gap) -> str:
    """The thing being refused, in words the model can match against a user question."""
    if gap.aspect:
        return gap.aspect
    if gap.kind == "unsupported_topic" and "'" in gap.description:
        return gap.description.split("'")[1]
    return gap.evidence[0] if gap.evidence else gap.description


def _gap_reason(gap: Gap) -> str:
    return {
        "unsupported_topic": "the documents do not cover this",
        "live_data": "this needs a live value and the documents are a static snapshot",
        "capability": "you cannot perform this action",
        "freshness": "the documents may be out of date",
        "partial_coverage": "the documents only partly cover this",
    }[gap.kind]


def build_sections(req: Requirements, selection: ToolSelection, config: ForgeConfig) -> PromptSections:
    identity, audience_tone = opening_sections(req, config)
    scope = _text(
        "generator/scope", in_scope=_bullets(req.scope.in_),
        topics=_bullets([t.topic for t in req.must_answer_topics]), out_scope=_bullets(req.scope.out),
    )
    refusal = _text("generator/refusal_policy", items=_bullets(refusal_items(req)))

    grounding = _text("generator/grounding_rules")
    notes = [
        _text("generator/snapshot_note", topic=t.topic, as_of=t.snapshot_as_of)
        for t in req.must_answer_topics if t.snapshot_as_of
    ]
    if notes:
        grounding += "\n\n" + "\n".join(notes)

    if selection.attached:
        tool_policy = _text("generator/tool_policy_attached", items=_bullets(
            [f"{d.tool}: needed for {d.requirement_id} ({d.justification.rstrip('.')})" for d in selection.attached]))
    else:
        tool_policy = _text("generator/tool_policy_none")
    missing = [tool_description(name) for name in selection.unattached_tools if name in REGISTRY]
    tool_policy += "\n\n" + _text("generator/tool_policy_missing", items=_bullets(missing))

    return PromptSections(
        identity=identity, audience_tone=audience_tone, scope=scope, refusal_policy=refusal,
        grounding_rules=grounding, citation_rules=_text("generator/citation_rules"),
        ambiguity_policy=_text("generator/ambiguity_policy"), tool_policy=tool_policy,
        output_contract=_text("generator/output_contract"), few_shot=[],
    )
