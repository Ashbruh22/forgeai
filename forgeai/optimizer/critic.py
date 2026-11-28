"""The critic: an LLM that writes only the TEXT of one prompt section once code has decided the category and section.

The critic sees the current section, the goal of the change and masked descriptions of the kinds of mistakes (never a
key-fact value). Its text is checked by ``guard.py``; a text that fails is sent back once with the reasons, then dropped.
"""

from __future__ import annotations

from pydantic import BaseModel

from forgeai.config import ForgeConfig
from forgeai.evals.models import EvalCase
from forgeai.llm import call_structured
from forgeai.optimizer import guard
from forgeai.optimizer.classify import Failure
from forgeai.optimizer.patches import PatchSet
from forgeai.runner.agent_def import AgentDef


class SectionDraft(BaseModel):
    rationale: str = ""
    text: str = ""


def mistakes(failures: list[Failure], cases: dict[str, EvalCase]) -> str:
    """One line per failing case: the kind of case and what kind of mistake. No question text, no values."""
    lines = []
    for f in failures[:8]:
        case = cases[f.case_id]
        lines.append(f"- a {case.type.replace('_', ' ')} question: {f.reason[:160]}")
    return "\n".join(lines) or "- (none)"


def draft_text(patch: PatchSet, agent: AgentDef, failures: list[Failure], dev_cases: dict[str, EvalCase], config: ForgeConfig,
               allowed_vocabulary: str) -> tuple[str | None, list[dict]]:
    """The accepted new text of ``patch.section`` (or None) and a log of every draft that was rejected and why."""
    current = getattr(agent.prompt_sections, patch.section)
    model = config.agent_model
    rejected: list[dict] = []
    feedback = ""
    cases = list(dev_cases.values())
    for attempt in range(2):  # one retry, with the reasons fed back
        draft = call_structured(
            "optimizer.critic", "optimizer/draft_section",
            {"section": patch.section, "instruction": patch.instruction, "current_text": current or "(empty)",
             "mistakes": mistakes(failures, dev_cases), "feedback": feedback},
            SectionDraft, model.name, 0.3, seed=model.seed + attempt, fallback=SectionDraft())
        text = draft.text.strip()
        problems = guard.size_ok(current, text) if current else ([] if len(text) >= 20 else ["empty draft"])
        if text:
            problems += guard.violations(text, cases, original=current, allowed_vocabulary=allowed_vocabulary)
            if patch.section == "refusal_policy":
                problems += guard.refusal_items_kept(current, text)
            if text.strip() == current.strip():
                problems.append("the text is unchanged")
        if not problems:
            return text, rejected
        rejected.append({"patch": patch.id, "attempt": attempt + 1, "text": text, "reasons": problems})
        feedback = "Your previous text was rejected: " + "; ".join(problems) + ". Write a general rule without those.\n"
    return None, rejected
