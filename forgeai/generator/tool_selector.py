"""Tool selection (spec §5.3): attach only what the requirements need, and prove why.

Procedure:
1. Deterministic preconditions (e.g. a table lookup needs a table in the corpus).
2. The agent model PROPOSES tools; each proposal must cite a must-answer topic id.
3. A validator drops anything unknown, uncited, duplicated or failing a precondition.
4. NECESSITY: a model from a different family (the judge) is asked whether the cited topic
   cannot be answered correctly without the tool, and must quote the words of the topic that
   show the need; the quote is verified in code. No verified quote, no tool.

A word merely appearing in the brief (a port, a date, "weather") never attaches a tool by
itself; the need has to be in a must-answer topic and survive a second model's review.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from forgeai.config import ForgeConfig
from forgeai.llm import call_structured
from forgeai.textmatch import is_verbatim
from forgeai.tools import REGISTRY, tool_description, tool_need_criteria


class ToolProposal(BaseModel):
    tool: str
    needed: bool = True  # one entry per available tool, so every tool is considered deliberately
    requirement_id: str = ""
    justification: str = ""


class ToolProposals(BaseModel):
    proposals: list[ToolProposal] = Field(default_factory=list)


class ToolNeed(BaseModel):
    # Field order matters for constrained decoding: the model must settle this first.
    stated_in_documents: bool = True  # the answer is one stated fact (a clause, period, deadline, cell)
    needed: bool = False
    quote: str = ""


@dataclass
class ToolDecision:
    tool: str
    attached: bool
    reason: str
    requirement_id: str = ""
    justification: str = ""
    necessity_quote: str = ""


@dataclass
class ToolSelection:
    decisions: list[ToolDecision] = field(default_factory=list)

    @property
    def attached(self) -> list[ToolDecision]:
        return [d for d in self.decisions if d.attached]

    @property
    def rejected(self) -> list[ToolDecision]:
        return [d for d in self.decisions if not d.attached]

    @property
    def unattached_tools(self) -> list[str]:
        """Registry tools that are NOT attached; each gets a negative eval case (spec §5.3.4)."""
        attached = {d.tool for d in self.attached}
        return [name for name in REGISTRY if name not in attached]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attached": [asdict(d) for d in self.attached],
            "rejected": [asdict(d) for d in self.rejected],
            "unattached_tools": self.unattached_tools,
        }


# World-state tools fetch something that changes (today's date, the weather). Both models were
# fooled by the same words in the keyword-trap briefs ("deadlines", "weather-related delays"),
# so these tools also need the TOPIC ITSELF to ask for a time-dependent value. This is a VETO
# only: a matching cue never attaches a tool, it merely fails to block one the two models agree on.
WORLD_STATE_CUES: dict[str, re.Pattern[str]] = {
    "current_date": re.compile(
        r"\b(today|current date|todays? date|right now|days? (?:until|since|remaining|left|to go)"
        r"|how long until|time (?:until|since|left)|from (?:the )?(?:lead time|order date)"
        r"|arriv\w*|delivery (?:date|time|estimate)|eta)\b", re.IGNORECASE),
    "weather": re.compile(
        r"\b(forecast|current weather|weather (?:at|in|for|on|conditions)|weather conditions|expected weather)\b",
        re.IGNORECASE),
}


def preconditions(tool: str, has_table: bool, topic: str = "") -> str | None:
    """Reason a tool cannot be used for ``topic``, or None. Rules only, no LLM."""
    if tool == "csv_lookup" and not has_table:
        return "the corpus has no table (CSV) to look up"
    cue = WORLD_STATE_CUES.get(tool)
    if cue is not None and not cue.search(topic):
        return "veto: the topic does not itself ask for a time-dependent value (a date, a forecast)"
    return None


def _tools_text() -> str:
    return "\n".join(f"- [{spec.capability}] {tool_description(name)}" for name, spec in REGISTRY.items())


def select_tools(
    brief: str, topics: dict[str, str], has_table: bool, config: ForgeConfig,
) -> ToolSelection:
    """Select tools for ``topics`` (``{requirement_id: topic text}``)."""
    selection = ToolSelection()
    agent = config.agent_model
    proposals = call_structured(
        "generator.tool_proposal", "generator/tool_proposal",
        {"brief": brief, "topics": "\n".join(f"{rid}: {text}" for rid, text in topics.items()) or "(none)",
         "has_table": "yes" if has_table else "no", "tools": _tools_text()},
        ToolProposals, agent.name, 0.0, seed=agent.seed, fallback=ToolProposals(),
    ).proposals

    pending: list[ToolProposal] = []
    seen: set[str] = set()
    for proposal in proposals:
        name = proposal.tool.strip()
        if name in REGISTRY and not proposal.needed:
            selection.decisions.append(ToolDecision(name, False, "the proposer found no topic that needs it"))
        elif name not in REGISTRY:
            selection.decisions.append(ToolDecision(name, False, "not a registered tool", proposal.requirement_id))
        elif proposal.requirement_id not in topics:
            selection.decisions.append(ToolDecision(
                name, False, f"cites {proposal.requirement_id or 'no requirement'}, which is not a must-answer topic",
                proposal.requirement_id, proposal.justification))
        elif name in seen:
            continue
        elif (blocked := preconditions(name, has_table, topics[proposal.requirement_id])) is not None:
            selection.decisions.append(ToolDecision(name, False, blocked, proposal.requirement_id, proposal.justification))
        else:
            seen.add(name)
            pending.append(proposal)

    judge = config.judge_model  # necessity review: every proposal, in one batch on the judge model
    for proposal in pending:
        topic = topics[proposal.requirement_id]
        need = call_structured(
            "generator.tool_need", "generator/tool_need",
            {"topic": topic, "tool": tool_description(proposal.tool), "justification": proposal.justification,
             "criteria": tool_need_criteria(proposal.tool)},
            ToolNeed, judge.name, judge.temperature, seed=judge.seed, fallback=ToolNeed(),
        )
        if need.stated_in_documents:
            reason = "necessity check: the answer is simply stated in a document, so reading it is enough"
        elif not need.needed:
            reason = "necessity check: the reviewer says the topic can be answered without it"
        elif not is_verbatim(need.quote, topic):
            reason = "necessity check: the reviewer's supporting quote is not in the topic text"
        else:
            selection.decisions.append(ToolDecision(
                proposal.tool, True, "needed and verified", proposal.requirement_id,
                proposal.justification, need.quote))
            continue
        selection.decisions.append(ToolDecision(
            proposal.tool, False, reason, proposal.requirement_id, proposal.justification, need.quote))
    return selection
