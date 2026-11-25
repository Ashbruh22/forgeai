"""The agent definition (``agent.yaml``): structured data, not one big prompt (spec §5.1).

The system prompt is *compiled* from addressable sections so the optimizer can patch one
section (e.g. ``refusal_policy``) without touching the others.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from forgeai.prompts import load_prompt
from forgeai.runner.contract import load_contract_schema
from forgeai.tools import tool_description

SECTION_ORDER = (
    "identity", "audience_tone", "scope", "refusal_policy", "grounding_rules",
    "citation_rules", "ambiguity_policy", "tool_policy", "output_contract",
)
EMPTY = "(none)"


class FewShot(BaseModel):
    """A format-only example. Never contains an eval question's answer (leakage-checked)."""

    question: str
    reply: dict[str, Any]


class PromptSections(BaseModel):
    identity: str
    audience_tone: str = ""
    scope: str = ""
    refusal_policy: str = ""
    grounding_rules: str = ""
    citation_rules: str = ""
    ambiguity_policy: str = ""
    tool_policy: str = ""
    output_contract: str = ""
    few_shot: list[FewShot] = Field(default_factory=list)


class Chunking(BaseModel):
    strategy: str = "heading_recursive"
    size_tokens: int = 350
    overlap_tokens: int = 60


class Hybrid(BaseModel):
    dense_weight: float = 0.6
    bm25_weight: float = 0.4
    fusion: str = "rrf"


class Retrieval(BaseModel):
    chunking: Chunking = Field(default_factory=Chunking)
    csv_mode: str = "row_as_record"
    hybrid: Hybrid = Field(default_factory=Hybrid)
    top_k: int = 5
    min_score: float = 0.25  # minimum dense cosine similarity for a chunk to count as evidence
    query_rewrite: bool = False


class AgentModel(BaseModel):
    name: str
    temperature: float = 0.1
    seed: int = 7


class Corpus(BaseModel):
    path: str  # folder of documents, relative to the agent.yaml file (or absolute)


class AgentDef(BaseModel):
    schema_version: int = 1
    name: str
    model: AgentModel
    corpus: Corpus
    prompt_sections: PromptSections
    retrieval: Retrieval = Field(default_factory=Retrieval)
    tools: list[str] = Field(default_factory=list)
    tool_justifications: dict[str, str] = Field(default_factory=dict)
    output_contract: str = "contracts/answer_v1.json"
    provenance: dict[str, Any] = Field(default_factory=dict)

    def corpus_dir(self, base_dir: Path) -> Path:
        path = Path(self.corpus.path)
        return path if path.is_absolute() else (base_dir / path).resolve()


def load_agent(path: Path) -> AgentDef:
    return AgentDef.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def save_agent(agent: AgentDef, path: Path) -> None:
    path.write_text(
        yaml.safe_dump(agent.model_dump(mode="json"), sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )


def compile_system_prompt(agent: AgentDef) -> str:
    """Assemble the sections, tool descriptions and contract schema into the system prompt."""
    sections = agent.prompt_sections
    variables: dict[str, str] = {name: getattr(sections, name).strip() or EMPTY for name in SECTION_ORDER}
    if agent.tools:
        descriptions = "\n".join(f"- {tool_description(name)}" for name in agent.tools)
        variables["tool_block"] = load_prompt("runner/tool_protocol").render({"tools": descriptions})[0]
    else:
        variables["tool_block"] = EMPTY
    variables["contract_schema"] = json.dumps(load_contract_schema(), indent=2)
    variables["few_shot"] = (
        "\n\n".join(
            f"Question: {ex.question}\nReply: {json.dumps(ex.reply, ensure_ascii=False)}"
            for ex in sections.few_shot
        )
        or EMPTY
    )
    return load_prompt("runner/system_compile").render(variables)[0]
