"""Stage 4: requirements + guard output -> ``02_agent/agent.v0.yaml`` and ``tool_selection.json``.

This stage reads only the requirements, the guard's coverage/calibration and the document
profile. It never sees an eval set (that stage never sees this one's output either).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

import forgeai
from forgeai.config import ForgeConfig
from forgeai.generator.retrieval_rules import retrieval_from_profile
from forgeai.generator.sections import build_sections
from forgeai.generator.tool_selector import ToolSelection, select_tools
from forgeai.ingest.models import DocRecord
from forgeai.intake.schema import load_requirements
from forgeai.manifest import prompt_versions
from forgeai.runner.agent_def import AgentDef, AgentModel, Corpus, save_agent


def generate_agent(run_dir: Path, docs_dir: Path, config: ForgeConfig) -> AgentDef:
    """Generate ``agent.v0.yaml`` for the run in ``run_dir`` (which holds 00_intake/ and 01_guard/)."""
    intake, guard = run_dir / "00_intake", run_dir / "01_guard"
    req = load_requirements(intake / "requirements.yaml")
    brief = (intake / "brief.txt").read_text(encoding="utf-8")
    records = [DocRecord(**r) for r in json.loads((intake / "doc_inventory.json").read_text(encoding="utf-8"))]
    calibration: dict[str, Any] = json.loads((guard / "coverage.json").read_text(encoding="utf-8"))["calibration"]

    has_table = any(r.status == "ok" and r.structure and r.structure.kind == "tabular" for r in records)
    selection = select_tools(brief, {t.id: t.topic for t in req.must_answer_topics}, has_table, config)
    sections = build_sections(req, selection, config)
    retrieval, why = retrieval_from_profile(records, calibration)

    agent_dir = run_dir / "02_agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = os.path.relpath(docs_dir.resolve(), agent_dir.resolve()).replace("\\", "/")
    agent = AgentDef(
        name=req.agent_name,
        model=AgentModel(name=config.agent_model.name, temperature=config.agent_model.temperature,
                         seed=config.agent_model.seed),
        corpus=Corpus(path=corpus_path),
        prompt_sections=sections,
        retrieval=retrieval,
        tools=[d.tool for d in selection.attached],
        tool_justifications={d.tool: f"{d.requirement_id}: {d.justification}" for d in selection.attached},
        provenance={
            "requirements": "../00_intake/requirements.yaml",
            "tool_selection": "tool_selection.json",
            "generated_by": f"forgeai {forgeai.__version__}",
            "prompt_versions": {k: v for k, v in prompt_versions().items() if k.startswith(("generator/", "runner/"))},
            "retrieval_rationale": why,
            "fewshot_leakage_check": "pending: needs the eval set (P5); see forgeai/generator/fewshot.py",
        },
    )
    save_agent(agent, agent_dir / "agent.v0.yaml")
    write_tool_selection(selection, agent_dir / "tool_selection.json")
    return agent


def write_tool_selection(selection: ToolSelection, path: Path) -> None:
    path.write_text(json.dumps(selection.to_dict(), indent=2), encoding="utf-8")
