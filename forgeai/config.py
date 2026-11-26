"""Configuration: model pins, thresholds, hard limits and the model-family check.

The judge model must come from a different model family than the agent/generator
model (spec §0, §6.2). That rule is enforced here, at config load, by raising
``ConfigError``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class ConfigError(Exception):
    """Raised when the configuration violates a rule ForgeAI refuses to run under.

    Deliberately not a ``ValueError`` so Pydantic does not wrap it into a
    ``ValidationError``: callers see the real reason.
    """


def model_family(model_name: str) -> str:
    """Return the model family for an Ollama tag, e.g. ``qwen3:8b`` -> ``qwen``.

    The family is the alphabetic stem of the model's base name, ignoring any
    registry path, version digits and size tag. ``llama3.1:8b`` -> ``llama``,
    ``hf.co/org/Gemma-3-4B-GGUF`` -> ``gemma``.
    """
    base = model_name.strip().lower().split("/")[-1].split(":")[0]
    match = re.match(r"[a-z]+", base)
    if not match:
        raise ConfigError(f"Cannot determine model family from {model_name!r}")
    return match.group(0)


class ModelSpec(BaseModel):
    """One pinned model: tag, sampling settings and (once known) its digest."""

    name: str
    temperature: float = 0.1
    seed: int = 7
    digest: str | None = None

    @property
    def family(self) -> str:
        return model_family(self.name)


class Thresholds(BaseModel):
    """Pass criteria (spec §1.5). Configurable, never hardcoded in logic."""

    dev_pass_rate: float = 0.90
    max_fabricated_on_refusals: int = 0
    canary_accuracy_min: float = 0.90
    paraphrase_overlap_max: float = 0.50
    dedup_similarity: float = 0.90
    fewshot_leakage_similarity: float = 0.80
    fact_fuzzy_score: int = 85


class Limits(BaseModel):
    """Hard limits taken from the assignment, enforced in code (spec §1)."""

    max_clarifying_questions: int = 5
    max_fix_cycles: int = 3
    min_eval_cases: int = 20
    max_tool_calls_per_turn: int = 3
    max_llm_retries: int = 2


class ForgeConfig(BaseModel):
    ollama_host: str = "http://localhost:11434"
    ollama_timeout_s: float = 600.0
    num_ctx: int = 8192
    max_output_tokens: int = 1536  # cap so a degenerate generation cannot run for minutes
    think: bool = False  # Qwen3 reasoning mode; off for every role (agent, generator, judge)
    agent_model: ModelSpec
    judge_model: ModelSpec
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_revision: str | None = None  # Hugging Face commit hash, pinned for reproducibility
    # bge models embed *queries* with an instruction prefix and documents without one.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    cache_dir: Path = Path(".cache")
    thresholds: Thresholds = Field(default_factory=Thresholds)
    limits: Limits = Field(default_factory=Limits)

    @model_validator(mode="after")
    def _judge_family_differs(self) -> ForgeConfig:
        if self.agent_model.family == self.judge_model.family:
            raise ConfigError(
                "Judge model must come from a different family than the agent model "
                f"(both are {self.agent_model.family!r}: "
                f"agent={self.agent_model.name!r}, judge={self.judge_model.name!r})."
            )
        return self

    @property
    def llm_cache_dir(self) -> Path:
        return self.cache_dir / "llm"

    @property
    def index_cache_dir(self) -> Path:
        return self.cache_dir / "index"


def load_config(path: str | Path = "forgeai.yaml") -> ForgeConfig:
    """Load and validate ``forgeai.yaml``. Raises ``ConfigError`` on rule violations."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return ForgeConfig.model_validate(raw)
