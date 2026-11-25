from pathlib import Path

import pytest

from forgeai.config import ConfigError, ForgeConfig, ModelSpec, load_config, model_family


@pytest.mark.parametrize(
    ("tag", "family"),
    [
        ("qwen3:8b", "qwen"),
        ("qwen2.5:7b-instruct", "qwen"),
        ("gemma3:4b", "gemma"),
        ("llama3.1:8b", "llama"),
        ("hf.co/someorg/Gemma-3-4B-GGUF:Q4_K_M", "gemma"),
        ("Mistral:7b", "mistral"),
    ],
)
def test_model_family(tag: str, family: str) -> None:
    assert model_family(tag) == family


def test_same_family_judge_is_rejected() -> None:
    with pytest.raises(ConfigError, match="different family"):
        ForgeConfig(
            agent_model=ModelSpec(name="qwen3:8b"),
            judge_model=ModelSpec(name="qwen2.5:7b"),
        )


@pytest.mark.parametrize("judge", ["gemma3:4b", "llama3.1:8b"])
def test_different_family_judge_is_accepted(judge: str) -> None:
    cfg = ForgeConfig(
        agent_model=ModelSpec(name="qwen3:8b"), judge_model=ModelSpec(name=judge)
    )
    assert cfg.judge_model.family != cfg.agent_model.family


def test_shipped_forgeai_yaml_loads() -> None:
    cfg = load_config(Path(__file__).parents[1] / "forgeai.yaml")
    assert cfg.limits.max_clarifying_questions == 5
    assert cfg.limits.max_fix_cycles == 3
    assert cfg.limits.min_eval_cases == 20


def test_same_family_in_yaml_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "agent_model: {name: qwen3:8b}\njudge_model: {name: qwen3:4b}\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        load_config(bad)


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
