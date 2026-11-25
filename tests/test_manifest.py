import json
from pathlib import Path

from forgeai.config import ForgeConfig, ModelSpec
from forgeai.manifest import prompt_versions, write_manifest


def test_manifest_records_models_digests_embedding_revision_and_prompts(tmp_path: Path) -> None:
    config = ForgeConfig(
        agent_model=ModelSpec(name="qwen3:8b"), judge_model=ModelSpec(name="gemma3:4b"),
        embedding_revision="abc123",
    )
    digests = {"qwen3:8b": "sha-q", "gemma3:4b": "sha-g"}
    write_manifest(tmp_path / "manifest.json", config, digests)
    data = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert data["models"]["agent"]["digest"] == "sha-q" and data["models"]["judge"]["digest"] == "sha-g"
    assert data["embedding"] == {"name": "BAAI/bge-small-en-v1.5", "revision": "abc123"}
    assert data["prompt_versions"]["runner/agent_turn"] == 1
    assert data["limits"]["max_fix_cycles"] == 3


def test_prompt_versions_cover_every_prompt_file() -> None:
    assert {"_repair", "ingest/summary", "tools/calculator"} <= set(prompt_versions())
