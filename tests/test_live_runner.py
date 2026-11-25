"""P2 exit test (live): a hand-written agent.yaml answers with verified citations.

Uses the real embedding model and the real agent model. Skipped loudly if Ollama or a
model is missing; never mocked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forgeai.config import load_config
from forgeai.embeddings import SentenceTransformerEmbedder
from forgeai.llm import LLMClient, set_client
from forgeai.runner.agent_def import load_agent
from forgeai.runner.runner import AgentRunner
from tests.live import require_models

pytestmark = pytest.mark.live
SMOKE = Path(__file__).parents[1] / "data" / "_smoke"


@pytest.fixture(scope="module")
def runner(tmp_path_factory: pytest.TempPathFactory):
    config = load_config().model_copy(update={"cache_dir": tmp_path_factory.mktemp("cache")})
    require_models(config.agent_model.name)
    set_client(LLMClient(config, trace_path=config.cache_dir / "trace.jsonl"))
    yield AgentRunner(
        load_agent(SMOKE / "agent.yaml"), SMOKE, config, SentenceTransformerEmbedder(config.embedding_model, config.embedding_revision)
    )
    set_client(None)


def test_answers_with_verified_citation(runner: AgentRunner) -> None:
    result = runner.ask("How many days can I borrow a book for?")
    assert result.contract_valid, result.issues  # every citation quote is a verbatim chunk span
    assert result.answer.status == "answered"
    assert "21" in result.answer.answer
    assert result.answer.citations, "an answered question must cite its source"
    assert any(c.doc == "library_rules.md" for c in result.answer.citations)


def test_refuses_out_of_scope_question(runner: AgentRunner) -> None:
    result = runner.ask("Who won the football world cup in 2018?")
    assert result.answer.status == "refused", result.answer


def test_answers_tabular_question_correctly_with_valid_citation(runner: AgentRunner) -> None:
    # The matching table row is retrieved, so answering without the tool is legitimate;
    # what must hold is a correct value and a verified citation.
    result = runner.ask("What is the hourly fee for Studio A?")
    assert result.contract_valid, result.issues
    assert "15" in result.answer.answer


def test_arithmetic_over_table_values_is_correct(runner: AgentRunner) -> None:
    result = runner.ask("What is the total cost to hire Studio A for 3 hours and Studio B for 2 hours?")
    print("tools used:", [c.tool for c in result.answer.tool_calls])
    assert "65" in result.answer.answer, result.answer  # 3 * 15 + 2 * 10
