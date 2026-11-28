"""Frozen eval sets, optional-tool lookup cases and the derived-value rule of the compose validator (P6.1)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from forgeai.evals.compose import asks_for_derived_value
from forgeai.evals.freeze import (
    EvalSetChangedError, EvalSetNotFrozenError, assert_eval_set_frozen, eval_set_hash, freeze_eval_set,
)
from forgeai.evals.models import EvalCase, KeyFact
from forgeai.judge.layer1 import check_layer1
from forgeai.judge.score import decide
from tests.test_evals import config  # noqa: F401  (fixture)
from tests.test_judge import TEXT_A, kf, run_of, span
from forgeai.judge.models import Layer2
from forgeai.runner.contract import ToolCallRecord

ROOT = Path(__file__).parents[1]
TODAY = date(2026, 11, 27)


def make_run(tmp_path: Path) -> Path:
    (tmp_path / "03_evals").mkdir()
    (tmp_path / "03_evals" / "eval_set.jsonl").write_text('{"id": "a-1"}\n{"id": "a-2"}\n', encoding="utf-8")
    return tmp_path


def test_a_run_whose_eval_set_was_never_frozen_is_refused(tmp_path) -> None:
    with pytest.raises(EvalSetNotFrozenError):
        assert_eval_set_frozen(make_run(tmp_path))


def test_the_frozen_hash_is_checked_and_any_change_to_the_eval_set_is_refused(tmp_path, config) -> None:
    run = make_run(tmp_path)
    entry = freeze_eval_set(run, config, digests={})
    assert entry["cases"] == 2 and entry["sha256"] == eval_set_hash(run)
    assert json.loads((run / "manifest.json").read_text(encoding="utf-8"))["eval_set"]["sha256"] == entry["sha256"]
    assert assert_eval_set_frozen(run) == entry["sha256"]
    (run / "03_evals" / "eval_set.jsonl").write_text('{"id": "a-1"}\n{"id": "a-2", "question": "edited"}\n', encoding="utf-8")
    with pytest.raises(EvalSetChangedError, match="changed after it was frozen"):
        assert_eval_set_frozen(run)


FROZEN_RUNS = sorted((ROOT / "runs" / "p6_1_checkpoint").glob("*/manifest.json"))


@pytest.mark.parametrize("manifest", FROZEN_RUNS or [None], ids=lambda p: p.parent.name if p else "none")
def test_every_committed_p6_1_eval_set_still_matches_its_frozen_hash(manifest) -> None:
    if manifest is None:
        pytest.skip("no frozen runs committed yet")
    assert_eval_set_frozen(manifest.parent)


def test_the_baseline_command_refuses_a_run_whose_eval_set_changed(tmp_path, config) -> None:
    from typer.testing import CliRunner

    from forgeai.cli import app

    run = make_run(tmp_path)
    freeze_eval_set(run, config, digests={})
    (run / "03_evals" / "eval_set.jsonl").write_text("{}\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["baseline", "--run", str(run), "--out", str(tmp_path / "out"), "--canaries", str(tmp_path / "c.json")])
    assert result.exit_code == 7 and "changed after it was frozen" in result.output.replace("\n", " ")


def test_a_merge_that_asks_for_a_total_is_rejected_but_a_merge_of_two_values_is_not() -> None:
    first, second = "What is the unit price of part MT-6017 from supplier S01?", "What discount applies to orders of USD 15,000 to USD 49,999?"
    assert asks_for_derived_value("What is the total cost for MT-6017 including the discount for orders of USD 15,000?", first, second) == {"total"}
    assert not asks_for_derived_value("What are the unit price of MT-6017 and the cost of a library card?", first, second)
    assert not asks_for_derived_value("What is the unit price of MT-6017 and the discount for orders of USD 15,000?", first, second)
    assert asks_for_derived_value("How much would 10 units cost?", first, second)
    assert not asks_for_derived_value("What is the total price?", "What is the total price?", second)  # the parent already asked


def lookup_case(**kw) -> EvalCase:
    base = dict(id="t-1", type="tool_required", question="What is the minimum order quantity of HD-1?", expected_status="answered",
                split="dev", seed_spans=[span("a.md", TEXT_A, 100)], key_facts=[kf("24", "number")], expected_sources=["a.md"],
                expected_tools=["csv_lookup"], tool_mode="optional")
    return EvalCase(**{**base, **kw})


def test_an_optional_tool_lookup_passes_with_the_value_in_a_cited_row_or_in_a_tool_result_but_not_without_either() -> None:
    case = lookup_case()
    cited = check_layer1(case, run_of("The minimum is 24."), today=TODAY)  # no tool call, the cited row covers it
    assert cited.grounded_ok and cited.tool_ok is None and not decide(case, cited, Layer2())
    result = [ToolCallRecord(tool="csv_lookup", args={}, result="moq: 24")]
    via_tool = check_layer1(case, run_of("The minimum is 24.", cite=[], tools=result), today=TODAY)
    assert via_tool.grounded_ok and not decide(case, via_tool, Layer2())
    nowhere = check_layer1(case, run_of("The minimum is 24.", cite=[]), today=TODAY)
    assert nowhere.grounded_ok is False and "not grounded" in decide(case, nowhere, Layer2())[0]
    wrong = check_layer1(case, run_of("The minimum is 25."), today=TODAY)
    assert decide(case, wrong, Layer2())
    failed_tool = [ToolCallRecord(tool="csv_lookup", args={}, result="ERROR: no such file")]
    assert check_layer1(case, run_of("The minimum is 24.", cite=[], tools=failed_tool), today=TODAY).grounded_ok is False


def test_a_computed_case_still_requires_its_tool() -> None:
    case = lookup_case(tool_mode="required", expected_tools=["calculator"], key_facts=[KeyFact(text="24", kind="number")])
    no_tool = check_layer1(case, run_of("The total is 24."), today=TODAY)
    assert no_tool.tool_ok is False and any("tool check failed" in r for r in decide(case, no_tool, Layer2()))


def test_the_hash_does_not_change_when_a_checkout_converts_line_endings(tmp_path, config) -> None:
    run = make_run(tmp_path)
    freeze_eval_set(run, config, digests={})
    path = run / "03_evals" / "eval_set.jsonl"
    lf = path.read_bytes().replace(bytes([13, 10]), bytes([10]))
    path.write_bytes(lf.replace(bytes([10]), bytes([13, 10])))  # CRLF whatever the file had before
    assert assert_eval_set_frozen(run)
