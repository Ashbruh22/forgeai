"""Invariants that must hold for every committed eval set (runs/**/03_evals/eval_set.jsonl)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgeai.evals.build import load_eval_set
from forgeai.evals.consistency import assert_cases_consistent
from forgeai.ingest.loaders import load_document
from forgeai.intake.schema import load_requirements

ROOT = Path(__file__).parents[1]
RUNS = sorted((ROOT / "runs" / "p4_checkpoint").glob("*/03_evals/eval_set.jsonl"))
DOCS = {"hr_shore": ROOT / "data" / "hr_shore" / "docs", "spares_procurement": ROOT / "data" / "spares_procurement" / "docs",
        "library_gap": ROOT / "data" / "_smoke" / "docs"}


@pytest.fixture(params=RUNS or [None], ids=lambda p: p.parents[1].name if p else "none")
def run(request):
    if request.param is None:
        pytest.skip("no committed eval sets yet")
    path: Path = request.param
    return path.parents[1], load_eval_set(path), DOCS[path.parents[1].name]


def test_every_case_is_consistent_with_the_requirements(run) -> None:
    run_dir, cases, _ = run
    assert_cases_consistent(cases, load_requirements(run_dir / "00_intake" / "requirements.yaml"))


def test_every_seed_span_is_real_text_at_real_offsets(run) -> None:
    _, cases, docs = run
    for case in cases:
        for span in [*case.seed_spans, *[i.seed_span for i in case.interpretations]]:
            canonical = load_document(docs / span.doc).canonical_text
            assert canonical[span.char_start:span.char_end] == span.span_text, (case.id, span.doc)


def test_ids_are_unique_the_split_is_complete_and_expectations_are_well_formed(run) -> None:
    run_dir, cases, _ = run
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))
    split = json.loads((run_dir / "03_evals" / "split.json").read_text())
    assert sorted(split["dev"] + split["test"]) == sorted(ids) and not set(split["dev"]) & set(split["test"])
    for case in cases:
        if case.type == "out_of_scope":
            assert case.expected_status == "refused" and not case.key_facts
        elif case.type == "ambiguous":
            assert case.expected_status == "needs_clarification" and len(case.interpretations) == 2
        else:
            assert case.expected_status == "answered" and case.key_facts and case.seed_spans
        assert all(len(f.text.split()) <= 10 for f in case.key_facts), case.id  # atomic facts only


def test_tool_cases_have_code_computed_values_and_snapshot_cases_expect_the_date(run) -> None:
    _, cases, _ = run
    for case in (c for c in cases if c.type == "tool_required"):
        assert case.validation["expected_value_computed_by_code"] is True and case.expected_tools
    for case in (c for c in cases if "snapshot" in c.tags):
        assert any(f.kind == "snapshot_date" for f in case.key_facts)


def test_rejected_cases_are_kept_with_reasons(run) -> None:
    run_dir, _, _ = run
    rejected = (run_dir / "03_evals" / "rejected_cases.jsonl").read_text().splitlines()
    assert rejected and all(json.loads(line)["reason"] for line in rejected)
    assert (run_dir / "03_evals" / "prompts_used" / "versions.yaml").is_file()
