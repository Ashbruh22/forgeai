"""Demo corpora (spec §11) and the hidden-fact-sheet isolation rule."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from forgeai.ingest.loaders import load_document
from forgeai.textmatch import normalize

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "forgeai"
HR = ROOT / "data" / "hr_shore"
PROC = ROOT / "data" / "spares_procurement"
SYNTH = ROOT / "data" / "synthesis"


def test_fact_sheets_are_never_read_by_the_factory() -> None:
    """forgeai/evals/ must never read a fact sheet; only forgeai/audit/ may (spec §6.1)."""
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE)
        if relative.parts[0] == "audit":
            continue
        text = path.read_text(encoding="utf-8")
        if "fact_sheet" in text or "data/synthesis" in text or "synthesis/" in text:
            offenders.append(str(relative))
    assert offenders == []


def test_evals_package_does_not_import_the_audit_package() -> None:
    evals = PACKAGE / "evals"
    if not evals.exists():  # built in P5; the scan above already covers everything else
        pytest.skip("forgeai/evals/ does not exist yet")
    for path in evals.rglob("*.py"):
        assert "forgeai.audit" not in path.read_text(encoding="utf-8"), path


def test_synthesis_prompts_and_plans_are_committed() -> None:
    assert (SYNTH / "prompts" / "section.md").is_file()
    for demo in ("hr_shore", "spares_procurement"):
        assert (SYNTH / demo / "plan.yaml").is_file()
        assert (SYNTH / demo / "fact_sheet.json").is_file(), demo


@pytest.mark.parametrize("demo", ["hr_shore", "spares_procurement"])
def test_every_planted_fact_is_verified_in_the_rendered_corpus(demo: str) -> None:
    sheet = json.loads((SYNTH / demo / "fact_sheet.json").read_text(encoding="utf-8"))
    texts = {p.name: normalize(load_document(p).canonical_text).lower()
             for p in (ROOT / "data" / demo / "docs").glob("*") if p.suffix in {".pdf", ".csv"}}
    planted = [f for f in sheet["facts"] if f.get("must_contain")]
    assert planted
    for fact in planted:
        corpus = texts[fact["doc"]]
        assert all(normalize(m).lower() in corpus for m in fact["must_contain"]), fact["id"]


def test_demo_a_has_three_pdfs_of_four_to_six_pages_and_no_payroll_document() -> None:
    pdfs = sorted((HR / "docs").glob("*.pdf"))
    assert [p.name for p in pdfs] == ["code_of_conduct_it_policy.pdf", "leave_policy.pdf", "travel_expense_policy.pdf"]
    for pdf in pdfs:
        assert 4 <= load_document(pdf).page_count <= 6, pdf.name
    everything = " ".join(load_document(p).canonical_text.lower() for p in pdfs)
    for word in ("payroll", "payslip", "salary", "wages"):
        assert word not in everything, word  # the planted gap: no payroll content anywhere


def test_demo_a_cross_document_rule_and_ambiguous_term_are_planted() -> None:
    leave = normalize(load_document(HR / "docs" / "leave_policy.pdf").canonical_text)
    travel = normalize(load_document(HR / "docs" / "travel_expense_policy.pdf").canonical_text)
    assert "Travel and Expense Policy" in leave and "time off in lieu" in leave
    assert "one day of time off in lieu" in travel and "not deducted from annual leave" in travel
    sheet = json.loads((SYNTH / "hr_shore" / "fact_sheet.json").read_text(encoding="utf-8"))
    assert any(f["kind"] == "ambiguity" for f in sheet["facts"])
    assert any(f["kind"] == "planted_gap" for f in sheet["facts"])
    assert sheet["expected_tools"] == []


def test_demo_b_catalogue_and_supplier_terms() -> None:
    with (PROC / "docs" / "spares_catalogue.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == ["part_no", "description", "equipment", "maker", "supplier_id",
                                     "unit_price_usd", "moq", "lead_time_days", "stock_port"]
    assert 280 <= len(rows) <= 320
    assert {r["supplier_id"] for r in rows} == {"S01", "S02", "S03"}
    assert len({(r["part_no"], r["supplier_id"]) for r in rows}) == len(rows)
    assert len(list((PROC / "docs").glob("terms_*.pdf"))) == 3
    sheet = json.loads((SYNTH / "spares_procurement" / "fact_sheet.json").read_text(encoding="utf-8"))
    assert sheet["expected_tools"] == ["csv_lookup", "calculator", "current_date"]
    assert sheet["not_expected_tools"] == ["weather"]


def test_briefs_match_the_spec() -> None:
    assert "salary slip" in (HR / "brief.txt").read_text(encoding="utf-8")
    assert "in stock right now" in (PROC / "brief.txt").read_text(encoding="utf-8")


def test_catalogue_generation_is_deterministic() -> None:
    import sys

    sys.path.insert(0, str(SYNTH))
    import synthesize

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        first = synthesize.make_catalogue(Path(tmp) / "a.csv")
        second = synthesize.make_catalogue(Path(tmp) / "b.csv")
        assert first == second
        assert (Path(tmp) / "a.csv").read_text(encoding="utf-8") == (PROC / "docs" / "spares_catalogue.csv").read_text(encoding="utf-8")
