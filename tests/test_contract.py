from forgeai.ingest.models import Chunk
from forgeai.runner.contract import (
    Answer, Citation, ToolCallRecord, contract_issues, load_contract_schema, schema_issues,
)
from forgeai.textmatch import is_verbatim, normalize

CHUNK = Chunk(
    chunk_id="rules.md#c1", doc="rules.md", index=1,
    text="Books may be borrowed for 21 days.\nDVDs may be borrowed for 7 days.",
)
BY_ID = {CHUNK.chunk_id: CHUNK}


def answer(**kwargs) -> Answer:
    base = {"status": "answered", "answer": "21 days.", "citations": [
        Citation(doc="rules.md", chunk_id="rules.md#c1", quote="Books may be borrowed for 21 days.")]}
    return Answer(**{**base, **kwargs})


def test_pydantic_model_matches_committed_json_schema() -> None:
    schema = load_contract_schema()
    assert set(schema["properties"]) == set(Answer.model_fields)
    full = answer(tool_calls=[ToolCallRecord(tool="calculator", args={"expression": "1+1"}, result="2")])
    assert schema_issues(full) == []


def test_valid_answer_has_no_issues() -> None:
    assert contract_issues(answer(), BY_ID) == []


def test_quote_matching_ignores_whitespace_but_nothing_else() -> None:
    assert is_verbatim("DVDs may be borrowed\nfor 7 days.", CHUNK.text)
    assert not is_verbatim("dvds may be borrowed for 7 days.", CHUNK.text)  # case-sensitive
    assert not is_verbatim("Books may be borrowed for 28 days.", CHUNK.text)
    assert not is_verbatim("", CHUNK.text)
    assert normalize("a \n\t b") == "a b"


def test_fabricated_quote_is_flagged() -> None:
    bad = answer(citations=[Citation(doc="rules.md", chunk_id="rules.md#c1", quote="Books last 30 days.")])
    assert any("not a verbatim span" in i for i in contract_issues(bad, BY_ID))


def test_unknown_chunk_and_wrong_doc_are_flagged() -> None:
    missing = answer(citations=[Citation(doc="rules.md", chunk_id="rules.md#c9", quote="x")])
    wrong_doc = answer(citations=[Citation(doc="other.md", chunk_id="rules.md#c1", quote="DVDs")])
    assert any("does not exist" in i for i in contract_issues(missing, BY_ID))
    assert any("belongs to" in i for i in contract_issues(wrong_doc, BY_ID))


def test_clarification_and_empty_answer_rules() -> None:
    assert contract_issues(Answer(status="needs_clarification"), BY_ID)
    assert contract_issues(Answer(status="needs_clarification", clarifying_question="Which one?"), BY_ID) == []
    assert contract_issues(Answer(status="answered", answer="  "), BY_ID)
    assert contract_issues(Answer(status="refused", answer="Please ask staff."), BY_ID) == []
