"""P7: classifier, patch library, anti-overfitting guard, acceptance gate and the loop (with scripted evaluators)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgeai.evals.models import EvalCase, KeyFact
from forgeai.ingest.models import Chunk
from forgeai.ingest.spans import SeedSpan
from forgeai.judge.models import CaseRun, CaseScore, ClaimVerdict, FactCheck, Layer1, Layer2
from forgeai.optimizer import guard
from forgeai.optimizer.classify import (
    HALLUCINATION, PROMPT_DEFECT, RETRIEVAL_MISS, WRONG_REFUSAL, Failure, classify, pick_category,
)
from forgeai.optimizer.gate import acceptance_gate, noise_flips
from forgeai.optimizer.loop import DevResult, optimize, target_met
from forgeai.optimizer.patches import apply_ops, candidates, check_bounds
from forgeai.runner.agent_def import AgentDef
from forgeai.runner.contract import Answer
from forgeai.runner.retrieval import Hit, reserve_slots
from tests.test_evals import config  # noqa: F401  (fixture)


def agent(**retrieval) -> AgentDef:
    return AgentDef.model_validate({
        "name": "A", "model": {"name": "qwen3:8b", "temperature": 0.1}, "corpus": {"path": "docs"},
        "prompt_sections": {"identity": "You are A.", "scope": "In scope: leave.", "refusal_policy": "Refuse:\n- payroll\n- legal advice",
                            "grounding_rules": "Answer from the context.", "citation_rules": "Cite."},
        "retrieval": {"top_k": 3, "min_score": 0.3, "min_score_bounds": [0.15, 0.45], **retrieval}})


def span(doc: str = "a.md") -> SeedSpan:
    return SeedSpan(doc=doc, char_start=0, char_end=20, span_text="x" * 20)


def case(cid: str = "c1", kind: str = "answerable", expected: str = "answered", **kw) -> EvalCase:
    base = dict(id=cid, type=kind, question=f"What is the limit for item {cid}?", expected_status=expected, split="dev",
                key_facts=[KeyFact(text="24", kind="number")] if expected == "answered" else [], seed_spans=[span()] if expected == "answered" else [])
    return EvalCase(**{**base, **kw})


def score(c: EvalCase, *, status: str = "answered", passed: bool = False, hit: bool | None = True, contract: bool = True, unsupported: int = 0,
          facts_ok: bool = True, citation_issue: bool = False, source_ok: bool | None = True, tool_ok: bool | None = None,
          low: bool = False, docs_hit: list[str] | None = None, covered: tuple[int, int] | None = None) -> CaseScore:
    claims = [ClaimVerdict(claim="x", supported=False, verdict="unsupported", missing="nothing says so")] * unsupported
    l1 = Layer1(contract_valid=contract, contract_issues=[] if contract else ["bad"], status_ok=status == c.expected_status, status=status,
                citation_issues=["citation: bad"] if citation_issue else [], source_ok=source_ok, tool_ok=tool_ok,
                facts=[FactCheck(fact="24", kind="number", matched=facts_ok)], retrieval_hit=hit, retrieval_docs_hit=docs_hit or (["a.md"] if hit else []),
                interpretations_covered=covered[0] if covered else None, interpretations_total=covered[1] if covered else None)
    return CaseScore(case_id=c.id, type=c.type, split="dev", passed=passed, layer1=l1, layer2=Layer2(ran=True, claims=claims, unsupported=unsupported),
                     expected_status=c.expected_status, answer_status=status, low_confidence=low)


# ---- classifier --------------------------------------------------------------------------------------------------


def test_every_branch_of_the_decision_tree_gives_exactly_one_of_the_four_categories() -> None:
    answerable, refusal, amb = case("a"), case("r", "out_of_scope", "refused"), case("m", "ambiguous", "needs_clarification", seed_spans=[span()])
    table = [
        (answerable, score(answerable, contract=False), PROMPT_DEFECT, "output_contract"),
        (refusal, score(refusal, status="answered", unsupported=2), HALLUCINATION, "grounding_rules"),
        (refusal, score(refusal, status="answered"), PROMPT_DEFECT, "scope"),
        (answerable, score(answerable, status="refused", hit=True), WRONG_REFUSAL, "refusal_policy"),
        (answerable, score(answerable, status="refused", hit=False), RETRIEVAL_MISS, ""),
        (answerable, score(answerable, hit=False), RETRIEVAL_MISS, ""),
        (answerable, score(answerable, unsupported=1), HALLUCINATION, "grounding_rules"),
        (answerable, score(answerable, citation_issue=True), PROMPT_DEFECT, "citation_rules"),
        (answerable, score(answerable, source_ok=False), PROMPT_DEFECT, "citation_rules"),
        (answerable, score(answerable, tool_ok=False), PROMPT_DEFECT, "tool_policy"),
        (answerable, score(answerable, facts_ok=False), PROMPT_DEFECT, "scope"),
        (amb, score(amb, status="answered", covered=(1, 2)), PROMPT_DEFECT, "ambiguity_policy"),
        (amb, score(amb, status="refused", hit=True), WRONG_REFUSAL, "refusal_policy"),
    ]
    for c, s, category, section in table:
        got = classify(c, s)
        assert (got.category, got.section) == (category, section), (c.id, s.answer_status, got)


def test_a_multi_document_case_missing_one_of_its_documents_is_a_retrieval_miss() -> None:
    multi = case("m", "multidoc", seed_spans=[span("a.md"), span("b.md")])
    assert classify(multi, score(multi, hit=True, docs_hit=["a.md"])).category == RETRIEVAL_MISS
    assert classify(multi, score(multi, hit=True, docs_hit=["a.md", "b.md"], facts_ok=False)).category == PROMPT_DEFECT


def test_a_hallucination_decided_by_a_low_confidence_layer_two_verdict_is_flagged_and_deprioritised() -> None:
    c = case("a")
    flagged = classify(c, score(c, unsupported=1, low=True))
    assert flagged.category == HALLUCINATION and flagged.low_confidence
    solid_prompt = Failure("p1", PROMPT_DEFECT, "scope", "x")
    assert pick_category([flagged, flagged, flagged, solid_prompt])[0] == PROMPT_DEFECT  # high-confidence failures are targeted first
    assert pick_category([flagged, flagged])[0] == HALLUCINATION  # nothing else left: the flagged ones are still targeted


def test_the_category_with_most_failures_is_targeted_and_ties_follow_the_order_of_harm() -> None:
    f = lambda cat: Failure("x", cat, "", "")  # noqa: E731
    assert pick_category([f(RETRIEVAL_MISS)] * 3 + [f(HALLUCINATION)] * 2)[0] == RETRIEVAL_MISS
    assert pick_category([f(PROMPT_DEFECT), f(RETRIEVAL_MISS), f(WRONG_REFUSAL), f(HALLUCINATION)])[0] == HALLUCINATION
    assert pick_category([f(PROMPT_DEFECT), f(RETRIEVAL_MISS), f(WRONG_REFUSAL)])[0] == WRONG_REFUSAL
    assert pick_category([f(PROMPT_DEFECT), f(RETRIEVAL_MISS)])[0] == RETRIEVAL_MISS
    assert pick_category([]) == (None, {})


# ---- patch library -----------------------------------------------------------------------------------------------


def test_retrieval_patches_are_bounded_and_offered_in_order() -> None:
    a = agent()
    ids = [p.id for p in candidates(RETRIEVAL_MISS, a, [], ["What is MT-6017?"])]
    assert ids[0] == "retrieval_miss/top_k+2" and "retrieval_miss/query_rewrite" in ids and "retrieval_miss/chunk_size+30%" in ids
    top = next(p for p in candidates(RETRIEVAL_MISS, a, [], ["What is MT-6017?"]) if p.id.endswith("top_k+2"))
    patched = apply_ops(a, top.ops)
    assert patched.retrieval.top_k == 5 and patched.retrieval.hybrid.bm25_weight == 0.6  # exact codes in the question shift toward BM25
    assert abs(patched.retrieval.hybrid.dense_weight + patched.retrieval.hybrid.bm25_weight - 1) < 1e-9
    plain = next(p for p in candidates(RETRIEVAL_MISS, a, [], ["How long is the notice?"]) if p.id.endswith("top_k+2"))
    assert apply_ops(a, plain.ops).retrieval.hybrid.bm25_weight == 0.4
    full = agent(top_k=10)
    assert not any(p.id.endswith("top_k+2") for p in candidates(RETRIEVAL_MISS, full, [], []))  # top_k stops at 10
    assert candidates(RETRIEVAL_MISS, a, [], [], crowded_out=True)[0].id == "retrieval_miss/reserve_prose=1"
    assert not any("min_score" in p.id for p in candidates(RETRIEVAL_MISS, agent(min_score=0.15), [], []))  # at the lower bound already


def test_min_score_never_leaves_its_bounds_and_temperature_only_goes_down() -> None:
    a = agent()
    with pytest.raises(ValueError, match="min_score"):
        apply_ops(a, [{"op": "replace", "path": "/retrieval/min_score", "value": 0.9}])
    with pytest.raises(ValueError, match="temperature"):
        apply_ops(a, [{"op": "replace", "path": "/model/temperature", "value": 0.5}])
    with pytest.raises(ValueError, match="top_k"):
        apply_ops(a, [{"op": "replace", "path": "/retrieval/top_k", "value": 12}])
    with pytest.raises(ValueError, match="overlap"):
        apply_ops(a, [{"op": "replace", "path": "/retrieval/chunking/overlap_tokens", "value": 300}])
    assert apply_ops(a, [{"op": "replace", "path": "/model/temperature", "value": 0.0}]).model.temperature == 0.0
    check_bounds(a, a)


def test_each_category_offers_its_own_patches_and_text_patches_name_one_section() -> None:
    a = agent()
    hall = candidates(HALLUCINATION, a, [], [])
    assert [p.id for p in hall] == ["hallucination/grounding_rules", "hallucination/min_score+0.05", "hallucination/temperature=0"]
    assert hall[0].section == "grounding_rules" and hall[0].needs_text and not hall[1].needs_text
    wrong = candidates(WRONG_REFUSAL, a, [], [])
    assert wrong[0].section == "refusal_policy" and wrong[1].id == "wrong_refusal/min_score-0.05"
    defects = [Failure("c1", PROMPT_DEFECT, "citation_rules", ""), Failure("c2", PROMPT_DEFECT, "citation_rules", ""), Failure("c3", PROMPT_DEFECT, "scope", "")]
    prompt = candidates(PROMPT_DEFECT, a, defects, [])
    assert [p.section for p in prompt] == ["citation_rules", "scope"] and all(p.needs_text for p in prompt)
    assert candidates(HALLUCINATION, a, [], [], exclude=frozenset({"hallucination/grounding_rules"}))[0].id.endswith("min_score+0.05")


def chunk_hit(kind: str, name: str, score_: float) -> Hit:
    doc = "t.csv" if kind == "row" else "p.md"
    return Hit(Chunk(chunk_id=name, doc=doc, index=0, text=name, kind=kind), score_, 0.5, 0.0)  # type: ignore[arg-type]


def test_source_diversity_keeps_prose_slots_so_table_rows_cannot_crowd_the_documents_out() -> None:
    hits = [chunk_hit("row", f"r{i}", 1.0 - i / 10) for i in range(6)] + [chunk_hit("prose", "p0", 0.3), chunk_hit("prose", "p1", 0.2)]
    assert [h.chunk.chunk_id for h in hits[:3]] == ["r0", "r1", "r2"]
    kept = reserve_slots(hits, 3, 1)
    assert sorted(h.chunk.chunk_id for h in kept) == ["p0", "r0", "r1"]  # the lowest table row gave its slot to the best prose chunk
    assert [h.chunk.chunk_id for h in reserve_slots(hits, 3, 0)] == ["r0", "r1", "r2"]  # reserve 0 keeps the plain ranking
    assert len(reserve_slots(hits, 4, 2)) == 4 and sum(h.chunk.kind == "prose" for h in reserve_slots(hits, 4, 2)) == 2
    assert [h.chunk.chunk_id for h in reserve_slots(hits[:6], 3, 1)] == ["r0", "r1", "r2"]  # no prose above the threshold: unchanged


# ---- anti-overfitting guard --------------------------------------------------------------------------------------


def guard_cases() -> list[EvalCase]:
    return [case("g1", question="What is the stock port for part PP-8597 from supplier S01?", key_facts=[KeyFact(text="Rotterdam", kind="phrase"),
                                                                                                    KeyFact(text="2026-11-01", kind="snapshot_date")]),
            case("g2", question="How many copies of 'Pride and Prejudice' are available?", key_facts=[KeyFact(text="3", kind="number"),
                                                                                                    KeyFact(text="cost 15 dollars", kind="phrase")])]


def test_a_general_rule_passes_the_guard() -> None:
    text = "When a table row carries a date, state that date in the answer. Cite the chunk that holds each fact you state."
    assert guard.violations(text, guard_cases(), original="Cite.") == []


@pytest.mark.parametrize(("text", "needle"), [
    ("Always say that the stock port for part PP-8597 from supplier S01 is listed.", "copies"),
    ("The stock port is Rotterdam.", "Rotterdam"),
    ("Report the snapshot date 2026-11-01 in every answer.", "date"),
    ("Remember PP-8597 is special.", "pp-8597"),
    ("Books such as Pride and Prejudice have 3 copies.", "pride and prejudice"),
    ("A late fee can cost 15 dollars.", "15 dollars"),
])
def test_case_specific_text_is_rejected_with_a_reason(text, needle) -> None:
    reasons = guard.violations(text, guard_cases(), original="Cite.")
    assert reasons and any(needle in r for r in reasons), reasons


def test_words_the_section_already_had_are_not_counted_against_the_critic() -> None:
    original = "Answer questions about Pride and Prejudice and similar titles."
    assert guard.violations("Answer questions about Pride and Prejudice and similar titles, citing the chunk.", guard_cases(), original=original) == []


def test_the_refusal_policy_must_keep_every_listed_item_and_a_text_must_not_balloon() -> None:
    original = "Refuse:\n- payroll\n- legal advice\nGive a redirect."
    assert guard.refusal_items_kept(original, "Refuse:\n- payroll\n- legal advice\nAnswer from passages.") == []
    assert guard.refusal_items_kept(original, "Refuse only payroll.")
    assert guard.size_ok("short", "x" * 2000) and guard.size_ok("a" * 100, "b" * 120) == []


# ---- gate --------------------------------------------------------------------------------------------------------


def test_the_gate_needs_a_net_gain_above_the_noise_floor_and_at_most_one_regression() -> None:
    before = {"a": False, "b": False, "c": False, "d": True, "e": True}
    assert acceptance_gate(before, {**before, "a": True, "b": True, "c": True}, noise_floor=2).accepted  # +3 > 2
    assert not acceptance_gate(before, {**before, "a": True, "b": True}, noise_floor=2).accepted  # +2 is not above 2
    assert acceptance_gate(before, {**before, "a": True}, noise_floor=0).accepted
    both = acceptance_gate(before, {"a": True, "b": True, "c": True, "d": False, "e": False}, noise_floor=0)
    assert not both.accepted and both.regressed == ["d", "e"] and "regressed" in both.reason  # net +1 but two regressions
    one = acceptance_gate(before, {"a": True, "b": True, "c": True, "d": False, "e": True}, noise_floor=0)
    assert one.accepted and one.fixed == ["a", "b", "c"] and one.regressed == ["d"] and one.net_gain == 2
    assert noise_flips({"a": True, "b": False}, {"a": False, "b": False}) == ["a"]


# ---- the loop ----------------------------------------------------------------------------------------------------


class Scripted:
    """An evaluator that decides each case from the agent's parameters (a stand-in for running and judging an agent)."""

    def __init__(self, cases: dict[str, EvalCase], rule, confirm_rule=None) -> None:  # noqa: ANN001
        self.cases, self.rule, self.calls, self.confirm_rule, self.confirmed = cases, rule, [], confirm_rule, []

    def confirm(self, agent_def: AgentDef, case_ids: list[str], iteration: int) -> dict:
        self.confirmed.append(sorted(case_ids))
        rule = self.confirm_rule or self.rule
        return {cid: score(self.cases[cid], passed=rule(agent_def, self.cases[cid]), hit=rule(agent_def, self.cases[cid])) for cid in case_ids}

    def dev(self, agent_def: AgentDef, iteration: int) -> DevResult:
        self.calls.append(iteration)
        scores, runs = {}, {}
        for cid, c in self.cases.items():
            ok = self.rule(agent_def, c)
            scores[cid] = score(c, passed=ok, hit=ok)  # a failing case is a retrieval miss
            runs[cid] = CaseRun(case_id=cid, answer=Answer(status="answered", answer="x"))
        return DevResult(scores, runs)


def seeded_bug_setup(tmp_path: Path, n: int = 6):
    cases = {f"c{i}": case(f"c{i}") for i in range(n)}
    # the seeded retrieval bug: top_k=3 leaves the gold chunk out; any top_k of 5 or more finds it
    evaluator = Scripted(cases, lambda a, c: a.retrieval.top_k >= 5)
    baseline = evaluator.dev(agent(), 0)
    return cases, evaluator, baseline


def test_a_seeded_retrieval_bug_is_classified_and_fixed_in_one_cycle(tmp_path, config) -> None:
    cases, evaluator, baseline = seeded_bug_setup(tmp_path)
    assert all(not s.passed for s in baseline.scores.values())
    hooked = []
    result = optimize(agent0=agent(), cases=cases, baseline=baseline, evaluator=evaluator, config=config, out=tmp_path, corpus_dir=tmp_path / "docs",
                      noise_floor=1, after_iteration=lambda n, a, ok, ch: hooked.append((n, ok)))
    change = json.loads((tmp_path / "04_iterations" / "iter_1" / "change.json").read_text(encoding="utf-8"))
    assert change["category"] == RETRIEVAL_MISS and change["patch_id"] == "retrieval_miss/top_k+2" and change["accepted"]
    assert change["score_before"]["dev"]["passes"] == 0 and change["score_after"]["dev"]["passes"] == 6 and len(change["fixed"]) == 6
    assert change["patch"][0] == {"op": "replace", "path": "/retrieval/top_k", "value": 5}
    assert result.final.retrieval.top_k == 5 and len(result.iterations) == 1 and "target met" in result.stop_reason
    assert hooked == [(1, True)] and evaluator.calls == [0, 1]
    assert (tmp_path / "04_iterations" / "iter_1" / "agent.yaml").is_file() and (tmp_path / "04_iterations" / "iter_1" / "dev" / "scorecard.json").is_file()


def test_a_patch_that_does_not_beat_the_noise_floor_is_reverted_and_the_loop_stops_at_the_plateau(tmp_path, config) -> None:
    cases, evaluator, baseline = seeded_bug_setup(tmp_path, 6)
    result = optimize(agent0=agent(), cases=cases, baseline=baseline, evaluator=evaluator, config=config, out=tmp_path, corpus_dir=tmp_path / "docs",
                      noise_floor=6)  # the same +6 gain is not above a noise floor of 6
    change = json.loads((tmp_path / "04_iterations" / "iter_1" / "change.json").read_text(encoding="utf-8"))
    assert not change["accepted"] and "noise floor" in change["gate_reason"]
    assert result.final.retrieval.top_k == 3 and "plateau" in result.stop_reason and len(result.iterations) == 1  # reverted


def test_the_loop_never_runs_more_than_three_cycles_after_the_baseline(tmp_path, config) -> None:
    cases = {f"c{i}": case(f"c{i}") for i in range(8)}
    evaluator = Scripted(cases, lambda a, c: int(c.id[1:]) < (a.retrieval.top_k - 1) // 2)  # every +2 of top_k fixes one more case
    baseline = evaluator.dev(agent(top_k=1), 0)
    result = optimize(agent0=agent(top_k=1), cases=cases, baseline=baseline, evaluator=evaluator, config=config, out=tmp_path, corpus_dir=tmp_path / "docs",
                      noise_floor=0, max_cycles=10)
    assert len(result.iterations) == 3 and evaluator.calls == [0, 1, 2, 3] and "cycle limit" in result.stop_reason
    assert result.final.retrieval.top_k == 7


def test_the_loop_stops_when_nothing_is_patchable(tmp_path, config) -> None:
    cases = {"c0": case("c0")}
    evaluator = Scripted(cases, lambda a, c: False)
    baseline = evaluator.dev(agent(top_k=10, min_score=0.15), 0)
    a = agent(top_k=10, min_score=0.15).model_copy(deep=True)
    a.retrieval.query_rewrite = True
    a.retrieval.chunking.size_tokens = 800
    a.retrieval.chunking.overlap_tokens = 320
    result = optimize(agent0=a, cases=cases, baseline=baseline, evaluator=evaluator, config=config, out=tmp_path, corpus_dir=tmp_path / "docs", noise_floor=0)
    assert result.iterations == [] and "no patchable failure" in result.stop_reason


def test_a_text_patch_goes_through_the_critic_and_the_guard_and_a_copied_question_is_logged_as_rejected(tmp_path, config) -> None:
    from tests.test_judge import install

    cases = {"c0": case("c0", question="How many days of leave does grade G7 get each year?")}
    a = agent()
    evaluator = Scripted(cases, lambda ag, c: "quote" in ag.prompt_sections.grounding_rules.lower())
    baseline = evaluator.dev(a, 0)
    baseline.scores["c0"] = score(cases["c0"], unsupported=1)  # a hallucination: the category whose first patch rewrites grounding_rules
    drafts = iter(["How many days of leave does grade G7 get each year? Always answer.", "Quote the passage before you state a claim; say so when it has no answer."])
    install(config, tmp_path, SectionDraft=lambda u: {"rationale": "r", "text": next(drafts)})
    result = optimize(agent0=a, cases=cases, baseline=baseline, evaluator=evaluator, config=config, out=tmp_path, corpus_dir=tmp_path / "docs", noise_floor=0)
    rejected = json.loads((tmp_path / "04_iterations" / "iter_1" / "rejected_patches.json").read_text(encoding="utf-8"))
    assert len(rejected) == 1 and rejected[0]["patch"] == "hallucination/grounding_rules" and any("copies" in r for r in rejected[0]["reasons"])
    change = json.loads((tmp_path / "04_iterations" / "iter_1" / "change.json").read_text(encoding="utf-8"))
    assert change["accepted"] and change["patch"][0]["path"] == "/prompt_sections/grounding_rules" and "Quote the passage" in change["patch"][0]["value"]
    assert result.final.prompt_sections.grounding_rules.startswith("Quote the passage")


def test_target_is_met_only_with_a_90_percent_pass_rate_and_no_answered_must_refuse_question() -> None:
    cases = {f"c{i}": case(f"c{i}") for i in range(9)} | {"r": case("r", "out_of_scope", "refused")}
    scores = {cid: score(c, passed=True, status="refused" if c.expected_status == "refused" else "answered") for cid, c in cases.items()}
    assert target_met(DevResult(scores), cases)
    scores["r"] = score(cases["r"], status="answered", passed=True)  # passed, yet answered where it must refuse
    assert not target_met(DevResult(scores), cases)
    scores["r"] = score(cases["r"], status="refused", passed=True)
    scores["c0"] = score(cases["c0"], passed=False)
    scores["c1"] = score(cases["c1"], passed=False)
    assert not target_met(DevResult(scores), cases)


def test_an_accepted_patch_is_rerun_uncached_on_the_fixed_and_regressed_cases_only(tmp_path, config) -> None:
    cases, evaluator, baseline = seeded_bug_setup(tmp_path, 6)
    optimize(agent0=agent(), cases=cases, baseline=baseline, evaluator=evaluator, config=config, out=tmp_path, corpus_dir=tmp_path / "docs", noise_floor=1)
    change = json.loads((tmp_path / "04_iterations" / "iter_1" / "change.json").read_text(encoding="utf-8"))
    assert evaluator.confirmed == [sorted(cases)]  # all six fixed cases were rerun, nothing else
    assert change["status"] == "accepted" and change["confirmation"]["fixes_dropped"] == [] and len(change["fixed"]) == 6


def test_fixes_that_do_not_hold_on_the_rerun_do_not_count_and_the_patch_is_rejected_as_unconfirmed(tmp_path, config) -> None:
    cases = {f"c{i}": case(f"c{i}") for i in range(6)}
    evaluator = Scripted(cases, lambda a, c: a.retrieval.top_k >= 5, confirm_rule=lambda a, c: c.id in ("c0", "c1"))  # 4 of 6 fixes were luck
    baseline = evaluator.dev(agent(), 0)
    result = optimize(agent0=agent(), cases=cases, baseline=baseline, evaluator=evaluator, config=config, out=tmp_path, corpus_dir=tmp_path / "docs",
                      noise_floor=2)  # first run +6 clears 2; only +2 held, which does not
    change = json.loads((tmp_path / "04_iterations" / "iter_1" / "change.json").read_text(encoding="utf-8"))
    assert change["status"] == "rejected: unconfirmed" and not change["accepted"] and "unconfirmed" in change["gate_reason"]
    assert change["confirmation"]["fixes_dropped"] == ["c2", "c3", "c4", "c5"] and change["net_gain"] == 2
    assert result.final.retrieval.top_k == 3  # reverted


def test_a_patch_rejected_by_the_first_run_is_not_rerun(tmp_path, config) -> None:
    cases, evaluator, baseline = seeded_bug_setup(tmp_path, 6)
    optimize(agent0=agent(), cases=cases, baseline=baseline, evaluator=evaluator, config=config, out=tmp_path, corpus_dir=tmp_path / "docs", noise_floor=6)
    assert evaluator.confirmed == []
