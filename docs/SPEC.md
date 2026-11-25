# ForgeAI — The Agent That Builds Agents

**Project Specification · OceanAI Jr. GenAI Engineer Assignment**
Author: Ashriwad Behera · B.Tech CSE (AI/ML), SRM IST

> ForgeAI takes a one-paragraph brief and a folder of documents. Running unattended, it produces a tested agent made of a system prompt, a retrieval config, the tools it needs, an output contract and a generated eval set. It then repairs that agent with targeted fixes until the evals pass or the budget runs out, and it ships the agent with a report that says plainly what still fails.

---

## 0. Design stance: assume the pipeline is lying

Every stage runs on a small local model. Small models drop facts, invent citations, and agree with themselves. So **no LLM output in ForgeAI is trusted without a check that doesn't depend on that same LLM.** The project rests on four rules:

| Principle | What it means in code |
|---|---|
| **Verify, don't trust** | Every citation quote, eval "key fact" and groundability verdict is checked against the source text with string or numeric matching before it is accepted. |
| **Deterministic first, LLM last** | Refusals, citations, numeric facts and tool results are scored by code. The LLM judge only rules on semantic equivalence, and even then it has to point to the exact span it relied on. |
| **Separation of powers** | The eval generator never sees the agent definition. The judge model comes from a different family than the agent model. The optimizer never sees the held-out test split. |
| **Measure the measurer** | Each run plants canary answers (known-correct and known-wrong) to measure the judge, and audits the generated eval set against a hidden fact sheet. The report includes both measurements. |

Section 9 lists every place I expect the factory to lie to itself and the check that catches each one.

---

## 1. How I read the ambiguous parts of the assignment

These are my interpretations, written down so a reviewer can disagree with a specific line instead of discovering it in the code:

1. **"At most 5 clarifying questions"** is a hard cap enforced in code (`assert len(questions) <= 5`). **Zero questions is valid** when the brief and documents already settle everything. The factory only asks about requirement fields it can't resolve on its own.
2. **"Runs unattended"** means `--non-interactive` mode. The interviewer still produces its questions, answers each one with the most defensible default it can infer from the docs, and labels each answer `ASSUMPTION` in the requirements document and the final report. Interactive mode is a CLI dialogue.
3. **"Maximum 3 iterations"** means one baseline evaluation (iteration 0) plus **at most 3 fix-and-re-run cycles**. Never more.
4. **"Wrong refusal"** means refusing a question that should have been answered (over-refusal). Answering a question that should have been refused counts as a **prompt defect** (scope policy failure) if the answer is grounded, and as a **hallucination** if it isn't (see the decision tree in §6.6).
5. **"Evals pass"** means dev-split pass rate ≥ 0.90 **and** zero fabricated answers on must-refuse cases. Both thresholds are configurable.
6. **"Targeted fix, not a full regeneration"**: each fix is a typed JSON-Patch operation on one addressable part of the agent definition, such as a prompt section, a retrieval parameter, or the tool list. Regenerating the whole system prompt is not an allowed operation.

---

## 2. System architecture

```
 brief.txt + docs/
       │
       ▼
┌──────────────────┐   doc_inventory.json, topic_map.json
│ 1. INGEST &      │── loaders: PDF, DOCX, MD, TXT, HTML, CSV
│    PROFILE       │   per-doc summary, structure profile, topic clusters
└────────┬─────────┘
         ▼
┌──────────────────┐   questions.json (≤5), answers.json
│ 2. INTAKE        │── requirements.draft.yaml
│    INTERVIEWER   │
└────────┬─────────┘
         ▼
┌──────────────────┐   coverage.json, gaps.md
│ 3. GROUNDABILITY │── probes every must-answer topic against the corpus
│    GUARD         │   UNSUPPORTED topics → gaps (+ gap question to requester)
└────────┬─────────┘
         ▼  requirements.yaml  (frozen; optional human checkpoint)
   ┌─────┴───────────────────────────────┐
   ▼                                     ▼
┌──────────────────┐            ┌──────────────────┐
│ 4. AGENT         │            │ 5. EVAL          │  ← never sees agent.yaml
│    GENERATOR     │            │    GENERATOR     │
│ prompt sections, │            │ over-generate 2×,│
│ retrieval cfg,   │            │ validate, filter,│
│ tools, contract  │            │ split dev/test   │
└────────┬─────────┘            └────────┬─────────┘
         ▼                               ▼
┌──────────────────────────────────────────────────┐
│ 6. CHAT RUNNER (the same code path users run)    │
└────────┬─────────────────────────────────────────┘
         ▼
┌──────────────────┐  deterministic checks → LLM judge (different family)
│ 7. JUDGE         │  + canaries measure the judge itself
└────────┬─────────┘
         ▼
┌──────────────────┐  classify → patch → re-run → accept/revert (≤3 cycles)
│ 8. CRITIC–       │
│    OPTIMIZER     │
└────────┬─────────┘
         ▼
┌──────────────────┐  final agent.yaml, scorecard.md, report.md
│ 9. REPORTER      │  (test split scored here, with confidence intervals)
└──────────────────┘
```

### 2.1 Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | |
| CLI | Typer + Rich | Clean interview dialogue and scorecards in the terminal |
| Schemas | Pydantic v2 | Every artifact and every LLM output is a validated model |
| LLM runtime | Ollama (local) | Free, offline, has JSON-schema-constrained output |
| Agent / generator model | e.g. `qwen3:8b` (configurable) | Strong tool-calling and JSON at 8B |
| Judge model | e.g. `gemma3:12b` or `llama3.1:8b` (configurable) | **Must be a different model family**, enforced at config load |
| Embeddings | `bge-small-en-v1.5` (sentence-transformers) | Small, fast on CPU |
| Lexical retrieval | `rank_bm25` | Hybrid retrieval with RRF fusion |
| Vector store | NumPy / FAISS-cpu, persisted per run | Few moving parts, fully reproducible |
| Parsing | PyMuPDF, python-docx, pandas, BeautifulSoup | |
| Fuzzy / numeric matching | rapidfuzz, word2number, dateparser | Deterministic fact checks |
| UI (stretch) | Streamlit | Factory UI and chat runner |
| Tests | pytest | |

Exact model tags and digests are pinned in `forgeai.yaml` and recorded in every run's `manifest.json`.

### 2.2 The LLM-call wrapper (handles small-model failures)

Every LLM call goes through one function:

```python
call_structured(role, prompt_id, variables, schema: type[BaseModel],
                model, temperature, max_retries=2) -> BaseModel
```

- Uses Ollama's `format=<json-schema>` constrained decoding.
- Validates with Pydantic. On failure it retries with the validation error fed back to the model. After the retries it falls back to a **deterministic default** (never a silent crash).
- Caches by `hash(model, prompt_id, prompt_version, variables, temperature, seed)`, so reruns are cheap and reproducible.
- Appends one line per call to `trace.jsonl`: role, prompt id and version, tokens in/out, latency, retries, cache hit.
- The report includes the **repair rate** (how often the model needed retries) for each stage. It's a measure of how imperfect the small model actually was.

All prompt templates live in `forgeai/prompts/` as versioned Markdown files, one per task, all committed to the repo.

---

## 3. Requirement 1: Intake interviewer

### 3.1 Document profiling (before any questions)
- Loads every file, chunks it provisionally, and builds `doc_inventory.json`: file type, pages/rows, title, a 3-sentence summary, and a structure profile (prose vs. tabular, average section length, heading depth).
- Clusters chunk embeddings and labels each cluster with an LLM to build `topic_map.json`, the set of topics the corpus **actually covers**. This map is what keeps the interviewer from asking things the documents already answer.

### 3.2 Question selection (≤ 5)
1. Draft requirements from the brief and the topic map. Every field gets a `confidence` (`resolved_from_brief | resolved_from_docs | unresolved`).
2. Generate candidate questions **only for unresolved fields**. Each candidate is tagged with the field it resolves.
3. Rank by priority: scope boundary > must-refuse topics > audience > tone > format. One slot is reserved for the groundability guard (§4) if it finds a gap.
4. Ask up to 5. The cap is enforced in code. Interactive mode asks on the CLI. In `--non-interactive` mode the factory answers its own questions with documented assumptions.

### 3.3 Output: `requirements.yaml` (also rendered as `requirements.md`)

```yaml
agent_name: Shore HR Policy Assistant
purpose: Answer HR policy questions for shore-based staff using company policy documents.
audience: Shore office employees (non-HR), English, mixed seniority
tone: Professional, concise, plain language; no legalese
scope:
  in:  [leave entitlements, travel & expense reimbursement, code of conduct]
  out: [crew/seafarer contracts, individual salary data, legal advice]
must_answer_topics:
  - id: T1
    topic: Annual and sick leave entitlements
    priority: high
    coverage: SUPPORTED          # filled by groundability guard
    evidence: [leave_policy.pdf#c12, leave_policy.pdf#c14]
must_refuse_topics:
  - id: R1
    topic: Individual salary or payslip queries
    reason: Not in documents; personal data
    redirect: "Contact payroll@<company> or the HR portal"
capabilities_needed: [document_qa]        # drives tool selection
gaps: []                                  # filled by guard
assumptions:
  - field: audience
    value: Shore staff only, not seafarers
    source: ASSUMPTION (non-interactive mode)
interview_log: questions.json
```

**Acceptance:** ≤5 questions in every run. Zero questions when nothing is unresolved. Every assumption is labeled. The schema validates.

---

## 4. Requirement 2: Groundability guard

The point is to catch "the brief asks for X, but the documents can't support X" **before** an agent is generated.

### 4.1 Algorithm (per must-answer topic)
1. Generate **3 probe questions** that a real user would ask about the topic.
2. Hybrid-retrieve the top-k chunks for each probe.
3. **Extractive support check:** the model must return a *verbatim* supporting span from the retrieved chunks, or `NONE`. The span is then **verified as a substring** (after whitespace normalization) of a real chunk. If verification fails, the probe counts as unsupported, whatever the model claimed.
3b. **Second check (judge model, different family):** after the span passes the substring and topic-term checks, the judge sees the span and the record it came from and must say whether the span contains the facts needed to answer the question, quoting the words that hold them. The quote is verified as a substring of the span. A probe is *verified* only if both checks pass. `coverage.json` reports how often the two checks disagree. *(Added 2026-11-25, see `docs/DECISIONS.md`.)*
4. Coverage verdict: **SUPPORTED** (≥2/3 probes verified by both checks), **PARTIAL** (1/3), **UNSUPPORTED** (0/3).

### 4.2 Structural gap detection (no LLM needed)
- **Live-data gap:** the brief asks for current or real-time values (e.g. "current stock levels") but the corpus is static, and no tool provides that data.
- **Capability gap:** the brief requires an action no registered tool can perform (e.g. "email the supplier", "raise a ticket").
- **Freshness gap:** the documents carry effective dates older than a configurable threshold. This is a warning, not a block.

### 4.3 What happens to a gap
- **Partly supported topics are split, not refused whole.** A topic that is not fully supported, or is tied to a live-data gap, is split into at most 3 aspects. Supported aspects stay answerable (recorded as `supported_aspects`); only the unsupported or live aspects become gaps. A topic whose evidence is a static snapshot carries `snapshot_as_of`, and the agent must say "as of <date>" and never present it as current. A topic with no supported aspect stays a whole gap.
- **UNSUPPORTED** topics are moved from `must_answer` to `gaps` with a recommended action (`refuse_with_redirect` or `request_documents`).
- **Precedence: brief > corpus evidence > assumptions.** A self-answered assumption may add refusals for things the corpus does not cover, but may not exclude something the brief asks for or the corpus supports; such assumptions are dropped and listed under `assumptions_rejected` with the evidence.
- In interactive mode the reserved question slot is used: *"The documents don't cover payroll queries. Should the agent refuse them, or will you add documents?"*
- In unattended mode the default is `refuse_with_redirect`, labeled as an assumption.
- **Closing the loop:** every gap automatically (a) gets an explicit refusal rule in the agent prompt and (b) becomes refusal cases in the eval set. If the agent ends up answering a gap question, the factory has proven its own guard failed, and the report says so.

### 4.4 Calibrating the retrieval threshold (`min_score`)
The guard already holds two kinds of probe: **in-scope probes** (3 per supported must-answer topic) and **near-miss probes** (adjacent questions the corpus does not cover, kept only if their extractive support check returns `NONE`). For each near-miss probe the guard records the **maximum dense cosine similarity** to any chunk; for each verified in-scope probe it records the similarity of the chunk that **held the verified span** (the best chunk is not enough: a two-part question needs its second chunk to pass the gate too). The query prefix is applied to the query only. `min_score` is placed between the two distributions: the midpoint between the low end (10th percentile) of the in-scope maxima and the high end (90th percentile) of the near-miss maxima. If the distributions overlap (same-subject questions the corpus cannot answer look like answerable ones to a dense embedder), no threshold separates them: the threshold is set 0.05 below the in-scope 10th percentile so answerable questions keep their best chunk, the report flags it as *low confidence*, and unanswerable near-misses are left to the refusal policy. (A first version used the midpoint of the medians; the first live run showed that would gate out relevant chunks, see `docs/DECISIONS.md`.) Probes are grounded: the model sees a few real excerpts so it uses the corpus's own names and codes, and probes that share no content term with the topic are dropped. The result, the separation margin and the optimizer's allowed range (`min_score_bounds`, ±0.10 around the threshold, clamped to [0.05, 0.95]) go in `coverage.json` and into `agent.yaml`. **Only guard probes are used. Eval cases are never used**, because that would leak the test set into the build. The optimizer may move `min_score` only inside the bounds.

**Acceptance:** demo briefs contain deliberately planted gaps (§11). Both must be flagged in `gaps.md`, refused by the final agent, and tested in the eval set.

---

## 5. Requirement 3: Agent generator

### 5.1 The agent definition is structured data, not one big prompt

The system prompt is **compiled** from addressable sections. The generator LLM fills in each section's content; the skeleton is fixed. This is what makes "targeted fix" possible in §6.6: the optimizer can patch `refusal_policy` without touching `grounding_rules`.

```yaml
# runs/<id>/05_final/agent.yaml
schema_version: 1
name: Shore HR Policy Assistant
model: {name: qwen3:8b, temperature: 0.1, seed: 7}

prompt_sections:            # compiled in this order into the system prompt
  identity:        "You are the HR policy assistant for shore staff at ..."
  audience_tone:   "..."
  scope:           "..."
  refusal_policy:  "..."    # includes every must-refuse topic and every gap
  grounding_rules: "Answer only from retrieved context. Quote before you claim..."
  citation_rules:  "..."
  ambiguity_policy:"If the question has two plausible readings, ask one clarifying question..."
  tool_policy:     ""       # empty when no tools are attached
  output_contract: "Respond ONLY with JSON matching the schema below..."
  few_shot:        [...]    # format-only examples, leakage-checked (§5.5)

retrieval:
  chunking: {strategy: heading_recursive, size_tokens: 350, overlap_tokens: 60}
  csv_mode: row_as_record          # used only for tabular files
  hybrid: {dense_weight: 0.6, bm25_weight: 0.4, fusion: rrf}
  top_k: 5
  min_score: 0.25                  # below this → treated as "no evidence"
  query_rewrite: false

tools: []                          # selected from registry, each with a justification
tool_justifications: {}

output_contract: contracts/answer_v1.json
provenance:
  requirements: ../00_intake/requirements.yaml
  generated_by: forgeai 0.x, prompt versions {...}
```

### 5.2 Initial retrieval parameters
These are set by **rules** from the document profile, not guessed by an LLM: short policy clauses get smaller chunks, long technical prose gets larger chunks, CSVs use row-as-record chunks plus a header schema chunk. Rules are more reliable than a small model on numeric choices, and the optimizer tunes the values later anyway.

**`min_score` is the exception: it is calibrated per corpus at build time** (§4.4), not set by a rule. Only guard probes are used for calibration, never eval cases. The bge query-instruction prefix is applied to queries only, never to passages. *(Updated 2026-11-25, see `docs/DECISIONS.md`.)*

### 5.3 Tool selection (attach only what the brief needs)

**Registry (4 tools):**

| Tool | Capability tag | Example need signal |
|---|---|---|
| `csv_lookup(file, filters, columns)` | `structured_lookup` | Tabular docs plus "find / filter / compare items" |
| `calculator(expression)` (safe AST eval, no `eval`) | `arithmetic` | Totals, discounts, conversions in must-answer topics |
| `current_date()` | `temporal` | "How many days until…", lead-time-to-date |
| `weather(lat, lon, date)` via open-meteo, no key | `live_weather` | Port or voyage weather explicitly in scope |

**Selection procedure:**
1. Requirements declare `capabilities_needed`, derived from the brief and the must-answer topics.
2. The LLM proposes tools, and **each proposal must cite a requirement ID** it serves.
3. A validator drops any tool with no valid requirement link, and checks the reverse: is there a must-answer topic that would fail without this tool?
4. Every attached tool gets ≥1 eval case that exercises it. Every *unattached* tool gets a negative case (e.g. HR agent asked "What's the weather in Singapore?" must refuse, not improvise).

**Tool-selection test suite** (`tests/tool_selection/`): 8 mini-briefs with expected tool sets. They include an HR brief (expects **no** tools), a port-call planner (expects `weather` + `current_date`), and a procurement brief (expects `csv_lookup` + `calculator` + `current_date`, **not** `weather`). This proves the selector works in both directions, including on the weather tool, which neither demo should use.

### 5.4 Output contract (`contracts/answer_v1.json`)

```json
{
  "status": "answered | refused | needs_clarification",
  "answer": "string",
  "citations": [{"doc": "leave_policy.pdf", "chunk_id": "c12", "quote": "verbatim span"}],
  "tool_calls": [{"tool": "calculator", "args": {...}, "result": "..."}],
  "clarifying_question": "string | null"
}
```

Putting `status` in the contract makes refusal **machine-readable**, so refusal scoring is fully deterministic. Requiring a `quote` makes citations **verifiable**: the quote must appear in the cited chunk.

### 5.5 Few-shot leakage check
Few-shot examples demonstrate format only. They are rejected if their embedding similarity to any eval question is above 0.80, so the agent can't be shown the answers to its own test.

---

## 6. Requirements 4–6: Evals, judge, critic–optimizer

### 6.1 Eval generator (Requirement 4)

**Independence:** the eval generator sees only `requirements.yaml` and the documents, **never** `agent.yaml`. This is enforced by module boundaries and asserted in a unit test.

**Target composition** (hard floor ≥ 20; target 32; over-generate about 2× then filter):

| Type | Target | How it's generated | How it's validated before acceptance |
|---|---|---|---|
| Answerable, single-doc | 12 | Stratified chunk sampling across docs and must-answer topics → Q + key facts + expected source | **Fact anchoring:** each key fact must match the seed span (normalized string / numeric / date match). **Paraphrase check:** the question can't be a lexical copy of the chunk (n-gram overlap < 0.5). **Independent answerability:** the judge model answers from the chunk alone and must recover the facts. |
| Edge: multi-doc | 5 | Pair semantically related chunks from *different* docs → a question that needs both | Each document must contribute ≥1 anchored key fact |
| Edge: ambiguous | 4 | Under-specified questions with two valid readings | Each case lists its interpretations; accepted statuses are `needs_clarification`, or `answered` covering all readings |
| Out-of-scope (must refuse) | 7 | Must-refuse topics + guard **gaps** + **near-miss** hard negatives (adjacent topics the corpus doesn't cover) | Max retrieval score across the corpus is below threshold **and** the guard's extractive probe fails |
| Tool-required | 4 | Question templates filled from CSV rows | **Expected answer is computed by code** from the CSV (lookup + arithmetic), never by an LLM |

If no tools are attached, the tool-required slots go to answerable and multi-doc cases, and the negative tool cases (§5.3) still go into out-of-scope.

**Dedup:** embedding similarity > 0.9 → keep one.
**Split:** stratified by type, about **70% dev / 30% test**. The optimizer only ever sees dev. The test split is reported, never optimized on.
**Committed artifacts:** `eval_set.jsonl`, `rejected_cases.jsonl` (with rejection reason), `split.json`, and copies of the exact generation prompts and versions used.

**Seed spans, not chunk ids.** An eval case records where its facts live as `{doc, char_start, char_end, span_text}`, in character offsets of the document's *canonical text* (the loader's blocks joined by blank lines; it does not depend on chunking parameters). `span_text` must equal the text at those offsets (verified). A **gold-chunk hit** is any retrieved chunk whose character range overlaps the seed range. Chunk ids and exact-text matching are never used for this, because the optimizer changes chunk size and a smaller chunk can split the span across two chunks. *(Updated 2026-11-25, see `docs/DECISIONS.md`.)*

**Eval case schema:**
```json
{
  "id": "hr-A07", "type": "answerable", "split": "dev",
  "question": "If I joined in October, how much annual leave do I get this year?",
  "expected_status": "answered",
  "expected_sources": ["leave_policy.pdf"],
  "key_facts": [{"text": "pro-rata", "kind": "phrase"}, {"text": "1.75 days per month", "kind": "number"}],
  "seed_spans": [{"doc": "leave_policy.pdf", "char_start": 4120, "char_end": 4388, "span_text": "verbatim text at those offsets"}],
  "provenance": {"prompt_id": "evals/answerable", "prompt_version": 3},
  "validation": {"fact_anchoring": true, "paraphrase_overlap": 0.31, "independent_answerable": true}
}
```

**Eval-set audit (measuring the measurer):** for the synthesized demo corpora, the synthesis step writes a hidden `fact_sheet.json` of planted facts. The eval generator never reads it. Afterwards the audit reports (a) what fraction of planted facts the generated evals cover and (b) what fraction of generated key facts agree with the fact sheet. That quantifies the eval generator's own quality.

### 6.2 Judge (Requirement 5): layered scoring

**Layer 1: deterministic checks (always run, authoritative where applicable)**

| Check | Method |
|---|---|
| Contract valid | JSON schema validation |
| Refusal correct | `status` vs `expected_status` |
| Citation valid | every `quote` is a substring of its cited chunk (normalized) |
| Source correct | ≥1 citation from `expected_sources` (multi-doc: all of them) |
| Key-fact recall | numbers / dates / currency normalized and compared exactly; phrases via rapidfuzz token-set ≥ 85 |
| Tool correctness | required tool called; args valid; final numeric value within tolerance of the code-computed answer |
| Retrieval hit | did any retrieved chunk's character range overlap a seed span (logged for failure diagnosis) |

**Layer 2: LLM judge (only where layer 1 can't decide)**
- **Claim support:** decompose the answer into atomic claims, then rate each claim `supported / unsupported` against *retrieved context only*. Any unsupported claim means ungrounded.
- **Fact equivalence:** if a key fact fails deterministic matching, the judge may upgrade it to pass **only by quoting the exact span in the answer** that expresses the fact. The span is verified as a substring. No quote, no upgrade.
- **Ambiguity handling:** checks whether the answer covers each listed interpretation.

**Defenses against known judge failure modes:**

| Failure mode | Defense |
|---|---|
| **Position bias** | Pointwise grading against a reference, never "which is better" during evals. In the pairwise prompt tournament (stretch), every pair is judged in **both orders**; if the two verdicts disagree, it's a tie. |
| **Self-preference** | Judge family ≠ agent/generator family, enforced at config load (`ConfigError` if equal). Judge inputs are blinded: no iteration number, no candidate name, no model name. |
| **Verbosity bias** | Claim-level grading with length-independent scoring. Extra unsupported claims are *penalized*, not rewarded. Fact matching is deterministic. A canary pair (concise-correct vs padded-correct) must score the same. |
| **Leniency / sycophancy** | Rubric demands a quoted span for every "supported" verdict. Temperature 0. |

**Judge canaries (run every time):** from 10 gold answers, code generates perturbed versions: swapped number, negated policy, wrong document cited, fabricated extra clause, padded but correct. Every run reports the judge's **canary detection accuracy**. If it falls below 90%, LLM-judge verdicts are flagged *low confidence* in the report and the headline score is shown both with and without layer-2 upgrades.

**Agreement log:** cases where layer 1 and layer 2 disagree are written to `judge_disagreements.jsonl`, and a sample is shown in the report.

### 6.3 Per-case pass rule

| Case type | Pass if |
|---|---|
| Answerable | `answered` ∧ contract valid ∧ citations valid ∧ expected source cited ∧ all key facts matched ∧ no unsupported claims |
| Multi-doc | as above, with citations covering every expected document |
| Ambiguous | `needs_clarification`, or `answered` covering all interpretations with valid citations |
| Out-of-scope | `refused` ∧ no factual claims beyond the redirect |
| Tool-required | correct tool called ∧ value within tolerance ∧ grounded explanation |

### 6.4 Metrics (per iteration, dev and test separately)
Pass rate (with a **Wilson 95% confidence interval**), groundedness rate, key-fact recall, refusal precision / recall, citation validity, retrieval hit@k, tool-call accuracy, mean latency, tokens.

### 6.5 Failure classification (Requirement 6): a deterministic decision tree

```
contract invalid ......................................... PROMPT DEFECT (format)
expected=refused, got=answered:
    any unsupported claim ................................ HALLUCINATION
    else ................................................. PROMPT DEFECT (scope)
expected=answered, got=refused:
    gold chunk in top-k .................................. WRONG REFUSAL
    else ................................................. RETRIEVAL MISS
expected=answered, got=answered but failed:
    gold chunk not in top-k .............................. RETRIEVAL MISS
    unsupported claims present ........................... HALLUCINATION
    supported but facts missing / wrong citation ......... PROMPT DEFECT
ambiguous, answered one reading only ..................... PROMPT DEFECT (ambiguity)
```

The classifier is code. An LLM critic is used **only** to draft the *text* of a prompt-section patch once the category and section are already decided.

### 6.6 Optimizer: targeted patches with an acceptance gate

**Patch library** (each one is a JSON-Patch on `agent.yaml`, with bounds; `min_score` may only move inside `retrieval.min_score_bounds`, §4.4):

| Category | Allowed patches |
|---|---|
| Retrieval miss | `top_k` +2 (≤10) · chunk size ±30% · overlap +20 · shift hybrid weight toward BM25 (when misses involve exact terms or part numbers) · enable `query_rewrite` · lower `min_score` |
| Prompt defect | rewrite **one** named section (`scope`, `citation_rules`, `ambiguity_policy`, `output_contract`, `tool_policy`) · add one format-only few-shot |
| Hallucination | strengthen `grounding_rules` (quote-before-claim) · raise `min_score` · lower temperature · add an explicit gap refusal |
| Wrong refusal | relax `refusal_policy` wording for the affected in-scope topic · add the topic to `scope` with an evidence pointer · lower `min_score` |

**Loop (at most 3 cycles after the baseline):**
1. Classify every dev failure.
2. Pick the category with the most failures. If tied, the order is hallucination > wrong refusal > retrieval miss > prompt defect (the order of harm to users).
3. Apply **one patch set** aimed at that category's cases.
4. Re-run the full dev split.
5. **Acceptance gate:** keep the patch only if the dev score rises **and** no more than one previously passing case regresses. Otherwise **revert** and log the patch as *rejected*. Rejected patches stay in the log; they are evidence too.
6. Stop early if: target met, budget exhausted, no failures in a patchable category, or the last cycle produced no accepted patch (plateau).

**Change log entry** (`04_iterations/iter_N/change.json`):
```json
{
  "iteration": 2,
  "category": "retrieval_miss",
  "target_cases": ["proc-A03", "proc-M02", "proc-A11"],
  "patch": [{"op": "replace", "path": "/retrieval/hybrid/bm25_weight", "value": 0.6}],
  "rationale": "3/4 misses involve exact part numbers (e.g. 'MAN-B&W-1123'); dense retrieval under-weights them.",
  "score_before": {"dev": 0.68, "test": 0.70},
  "score_after":  {"dev": 0.77, "test": 0.70},
  "fixed": ["proc-A03", "proc-M02"], "regressed": [], "still_failing": ["proc-A11"],
  "accepted": true
}
```

(The numbers above show the format only. Real values come from the committed runs.)

---

## 7. Requirement 7: Honest reporting

`05_final/report.md` is generated from run data. Every number in it traces back to a file in the run folder.

1. **Headline:** final dev and **test** pass rate with 95% CI and case counts (e.g. "test: 7/10, CI 40–89%"). Small eval sets get wide intervals, and the report says so.
2. **Per-iteration scorecard:** a table of all metrics for iterations 0 to N, with accepted and rejected patches.
3. **Failure breakdown:** a category × iteration matrix showing what each fix moved.
4. **What still fails and why:** every failing test and dev case with its category, the diagnosis, and what fix would be needed. That fix may be outside the factory's patch library, e.g. "the document itself contradicts itself on p.4 vs p.9".
5. **Groundability gaps** and how the agent handles them.
6. **Tool selection:** attached tools with their requirement links, plus rejected tools and why.
7. **Trust in the measurements:** judge canary accuracy, layer-1/layer-2 disagreement rate, eval-set rejection rate by reason, fact-sheet audit (demo runs), LLM repair rate per stage.
8. **Cost:** tokens, wall-clock time, and LLM calls per stage.
9. **Claims this report does not make:** for example, "the test split is 10 cases, so a single case moves the score by 10 points", and "eval questions were generated from the same corpus the agent retrieves from, so this measures grounding, not real-world question coverage".

---

## 8. Requirement 8: Runnable product (generic chat runner)

```bash
forgeai chat runs/hr-shore/05_final/agent.yaml
```

- Loads **any** produced definition. It knows nothing about HR or procurement.
- Builds the retrieval index from the referenced documents, or loads it from a cache keyed by `hash(docs + chunking params)`.
- Agent loop: retrieve → compile prompt → model → (≤3 tool calls via a JSON tool-call protocol) → validate the contract → one repair retry → render the answer with sources.
- CLI commands: `/sources` (show retrieved chunks), `/debug` (raw JSON + tool trace), `/reset`.
- **The eval harness imports this same runner.** Evals test exactly what a user runs; there is no separate evaluation path.

---

## 9. Where the factory can lie to itself, and the check that catches it

| How ForgeAI could fool itself | Check that catches it |
|---|---|
| Eval "expected facts" not actually in the documents | Fact anchoring against the seed chunk + fact-sheet audit |
| Eval questions copied from the text, so retrieval looks perfect | Paraphrase-overlap filter; retrieval hit@k reported |
| Judge passes wrong answers | Canary perturbations every run, with accuracy reported |
| Judge favors its own family's outputs | Different-family judge, enforced at config load; blinded inputs |
| Agent cites a source but invents the quote | Verbatim quote verification against the cited chunk |
| Optimizer overfits the dev set | Held-out test split never seen by the optimizer, reported separately |
| A "fix" silently breaks passing cases | Acceptance gate with a regression limit, plus revert |
| Guard misses a gap | Gap and near-miss refusal cases; an answered gap is reported as a guard failure |
| Tool over-attachment (HR agent with weather) | Requirement-linked justification + necessity check + negative cases + selector test suite |
| Eval generator tailors tests to the agent | Module isolation: it never receives `agent.yaml` (unit-tested) |
| Few-shot examples leak the test answers | Embedding-similarity leakage check |
| Small model emits malformed JSON | Schema-constrained decoding + repair retry + deterministic fallback; repair rate reported |
| Tool-case "expected answers" are themselves wrong | Computed by code from the CSV, never by an LLM |

---

## 10. Repository layout

```
forgeai/
├── forgeai/
│   ├── cli.py                 # build | chat | eval | regress | ui
│   ├── config.py              # model pins, thresholds, family check
│   ├── llm.py                 # call_structured, cache, trace
│   ├── ingest/                # loaders, profiler, topic_map
│   ├── intake/                # interviewer, requirements schema
│   ├── guard/                 # probes, extractive verification, structural gaps
│   ├── generator/             # prompt sections, retrieval rules, tool selector
│   ├── evals/                 # generators per case type, validators, split, audit
│   ├── judge/                 # deterministic checks, LLM judge, canaries
│   ├── optimizer/             # classifier, patch library, acceptance gate
│   ├── runner/                # generic chat runner (shared with evals)
│   ├── tools/                 # csv_lookup, calculator, current_date, weather
│   ├── audit/                 # eval-set audit vs hidden fact sheets (the ONLY reader of fact_sheet.json;
│   │                          #   forgeai/evals/ never reads fact sheets; enforced by a test)
│   ├── report/                # scorecard + report renderers
│   └── prompts/               # every prompt, versioned, committed
├── contracts/answer_v1.json
├── data/
│   ├── synthesis/             # synthesis prompts + fact sheets (committed)
│   ├── hr_shore/              # brief.txt + docs/
│   └── spares_procurement/    # brief.txt + docs/
├── runs/                      # complete artifacts of both demo runs (committed)
├── tests/                     # unit tests, tool-selection suite, isolation tests
├── app/streamlit_app.py       # stretch
├── forgeai.yaml
└── README.md                  # quickstart + review-day runbook
```

**Run artifact layout** (identical for every run):
```
runs/<run_id>/
├── manifest.json                # models + digests, prompt versions, git SHA, timings
├── trace.jsonl                  # every LLM call
├── 00_intake/   brief.txt, doc_inventory.json, topic_map.json, questions.json,
│                answers.json, requirements.yaml, requirements.md
├── 01_guard/    coverage.json, gaps.md
├── 02_agent/    agent.v0.yaml, tool_selection.json, prompt_candidates/ (stretch)
├── 03_evals/    eval_set.jsonl, rejected_cases.jsonl, split.json, prompts_used/, audit.json
├── 04_iterations/iter_0..3/  results.jsonl, failures.json, change.json, scorecard.json
└── 05_final/    agent.yaml, scorecard.md, report.md, judge_disagreements.jsonl
```

---

## 11. Requirement 9: Two full demonstrations

The two demos are deliberately different on every axis that matters: prose vs. tabular data, zero tools vs. three tools, policy Q&A vs. lookup-and-compute. Both are set in **shore-side maritime operations** to match OceanAI's world. All companies and suppliers are fictional, and all documents are LLM-synthesized with the synthesis prompts committed.

### Demo A: Shore HR Policy Assistant
- **Brief:** *"I need an assistant that answers HR policy questions for our shore staff: leave, travel claims, conduct rules. Also help people with salary slip questions."*
- **Documents (3 PDFs):** Leave & Holiday Policy, Travel & Expense Reimbursement Policy, Code of Conduct & IT Acceptable Use. Each is 4–6 pages with numbered clauses, plus some cross-references (e.g. leave during business travel).
- **Planted gap:** salary slips. No payroll document exists, so the guard must flag it and the agent must refuse with a redirect.
- **Planted difficulty:** one cross-document rule (travel days that fall on public holidays) and one ambiguous term ("leave" meaning annual vs. sick).
- **Expected tools:** **none.** If the factory attaches anything, that is a defect and the run reports it.

### Demo B: Ship-Spares Procurement Assistant
- **Brief:** *"Our technical superintendents need help finding ship spares, comparing suppliers, estimating order cost with discounts, and knowing when parts will arrive. It should also tell them what's in stock right now at each port."*
- **Documents:** `spares_catalogue.csv` (~300 rows: part_no, description, equipment, maker, supplier_id, unit_price_usd, moq, lead_time_days, stock_port, snapshot_date: the date the static export was taken, so answers can say "as of <date>") plus 3 supplier terms PDFs (volume discount tiers, Incoterms, warranty, expedite surcharges).
- **Planted gap:** "in stock right now". The catalogue is a static snapshot, so this is a live-data gap that must be flagged.
- **Planted difficulty:** questions that combine a CSV price with a PDF discount tier (multi-doc + calculator) and lead time + today's date (`current_date`).
- **Expected tools:** `csv_lookup`, `calculator`, `current_date`, and **not** `weather`.

### Committed for both runs
Requirements, guard coverage and gaps, the agent definition at each version, the generated eval set and its rejections, the generation prompts, iteration logs and change files, scorecards, the final report, the trace and the manifest.

### Blind rehearsal (preparing for the review day)
To build for the unseen brief rather than for my two demos, I'll run ForgeAI on **2 additional briefs I don't tune against** (e.g. a public-domain safety manual, and a third document set prepared by a classmate). These runs are committed under `runs/rehearsal/` along with whatever broke and what I changed in response.

---

## 12. Review-day runbook (unseen brief, unattended)

```bash
ollama pull <agent-model> && ollama pull <judge-model>     # pre-pulled, offline
forgeai build --brief reviewer/brief.txt --docs reviewer/docs/ \
              --out runs/review --non-interactive --max-iter 3 --budget-tokens 400000
forgeai chat runs/review/05_final/agent.yaml
```

**Robustness guarantees:**
- An unparseable file is logged in `doc_inventory.json` as `skipped` with the reason; the run continues.
- An empty or tiny corpus triggers a guard warning, and the eval floor is enforced honestly: if fewer than 20 valid cases can be generated, the report states the real count instead of padding with weak cases.
- Every stage checkpoints to disk; `--resume` continues from the last completed stage.
- A progress bar per stage, with the expected runtime printed up front.

---

## 13. Stretch goals (only after all nine core requirements pass)

In priority order, cheapest and highest-signal first:

1. **Human checkpoint:** `--approve` opens `requirements.yaml` in `$EDITOR` (or a Streamlit form). The edited file is re-validated and its diff logged before generation continues.
2. **Budget enforcement:** `trace.jsonl` already counts tokens and latency. Early stopping triggers when an iteration's score gain per 10k tokens falls below a threshold. The report includes a cost-vs-score curve.
3. **Prompt tournament:** the generator proposes 3 system-prompt variants (strict-grounding, conversational, checklist-style). Each runs on the dev split; the top two face off pairwise in **both orders**; the winner proceeds, and the comparison table goes in the report.
4. **Regression mode:** `forgeai regress --run runs/hr-shore --docs data/hr_shore/docs_v2/` regenerates the agent and re-runs the **old** eval set. It reports which cases broke and whether the cause was a document change (seed chunk changed) or an agent change.
5. **UI + hosting:** a Streamlit app with a factory tab (live stage progress, scorecards) and a chat tab. Hosted on Hugging Face Spaces: the free CPU tier runs the chat runner with a small model and lets visitors browse both committed demo runs.
6. **Factory vs. human baseline:** hand-write the HR agent myself and evaluate both on the same test split. The report shows, case by case, where the factory beats me and where it loses.

---

## 14. Build plan (phase by phase, each with an exit test)

| Phase | Build | Exit test |
|---|---|---|
| P0 | Config, `call_structured`, cache, trace, model-family check | Malformed-JSON fixture repaired; same-family config rejected |
| P1 | Ingest + profiler + topic map | All file types parse; corrupt file skipped without a crash |
| P2 | Runner + tools + output contract | Hand-written `agent.yaml` answers with verified citations |
| P3 | Intake + guard | Demo A flags payroll; Demo B flags live stock; ≤5 questions |
| P4 | Agent generator + tool selector | Tool-selection suite 8/8; HR gets no tools |
| P5 | Eval generator + validators + split + audit | ≥20 valid cases per demo; isolation test passes |
| P6 | Judge layers + canaries | Canary accuracy reported; position/verbosity canaries behave |
| P7 | Classifier + patch library + acceptance gate | A seeded retrieval bug is classified and fixed in 1 cycle |
| P8 | Reporter | Every number in `report.md` traces to a run file |
| P9 | Both demo runs + blind rehearsal, committed | Full artifact tree present for every run |
| P10 | Stretch goals, in the §13 order | — |

---

## 15. Requirement traceability

| # | Requirement | Module | Committed evidence |
|---|---|---|---|
| 1 | Intake interviewer (≤5 Qs, structured requirements) | `intake/` | `questions.json`, `requirements.yaml/.md` |
| 2 | Groundability guard | `guard/` | `coverage.json`, `gaps.md`, gap refusal cases |
| 3 | Agent generator (prompt, retrieval, tools, contract) | `generator/` | `agent.v0.yaml`, `tool_selection.json` |
| 4 | Eval generator (≥20, generated, prompts committed) | `evals/`, `prompts/evals/` | `eval_set.jsonl`, `rejected_cases.jsonl`, `prompts_used/` |
| 5 | Judge with bias defenses | `judge/` | per-case verdicts, canary accuracy, disagreement log |
| 6 | Critic–optimizer (classify → targeted fix → ≤3 cycles, logged) | `optimizer/` | `iter_*/change.json`, `scorecard.json` |
| 7 | Honest reporting | `report/` | `report.md`, `scorecard.md` |
| 8 | Runnable generic chat runner | `runner/`, `cli.py chat` | Works on both demo agents and the rehearsal agents |
| 9 | Two full demonstrations | `data/`, `runs/` | Complete artifact trees for HR and procurement |