# ForgeAI: conventions for Claude sessions

**Source of truth: `docs/SPEC.md`.** Read it before changing anything. Build order is §14 (P0 to P10).
Do ONE phase at a time. When a phase's exit test passes: commit, push, show the results, and
**wait for the user's go-ahead** before starting the next phase.

## Rules that never change
1. **Nothing in `forgeai/` is specific to a demo.** No HR or procurement logic. Demo specifics live
   only in `data/`. The factory must work on an unseen brief and folder.
2. **Every LLM call goes through `call_structured()` in `forgeai/llm.py`.** No direct Ollama calls
   elsewhere (a test enforces this). Embeddings use sentence-transformers, not Ollama.
3. **Every prompt is a versioned `.md` file in `forgeai/prompts/`** (frontmatter `version: N`,
   `# system` / `# user` sections, `{{var}}` placeholders). No prompt strings in Python.
4. **Deterministic checks before the LLM judge, always.** "Verified as a substring" and "computed by
   code" in the spec are mandatory.
5. **Separation of powers**, each with a unit test: the eval generator never imports or receives
   `agent.yaml`; the judge's model family differs from the agent's (`ConfigError`); the optimizer
   never sees the test split.
6. **Never fake results.** Never hand-edit anything under `runs/`, never hardcode scores, never
   mock the LLM in demo runs. Fakes exist only in unit tests of failure handling. If something
   fails, say so. An honest 78% beats a fake 100%.
7. **Hard limits enforced in code:** <=5 clarifying questions, <=3 fix cycles after the baseline,
   >=20 eval cases (or state the real count honestly). Values live in `Limits` in `config.py`.
8. **pytest** for everything. Each phase's exit test is an automated test or reproducible command.
9. Readable, typed code. Clarity over cleverness.

## Environment and decisions
- Windows 11, Python 3.11, venv at `.venv` (`.venv\Scripts\python.exe`). Dependencies pinned in
  `pyproject.toml`. Repo lives at `C:\dev\forgeai` (not OneDrive); GitHub repo `forgeai` is PRIVATE
  until submission. Commit at the end of each phase and push.
- Hardware: RTX 4060 Laptop 8 GB VRAM, 15.7 GB RAM. `OLLAMA_MAX_LOADED_MODELS=1` is set (user env)
  so two models never half-load and spill to CPU.
- Models: agent/generator `qwen3:8b`. Judge is a config value chosen **by measured canary
  accuracy** (spec §6.2) between `gemma3:4b` and `llama3.1:8b`; record the comparison for the
  final report. Both are pulled locally.
- **Batch by stage** to limit model swaps on 8 GB VRAM: generate all agent answers for an
  iteration first, then judge them all. `trace.jsonl` logs `swap` and `load_ms` per call.
- `call_structured` sends `think: false` (Qwen3 reasoning conflicts with constrained JSON). Ollama
  seed determinism is not guaranteed on GPU; the response cache is what makes reruns reproducible.
- A dead Ollama raises `LLMUnavailableError`; the deterministic fallback is only for *invalid
  output* after retries, and is always recorded in the trace (`fallback_used`).
- **Do NOT gitignore `runs/`**: demo outputs are required deliverables. Ignored: `.venv`,
  `__pycache__`, `.env`, `.cache/` (embedding/index and LLM response caches).
- Tests that need live Ollama are marked `live` and skipped loudly when Ollama or the models are
  missing; they are never silently passed.
- Tests: `.venv\Scripts\python.exe -m pytest`.

## Decisions log
- Retrieval uses plain NumPy cosine search (FAISS adds nothing at this scale).
- Test-split scores are computed in a separate step the optimizer cannot read (spec §6.6 vs §6.1).
- Failure tree implemented exactly as spec §6.5; "expected answered, got refused, gold chunk not in
  top-k" is a RETRIEVAL MISS.
