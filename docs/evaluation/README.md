# Fixture evaluation results

This folder stores benchmark runs against local JSON traces in `test cases/`.

## Latest run (phased, honest LLM — 2026-08-13)

| Metric | Value |
|--------|-------|
| Fixtures | 60 |
| Passed | **53** |
| Failed | 7 |
| **Accuracy** | **88.3%** |
| LLM model | `gemini-3.5-flash-lite` |
| Run time | 251.8s |
| API calls | **63** (rate-limited 15/min) |
| Mock-resolved (no API) | **36** fixtures |
| Live LLM | **24** fixtures |

**Method:** Phased runner — infinite loop / token-only traces without API; hallucination and all LLM-required suites always use live Gemini; remaining fixtures mock-probed only when tier 1–2 resolves without LLM.

### By category

| Category | Total | Passed | Accuracy |
|----------|-------|--------|----------|
| context_loss | 6 | 6 | 100% |
| infinite_loop | 9 | 9 | 100% |
| HALLUCINATION | 9 | 9 | **100% (live LLM)** |
| premature_termination | 6 | 6 | 100% |
| classifier_e2e | 5 | 4 | 80% |
| Goal Statisfiction | 11 | 9 | 81.8% |
| TOOL_USE FAILURE | 14 | 10 | 71.4% |

See **`SUMMARY.md`** for failures and **`fixture_results.json`** for per-fixture detail (`run_mode`: `no_llm`, `mock_pass`, or `llm`).

## Re-run (phased — respects API rate limits)

Mocks first only for detectors that can resolve **without** calling the LLM on that trace (schema checks, embedding ranking, constraint validation, etc.). Fixtures that **always need live LLM** are never mock-skipped:

- All `HALLUCINATION/` (blend requires real judge)
- All `classifier_e2e/`
- Goal misinterpretation + aggregator edge cases
- Any fixture with `llm_fallback`, `llm_primary`, or `llm_uncertain` in the name

Hallucination detector: grounding-only escalation runs **only when the LLM API fails**, not when the judge returns uncertain/low confidence.

```powershell
$env:LLM_PROVIDER = "gemini"
$env:LLM_API_KEY = "your-key"
$env:LLM_MODEL = "gemini-3.5-flash-lite"
C:\Users\Admin\Desktop\agent-diagnostican\venv\Scripts\python.exe scripts/run_phased_fixture_evaluation.py --initial-used 0 --first-batch 3 --batch-size 15
```

Options: `--initial-used 11` if you already consumed quota this minute; `--max-per-minute 15`.

## Re-run (full — all fixtures hit API path where detectors call LLM)

### Which detectors use the LLM judge?

| Detector | LLM judge | LLM methods | Notes |
|----------|-----------|-------------|-------|
| **Token exhaustion** | No | — | Fully rule-based (fuzzy error match + token ratio heuristics) |
| **Infinite loop** | No* | — | Accepts `llm_judge` in constructor but **never calls it**; uses rules + embeddings for thought similarity |
| **Tool use** | Yes | `evaluate_wrong_tool`, `evaluate_parameter_structure`, `evaluate_parameter_values` | Tier 3 fallback after schema / embedding / thought checks |
| **Goal failure** | Yes | `evaluate_goal_alignment` | Stage 2 core engine for task misinterpretation; constraint violations can fire without LLM |
| **Hallucination** | Yes | `evaluate_hallucination` | Always invoked; grounding-only escalation **only on API failure** |
| **Context loss** | Yes | `evaluate_context_loss` | Tier 3 fallback after grounding drop and thought-contradiction checks |
| **Premature termination** | Yes | `evaluate_premature_termination` | Tier 4 fallback when output-similarity and input-progress signals are inconclusive |

\*Infinite loop still accepts `MockLLMJudge` for API compatibility; production diagnosis does not depend on it.

### Self-capability (works without a live LLM)

| Level | Detectors | What works without API |
|-------|-----------|------------------------|
| **Fully self-capable** | Token exhaustion | All subtypes via explicit error text or token-ratio heuristics |
| **Self-capable in practice** | Infinite loop | All 9 fixture subtypes via loop detection, input/error analysis, and optional embeddings |
| **Partial (tier 1–2)** | Tool use | Wrong-tool via thought mismatch and embedding ranking; invalid parameters via schema inference |
| **Partial (stage 1)** | Goal failure | Constraint violation path (numeric, categorical, structural) without LLM |
| **Partial (API outage only)** | Hallucination | Grounding-only escalation if LLM API fails |
| **Partial (checks 1–2)** | Context loss | Grounding drop and thought-vs-prior-context contradiction |
| **Partial (steps 1–3)** | Premature termination | Low output similarity, incomplete inputs, failed-step ratio |

For production accuracy on ambiguous cases, inject a real judge:

```python
from agent_diagnostician.analysis.llm import create_llm_judge_from_env
from agent_diagnostician.classifier import Classifier

classifier = Classifier(llm_judge=create_llm_judge_from_env())
```

### Fixture coverage by detector

| Detector | Dedicated fixture folder | Trace fixtures | Sidecars / notes |
|----------|-------------------------|----------------|------------------|
| Tool use | `test cases/TOOL_USE FAILURE/` | 14 | 3 `available_tools__*.json` sidecars |
| Goal failure | `test cases/Goal Statisfiction/` | 11 | — |
| Hallucination | `test cases/HALLUCINATION/` | 9 | — |
| Context loss | `test cases/context_loss/` | 6 | — |
| Premature termination | `test cases/premature_termination/` | 6 | — |
| Infinite loop | `test cases/infinite_loop_and_e2e_traces/infinite_loop/` | 9 | — |
| Token exhaustion | **None** | **0 dedicated** | 2 classifier e2e traces only |
| Classifier (multi-detector) | `test cases/infinite_loop_and_e2e_traces/classifier_e2e/` | 5 | Tiebreaker, selective detectors, multi-fire |

**Total:** 60 trace fixtures + 3 tool-list sidecars.

### Fixtures still to write

**Token exhaustion** has no dedicated folder. Recommended cases:

| Case | Subtype | Purpose |
|------|---------|---------|
| Part A — explicit `error_message` match | `token_exhaustion` | `"Token limit exceeded"` / `"Context window exceeded"` phrases |
| Part B — ratio above threshold (known model) | `token_exhaustion` | `total_tokens` near model limit from `available_tools` metadata |
| Part B — ratio below threshold | `no_token_exhaustion` | High usage but under limit |
| Part B — floor heuristic (unknown model) | `token_exhaustion` (low confidence) | `total_tokens > 100k` + failed status |
| No failure — healthy run | `no_token_exhaustion` | Normal token counts, success status |
| Insufficient evidence | `insufficient_evidence` | Missing token fields, ambiguous signals |

Other detectors have baseline coverage; gaps are mostly **LLM-fallback edge cases** (see failures in `SUMMARY.md` for goal failure, tool use, and premature termination).

## Re-run

**Phased (recommended for free-tier limits):**

```powershell
$env:LLM_PROVIDER = "gemini"
$env:LLM_API_KEY = "your-key"
$env:LLM_MODEL = "gemini-3.5-flash-lite"
C:\Users\Admin\Desktop\agent-diagnostican\venv\Scripts\python.exe scripts/run_phased_fixture_evaluation.py --initial-used 0 --first-batch 3 --batch-size 15
```

**Full classifier (all 60 fixtures):**

```powershell
$env:LLM_PROVIDER = "gemini"
$env:LLM_API_KEY = "your-key"
$env:LLM_MODEL = "gemini-3.5-flash-lite"
C:\Users\Admin\Desktop\agent-diagnostican\venv\Scripts\python.exe scripts/run_fixture_evaluation.py --output docs/evaluation
```

## README-ready snippet

Copy from the bottom of `SUMMARY.md` after each full run.
