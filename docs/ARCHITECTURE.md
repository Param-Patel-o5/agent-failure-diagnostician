# Architecture

High-level layout of Agent Diagnostician. See also [`agent_diagnostician/models/README.md`](../../agent_diagnostician/models/README.md) for data contracts.

## Pipeline

```
Trace JSON  →  tracer.load_fixture()  →  AgentTrace
                                              ↓
                         Classifier.diagnose() runs enabled detectors
                                              ↓
                         Each detector → DetectionResult
                                              ↓
                         Classifier picks highest-confidence diagnosis
                                              ↓
                         DiagnosisResult (failure_type, subtype, evidence)
```

## Components

| Layer | Role |
|-------|------|
| `classifier.py` | Runs detectors, aggregates, tiebreaker by priority |
| `detectors/` | One module per failure family (tool use, goal, hallucination, …) |
| `analysis/` | Shared: embeddings, grounding, constraints, schema, `llm/` judge |
| `models/` | Pydantic trace + result types |
| `prompts/` | LLM prompt templates |
| `reporter.py` | Format diagnosis for humans / JSON export |

## Detection tiers

Most detectors use a **layered funnel**:

1. Deterministic rules (schema, constraints, fuzzy error text)
2. Embeddings (tool ranking, thought vs output similarity)
3. LLM judge fallback (ambiguous cases only)

Exceptions:

- **Token exhaustion** — rules only (no LLM)
- **Infinite loop** — rules + embeddings for thought similarity (no LLM calls in practice)
- **Hallucination** — grounding + LLM blend every step; grounding-only escalation only on API failure

## LLM judge

Provider-agnostic package: `agent_diagnostician/analysis/llm/`

```python
from agent_diagnostician.analysis.llm import create_llm_judge_from_env
judge = create_llm_judge_from_env()
```

Factories: `create_llm_judge()`, providers in `llm/providers/` (Gemini, OpenAI, Anthropic).

## Classifier priority (tiebreaker)

When multiple detectors fire with similar confidence:

1. Tool use failure
2. Goal satisfaction failure
3. Context loss
4. Token exhaustion
5. Premature termination
6. Infinite loop
7. Hallucination

## Evaluation

60 local JSON fixtures in `test cases/` (gitignored). Runners:

- `scripts/run_phased_fixture_evaluation.py` — rate-limited, recommended
- `scripts/run_fixture_evaluation.py` — full sequential run

Results: `docs/evaluation/`.

See also [Troubleshooting](TROUBLESHOOTING.md) for API keys and rate limits.
