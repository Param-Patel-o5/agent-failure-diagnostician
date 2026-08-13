# Agent Diagnostician

A framework-agnostic Python library that diagnoses LLM agent execution failures from JSON traces. Given a run log (task, steps, tool calls, outputs), it returns a failure category, subtype, confidence score, evidence chain, and suggested fix direction.

Works with traces from LangChain, LangGraph, AutoGen, custom agents, or plain JSON fixtures. Python 3.11+.

## What you get

For each trace, the classifier returns:

- **Failure type** — e.g. tool use failure, hallucination, goal satisfaction failure
- **Subtype** — e.g. wrong tool selected, constraint violation, stuck on failure
- **Confidence** — numeric score (0–1) and band (confirmed, likely, maybe, insufficient evidence)
- **Evidence** — which signals fired and at which detection stage
- **Fix direction** — actionable guidance for prompts, tools, or recovery logic

## Benchmark

Validated on **60 local JSON trace fixtures** with phased evaluation and live LLM where required (`gemini-3.5-flash-lite`):

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **53/60 (88.3%)** |
| Run time | 251.8s (63 API calls, 15 req/min limit) |
| Fixtures without API | 36 (tier-1/2 rules, embeddings, or no-LLM detectors) |
| Fixtures with live LLM | 24 |

### By failure category

| Category | Fixtures | Passed | Accuracy |
|----------|----------|--------|------------|
| Context loss | 6 | 6 | 100% |
| Infinite loop | 9 | 9 | 100% |
| Hallucination | 9 | 9 | 100% |
| Premature termination | 6 | 6 | 100% |
| Classifier (multi-detector e2e) | 5 | 4 | 80% |
| Goal satisfaction failure | 11 | 9 | 81.8% |
| Tool use failure | 14 | 10 | 71.4% |

All hallucination fixtures were run with a **live LLM judge** (grounding + model blend), not mock bypass.

Full results, per-fixture detail, and failure analysis: [`docs/evaluation/SUMMARY.md`](docs/evaluation/SUMMARY.md).

Known gaps (7 failing fixtures): goal-failure aggregator edge cases (2), classifier tiebreaker (1), tool-use LLM-fallback and negative-control cases (4). See SUMMARY for expected vs actual.

## Installation

```bash
pip install -e .
```

Optional LLM providers:

```bash
pip install -e ".[llm-openai]"
pip install -e ".[llm-anthropic]"
pip install -e ".[llm-all]"
```

Core dependencies include Pydantic, sentence-transformers (local embeddings), scikit-learn, and google-generativeai (default Gemini provider).

## Quick start

```python
from agent_diagnostician import Classifier
from agent_diagnostician.analysis.llm import create_llm_judge_from_env
from agent_diagnostician.tracer import load_fixture

# Configure via environment (see LLM configuration below)
classifier = Classifier(llm_judge=create_llm_judge_from_env())

trace = load_fixture("path/to/trace.json")
result = classifier.diagnose(trace)

print(result.failure_type.value)
print(result.subtype)
print(result.confidence_score, result.confidence_band.value)
print(result.reason)
```

Load a fixture from disk or build an `AgentTrace` in code:

```python
from agent_diagnostician.models.trace import AgentTrace, Step

trace = AgentTrace(
    run_id="run_001",
    task="Refund order ORD-123 for $49.99",
    status="failed",
    total_steps=1,
    final_output=None,
    steps=[
        Step(
            step_index=0,
            tool_name="issue_refund",
            tool_input={"order_id": "ORD-999", "amount": 49.99},
            tool_output={"error": "Order not found"},
        ),
    ],
)

result = classifier.diagnose(trace)
```

Human-readable output:

```python
from agent_diagnostician.reporter import Reporter

Reporter.print_cli(result)
```

More examples: [`examples/basic_usage.py`](examples/basic_usage.py).

## LLM configuration

Detectors use a provider-agnostic judge for ambiguous cases. Set environment variables and inject the judge into the classifier:

```bash
export LLM_PROVIDER=gemini          # gemini | openai | anthropic | mock
export LLM_API_KEY=your-key         # or GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
export LLM_MODEL=gemini-3.5-flash-lite
```

```python
from agent_diagnostician.analysis.llm import create_llm_judge_from_env

judge = create_llm_judge_from_env()
classifier = Classifier(llm_judge=judge)
```

Smoke test:

```bash
python scripts/configure_llm.py --test
```

Without an API key, detectors fall back to `MockLLMJudge` (uncertain verdicts). Production diagnosis and benchmark runs should use a real provider. See [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) for rate limits and mock vs live behavior.

Run only specific detectors:

```python
from agent_diagnostician.models.enums import FailureType

classifier = Classifier(
    llm_judge=judge,
    enabled_detectors=[FailureType.TOKEN_EXHAUSTION, FailureType.INFINITE_LOOP],
)
```

## Failure types and detectors

Seven detectors cover the main agent failure modes. The classifier runs enabled detectors and returns the single highest-confidence diagnosis (with priority tiebreaking).

| Failure type | Detector | LLM | Typical subtypes |
|--------------|----------|-----|------------------|
| Tool use failure | `ToolUseDetector` | Fallback tier | Wrong tool, invalid parameters, incorrect values |
| Goal satisfaction failure | `GoalFailureDetector` | Core for misinterpretation | Constraint violation, task misinterpretation |
| Hallucination | `HallucinationDetector` | Always (per step) | Detected, no hallucination, insufficient evidence |
| Context loss | `ContextLossDetector` | Fallback tier | Detected, no context loss |
| Token exhaustion | `TokenExhaustionDetector` | No | Token exhaustion, no exhaustion |
| Premature termination | `PrematureTerminationDetector` | Fallback tier | Detected, no premature termination |
| Infinite loop | `InfiniteLoopDetector` | No | Exact repetition, stuck on failure, reasoning loop, degraded success |

Classifier priority when scores tie: tool use → goal failure → context loss → token exhaustion → premature termination → infinite loop → hallucination.

### Detection approach

Most detectors use a **layered funnel**:

1. **Rules** — schema checks, constraint validation, error-message matching
2. **Embeddings** — semantic similarity (tool ranking, thought vs output, grounding)
3. **LLM judge** — structured prompts when earlier tiers are inconclusive

Exceptions:

- **Token exhaustion** — explicit error phrases and token-ratio heuristics only
- **Infinite loop** — repetition analysis, input/error patterns, optional thought similarity (no LLM calls)
- **Hallucination** — grounding score blended with LLM confidence each step; grounding-only escalation applies only when the API fails, not when the judge returns uncertain

Thresholds and weights live in [`agent_diagnostician/config.py`](agent_diagnostician/config.py).

## Architecture

```
Trace JSON  →  tracer.load_fixture()  →  AgentTrace
                                              ↓
                         Classifier.diagnose()  (enabled detectors)
                                              ↓
                         DetectionResult per detector
                                              ↓
                         Best diagnosis + evidence chain
```

| Package area | Role |
|--------------|------|
| `classifier.py` | Orchestration, tiebreaking |
| `detectors/` | Planning, execution, termination pipelines |
| `analysis/` | Embeddings, grounding, constraints, `llm/` judge |
| `models/` | Pydantic trace and result types |
| `prompts/` | LLM prompt templates |
| `reporter.py` | CLI, JSON, Markdown formatting |
| `tracer.py` | JSON fixture loading (framework adapters planned) |

Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Trace format

Traces are JSON objects with at least:

```json
{
  "run_id": "run_001",
  "task": "User task text",
  "status": "failed",
  "total_steps": 2,
  "final_output": null,
  "steps": [
    {
      "step_index": 0,
      "tool_name": "search",
      "tool_input": {"query": "..."},
      "tool_output": {"results": []},
      "thought": "optional reasoning",
      "error_message": "optional"
    }
  ]
}
```

Optional fields: `available_tools`, `constraints`, `constraint_list`, `total_tokens`, per-step token counts. See [`agent_diagnostician/models/README.md`](agent_diagnostician/models/README.md).

## Running evaluations

Phased runner (recommended; respects API rate limits):

```bash
python scripts/run_phased_fixture_evaluation.py \
  --initial-used 0 \
  --first-batch 3 \
  --batch-size 15 \
  --output docs/evaluation
```

Full sequential runner:

```bash
python scripts/run_fixture_evaluation.py --output docs/evaluation
```

Output: `docs/evaluation/SUMMARY.md` and `docs/evaluation/fixture_results.json`.

Methodology: [`docs/evaluation/README.md`](docs/evaluation/README.md).

## Documentation

| Document | Contents |
|----------|----------|
| [API Reference](docs/API.md) | Classes, methods, enums |
| [Architecture](docs/ARCHITECTURE.md) | Pipeline, tiers, LLM layer |
| [Integrations](docs/INTEGRATIONS.md) | LangChain and other frameworks |
| [Performance](docs/PERFORMANCE.md) | Embeddings, batching, cost |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | API keys, quotas, mock vs live |
| [Evaluation](docs/evaluation/README.md) | Fixture benchmarks |
| [Contributing](CONTRIBUTING.md) | Development workflow |

## Project layout

```
agent_diagnostician/
├── classifier.py
├── config.py
├── tracer.py
├── reporter.py
├── detectors/
│   ├── planning/          # tool_use, goal_failure, hallucination
│   ├── execution/         # context_loss, token_exhaustion
│   └── termination/       # infinite_loop, premature_termination
├── analysis/
│   ├── embeddings.py
│   ├── grounding.py
│   ├── constraint_extractor.py
│   └── llm/               # provider-agnostic judge
├── models/
└── prompts/
docs/
examples/
scripts/                   # configure_llm, fixture evaluation
```

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, detector patterns, and prompt guidelines.

## License

MIT — see [LICENSE](LICENSE).
