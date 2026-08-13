# Agent Failure Diagnostician — Architecture Reference

This document is the locked source of truth for this project's architecture.
Do not redesign, restructure, or deviate from anything in this file without
explicit instruction. If something seems missing or unclear, ask before
inventing a solution.

---

## Project Goal

A pip-installable, framework-agnostic Python library. Given an agent
execution trace (JSON), it returns:

- Failure category
- Failure subtype
- Confidence score
- Supporting evidence
- Human-readable reason
- Fix direction

Works with traces from LangChain, LlamaIndex, AutoGen, OpenAI Responses API,
or custom/prompt-based agent frameworks.

---

## Design Philosophy

- Composition over inheritance.
- Deterministic checks before heuristic checks. Heuristic checks before LLM
  reasoning. Priority order: Rules > Runtime Inference > Embeddings > LLM.
- One responsibility per module.
- Shared data contracts through typed models — Pydantic throughout, no raw
  dicts passed between modules.
- Analysis modules answer isolated questions only. They never decide
  pipeline flow or control execution order.
- Detectors own all business logic and orchestration.
- classifier.py aggregates results. It performs no analysis itself.
- tracer.py handles framework-specific ingestion only.
- reporter.py handles output formatting only. No detection logic belongs
  here.
- Explainable outputs — every result carries evidence and confidence, not
  just a verdict.
- Extensible — new failure categories should be addable with minimal
  changes to existing code.

---

## Folder Structure

```
agent_diagnostician/
│
├── classifier.py
├── tracer.py
├── reporter.py
│
├── detectors/
│   ├── base.py
│   │
│   ├── planning/
│   │   ├── tool_use.py
│   │   ├── hallucination.py
│   │   ├── goal_failure.py
│   │   └── reflection_failure.py
│   │
│   ├── execution/
│   │   ├── context_loss.py
│   │   └── token_exhaustion.py
│   │
│   └── termination/
│       ├── premature_termination.py
│       └── infinite_loop.py
│
├── analysis/
│   ├── schema.py
│   ├── embeddings.py
│   ├── similarity.py
│   ├── grounding.py
│   ├── traceability.py
│   ├── constraint_extractor.py
│   ├── confidence.py
│   └── llm/                  # Provider-agnostic LLM judge (gemini, openai, anthropic)
│
├── models/
│   ├── trace.py
│   ├── tool_call.py
│   ├── evidence.py
│   ├── result.py
│   └── enums.py
│
├── prompts/
│   ├── wrong_tool.txt
│   ├── parameter_structure.txt
│   ├── parameter_values.txt
│   ├── goal_constraint.txt
│   ├── goal_misinterpretation.txt
│   └── ...
│
├── utils/
│
└── config.py
```

Note: original spec had 8 categories under Planning/Execution/Termination.
Error Recall Failure has been dropped — 7 categories total. Update any
generated enums/detectors to reflect 7, not 8.

---

## Data Model Requirements (models/)

These are Pydantic models. No detector or analysis module should ever
receive or pass a raw dict for trace data.

### Trace Tiers (fields the models must support)

**Tier 1 — Universal, always present**
- Run level: run_id, task, status, total_steps, final_output
- Step level: step_index, tool_name, tool_input, tool_output

**Tier 2 — Common, not guaranteed**
- timestamp, error_message, total_tokens, step_status

**Tier 3 — Rare, needs explicit instrumentation**
- thought / reasoning (per step)
- prompt_tokens, completion_tokens (per step)
- retry_count
- available_tools (list of {name, description, schema} — schema itself
  optional per tool)
- constraints (raw, if framework provides them explicitly)

**Tier 4 — Derived, computed by the library itself, not present in raw input**
- repeated_tool_pattern
- step_token_estimate
- tool_output_referenced
- constraint_list (output of ConstraintExtractor)
- task_achieved
- input_grounded
- step_count_vs_progress

All Tier 2/3 fields must be optional in the models (nullable / default None).
Detector logic must degrade gracefully when they're absent — this is the
entire reason the fallback-level pipelines exist.

### Required Models

- `AgentTrace` — run-level container: run_id, task, status, total_steps,
  final_output, steps: List[Step], available_tools: Optional[List[ToolSpec]]
- `Step` — step_index, tool_name, tool_input, tool_output, thought
  (optional), timestamp (optional), error_message (optional), step_status
  (optional)
- `ToolSpec` — name, description, schema (optional)
- `Evidence` — what signal fired, which detection stage produced it,
  confidence contribution, human-readable explanation
- `DetectionResult` — failure_type, subtype, confidence (0–1 + banded:
  Confirmed/Likely/Possible/Insufficient Evidence), evidence: List[Evidence],
  reason, fix_direction, detection_stage, secondary_evidence (optional,
  for cases like Goal Failure where a primary + supporting signal both
  exist)
- Enums: FailureType, FailureSubtype (per category), ConfidenceBand

---

## Detector 1 (Reference Implementation): Tool Use Failure

Exactly one subtype returned per tool invocation.

Outcomes: Wrong Tool Selected | Valid Tool, Invalid Parameters |
Valid Tool, Incorrect Parameter Values | No Tool Use Failure |
Insufficient Evidence

### Pipeline (stop-on-first-hit)

```
Stage 1 — Wrong Tool Selected?
  1A. If thought present: task vs thought (comprehension check)
  1B. If 1A passes: thought vs tool_name (selection check)
  2.  If available_tools with descriptions present: embed task against
      each tool description, rank. Do NOT require rank-1 — use a
      similarity gap threshold so equivalent tools (e.g. web_search vs
      wikipedia_search) don't false-flag.
  3.  LLM fallback if 1/2 unavailable or inconclusive.
  → If failed: return Wrong Tool Selected. STOP.

Stage 2 — Invalid Parameters? (assumes correct tool)
  1. If schema present (in available_tools): validate directly —
     required fields, types, enums, nesting, etc.
  2. If schema absent: infer runtime schema from ≥2 prior successful
     calls to the same tool in this trace.
     Edge case: if this is the first/only call to this tool, skip
     directly to step 3 (no inference possible).
  3. LLM fallback — judges structure only, not values.
  → If failed: return Valid Tool, Invalid Parameters. STOP.

Stage 3 — Incorrect Parameter Values? (assumes tool + schema correct)
  1. Grounding check (default, zero-cost): for each value in tool_input,
     check fuzzy (not exact) match against task string and all prior
     tool_output. Derived values (e.g. computed from a prior output)
     must be recognized as grounded, not false-flagged.
  2. If thought present: extract intended values from thought text,
     compare (semantically) against actual tool_input values.
  3. LLM judge / embedding fallback — given task, all prior tool_outputs,
     and current tool_input, judge whether each value is justified.
  → If failed: return Valid Tool, Incorrect Parameter Values. STOP.

→ If none failed: No Tool Use Failure.
```

Reusable analysis modules for this detector: SchemaValidator,
EmbeddingMatcher, GroundingAnalyzer, TraceabilityAnalyzer, LLMJudge,
ConfidenceAggregator.

---

## Detector 2: Goal Failure

Two subtypes. Unlike Tool Use, **both branches run independently** — this
is NOT a stop-on-first-hit pipeline. A single primary failure is reported,
with the other kept as optional secondary evidence if it clears a
reporting threshold. Never report two co-equal failures.

Outcomes: Constraint Violation | Task Misinterpretation | No Goal Failure |
Insufficient Evidence

### Pipeline

```
Stage 0 — Constraint Extraction
  Use ConstraintExtractor to pull constraints from task text.
  Types: Numeric ("under 100"), Categorical ("use Python"),
  Structural ("return JSON"), Semantic ("professional tone").
  Output: normalized constraint_list (Tier 4, derived).

Stage 1 — Constraint Validation (parallel branches, not sequential)
  a. Rule/numeric check — deterministic extraction + comparison for
     numeric constraints. No embedding, no LLM.
  b. Keyword/categorical check — string/fuzzy match for categorical
     and structural constraints.
  c. Semantic check — embedding similarity for semantic-type constraints
     (tone, style) with no exact match possible.
  d. Thought consistency (if present) — bonus signal only, does not gate.
  → Combine into one Constraint Confidence Score.

Stage 2 — Task Misinterpretation Analysis
  a. Thought vs execution (if thought present) — compares stated intent
     to actual tool_output per step. Checks execution followed the plan;
     does NOT judge if the plan itself was correct.
  b. Embedding similarity (task vs final_output) — computed as a WEAK
     SUPPORTING SIGNAL ONLY, fed to the LLM as context. This is NOT a
     standalone gating stage — proven too weak/misleading on its own
     (e.g. correct-sounding final_output text can mask wrong underlying
     logic).
  c. Full LLM reasoning — the core engine for this subtype. Inputs:
     task, every step's tool_input/tool_output, thought if present,
     final_output, embedding score as context. Judges whether task was
     solved correctly, including cases where stated intent and execution
     agree but the underlying logic itself is wrong.
  → Combine into Misinterpretation Confidence Score.

Aggregator
  Compare both confidence scores. Report the higher as Primary Failure.
  Report the other as Supporting Evidence only if it clears a minimum
  reporting threshold. Never report both as co-equal failures.
```

Reusable analysis modules: ConstraintExtractor (new), EmbeddingMatcher
(shared with Tool Use), LLMJudge (shared, needs new method
evaluate_goal_alignment()), ConfidenceAggregator (shared).

No GroundingAnalyzer/TraceabilityAnalyzer reuse here.

---

## Confidence Bands

All detectors normalize confidence to 0–1, then bucket into:
Confirmed | Likely | Possible | Insufficient Evidence

Evidence weighting follows: Rules > Runtime Inference > Embeddings > LLM.
Exact numeric weights are tunable per-detector, not fixed globally.

---

## Development Order (locked — do not skip ahead)

1. Data models (AgentTrace, Step, ToolSpec, DetectionResult, Evidence,
   enums) — models/ only, no logic.
2. BaseDetector — interface only (`detect(trace) -> DetectionResult`),
   common utilities (input validation, result creation helpers, logging).
   No failure logic here.
3. Minimum analysis modules needed for Tool Use (SchemaValidator,
   EmbeddingMatcher, GroundingAnalyzer, LLMJudge interface — mock LLM
   implementation acceptable for now).
4. ToolUseDetector, end to end.
5. Validate against the Tool Use test trace fixtures (test_traces/
   wrong_tool_selected/, invalid_parameters/, incorrect_parameter_values/).
6. Only once Tool Use is stable: remaining detectors, following the same
   pattern, reusing analysis components where they genuinely apply.

Do not build all detectors' code before Tool Use is validated end-to-end.
Do not redesign the architecture mid-implementation without explicit
sign-off — pressure-test against real traces first, raise concerns, then
revise deliberately if needed.

---

## Test Fixtures

Location: test_traces/<failure_type>/
Naming convention: <failure_type>__<subtype_or_stage>__<fallback_level>.json
Each fixture includes an `expected_diagnosis` field for test assertions
only — this field must be stripped/ignored before feeding a trace to any
production ingestion path (tracer.py), since real traces will never
contain it.

---

## LLM Judge Notes

LLMJudge is not a generic prompt executor. It exposes task-specific
methods, e.g.:
- evaluate_wrong_tool()
- evaluate_parameter_structure()
- evaluate_parameter_values()
- evaluate_goal_alignment()

Each method loads its own prompt template from prompts/, fills variables,
calls the configured LLM, parses the response into a structured output.
Initial implementation may use a mock LLM for development/testing.
Real provider support (OpenAI, Anthropic, Gemini, Ollama,
OpenRouter-compatible) comes later — detectors must never need code
changes when the provider is swapped.
