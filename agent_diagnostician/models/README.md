# models/ — Documentation

Living reference for the three model files. Update this whenever a field
is added, removed, or changed in enums.py, trace.py, or result.py — this
doc should always reflect what's actually built, not what was planned.

---

## Why these files exist

Every detector needs to (1) know exactly what shape an incoming trace is,
and (2) return output in one consistent shape, regardless of which failure
category it's checking for. These three files are that contract. No
detector or analysis module should ever pass around a raw dict — always
these Pydantic objects.

---

## enums.py

Fixed lists of named constants. Built on Python's built-in `enum` module
(not a third-party library — no install needed). Each class inherits from
both `str` and `Enum`, meaning values serialize cleanly to/from JSON.

| Enum | Values | Used for |
|---|---|---|
| `FailureType` | Tool Use Failure, Hallucination, Goal Satisfaction Failure, Context Loss, Token Exhaustion, Premature Termination, Infinite Loop, None | Top-level category, set on every `DetectionResult` |
| `ToolUseSubtype` | Wrong Tool Selected, Invalid Parameters, Incorrect Parameter Values, No Tool Use Failure, Insufficient Evidence | Tool Use detector's verdict |
| `GoalFailureSubtype` | Constraint Violation, Task Misinterpretation, No Goal Failure, Insufficient Evidence | Goal Failure detector's verdict |
| `ConfidenceBand` | Confirmed, Likely, Possible, Insufficient Evidence | Bucketed confidence, on every `DetectionResult` |

**Note:** subtype enums for Hallucination, Context Loss, Token Exhaustion,
Premature Termination, and Infinite Loop don't exist yet — add them when
each detector's logic is actually designed, not before.

---

## trace.py

Defines the internal, framework-agnostic shape of an agent trace.
`tracer.py` (not yet built) is responsible for converting raw
framework-specific logs (LangChain, LangGraph, AutoGen, custom) into
these models. Everything downstream — detectors, analysis modules — only
ever sees this shape.

### ToolSpec
One entry in `available_tools` (Tier 3, optional). Used for embedding-based
tool ranking and schema validation.

| Field | Type | Required? | Notes |
|---|---|---|---|
| `name` | str | yes | |
| `description` | str | yes | used for embedding similarity |
| `schema_` | dict or None | no | trailing underscore — `schema` is a reserved Pydantic name, can't be used directly |

### Step
One tool invocation. Tier 1 fields required; Tier 2/3 optional and default
to `None`. **A field being `None` is not an error — it's the actual
mechanism that triggers fallback-level logic in detectors** (e.g. `if
step.thought is not None:` decides which fallback stage runs).

| Field | Tier | Type | Required? |
|---|---|---|---|
| `step_index` | 1 | int | yes |
| `tool_name` | 1 | str | yes |
| `tool_input` | 1 | dict | yes |
| `tool_output` | 1 | Any | no (call may fail with no output) |
| `timestamp` | 2 | str | no |
| `error_message` | 2 | str | no |
| `step_status` | 2 | str | no |
| `thought` | 3 | str | no |
| `retry_count` | 3 | int | no |
| `prompt_tokens` | 3 | int | no |
| `completion_tokens` | 3 | int | no |

### AgentTrace
Run-level container. This is the object every `detect(trace)` call
receives.

| Field | Tier | Type | Required? |
|---|---|---|---|
| `run_id` | 1 | str | yes |
| `task` | 1 | str | yes |
| `status` | 1 | str | yes |
| `total_steps` | 1 | int | yes |
| `final_output` | 1 | Any | no |
| `steps` | 1 | list[Step] | yes |
| `total_tokens` | 2 | int | no |
| `available_tools` | 3 | list[ToolSpec] | no |
| `constraints` | 3 | list[str] | no (raw, if framework provides directly) |
| `constraint_list` | 4 (derived) | list[dict] | no — computed later by `ConstraintExtractor`, never present in raw JSON |

`tool_output`, `final_output`, and `tool_input`'s inner values are typed
loosely (`Any`) on purpose — real traces show these can be a string,
number, or dict depending on the tool. Tightening this would break the
framework-agnostic goal.

---

## result.py

Defines what every detector hands back. `classifier.py` compares results
from different detectors using this shared shape.

### Evidence
One signal that contributed to a verdict. All fields required — an
Evidence entry with no explanation isn't useful.

| Field | Type | Meaning |
|---|---|---|
| `detection_stage` | str | which stage produced this, e.g. "1A - Task vs Thought" |
| `signal` | str | short label, e.g. "grounding_check_failed" |
| `confidence_contribution` | float | 0–1, this signal's weight toward final confidence |
| `explanation` | str | human-readable reason |

### DetectionResult
| Field | Type | Notes |
|---|---|---|
| `failure_type` | `FailureType` enum | not a raw string — Pydantic rejects anything outside the 7 defined categories |
| `subtype` | str | plain string, not enum — each category has its own subtype enum, a shared field can't type against multiple enums at once |
| `confidence_score` | float | 0–1 |
| `confidence_band` | `ConfidenceBand` enum | |
| `evidence` | list[Evidence] | every signal that contributed |
| `reason` | str | human-readable summary |
| `fix_direction` | str or None | optional actionable suggestion |
| `detection_stage` | str | which stage produced the final verdict |
| `secondary_evidence` | `DetectionResult` or None | self-referencing — see note below |

**Self-reference note:** `secondary_evidence` is typed as
`Optional["DetectionResult"]` — in quotes, because at the point Python
reads that line, `DetectionResult` isn't fully defined yet (it's defining
itself). Quoting it tells Python to resolve the reference later, once the
whole class exists. This is what allows Goal Failure's design (primary
failure + a lesser secondary `DetectionResult`) to work without a separate
custom structure.

---

## Expected changes going forward

These models are a first-pass, not final:
- New Tier fields likely needed as Hallucination, Context Loss, Token
  Exhaustion, Premature Termination, and Infinite Loop get designed
  (e.g. `retry_count` exists already but is barely used yet — Infinite
  Loop detection will likely lean on it heavily).
- Real trace ingestion via `tracer.py` may surface a framework-specific
  shape these models don't yet handle — expected, not a design flaw.
- Don't change these casually mid-implementation without noting why here
  — every detector downstream depends on these exact shapes staying
  stable within a working session.
