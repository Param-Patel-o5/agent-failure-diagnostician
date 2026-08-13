from enum import Enum


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"


class LLMResponseStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    PARSE_FAILED = "parse_failed"


class LLMErrorType(str, Enum):
    QUOTA_EXCEEDED = "quota_exceeded"
    AUTHENTICATION = "authentication"
    MODEL_NOT_FOUND = "model_not_found"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    UNKNOWN = "unknown"


class FailureType(str, Enum):
    TOOL_USE_FAILURE = "tool_use_failure"
    HALLUCINATION = "hallucination"
    GOAL_SATISFACTION_FAILURE = "goal_satisfaction_failure"
    CONTEXT_LOSS = "context_loss"
    TOKEN_EXHAUSTION = "token_exhaustion"
    PREMATURE_TERMINATION = "premature_termination"
    INFINITE_LOOP = "infinite_loop"
    NONE = "none"


class ClassifierSubtype(str, Enum):
    """Classifier-level aggregate result (not a detector subtype)."""
    NO_FAILURE = "no_failure"


class ToolUseSubtype(str, Enum):
    WRONG_TOOL_SELECTED = "wrong_tool_selected"
    INVALID_PARAMETERS = "invalid_parameters"
    INCORRECT_PARAMETER_VALUES = "incorrect_parameter_values"
    NO_FAILURE = "no_tool_use_failure"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class GoalFailureSubtype(str, Enum):
    CONSTRAINT_VIOLATION = "constraint_violation"
    TASK_MISINTERPRETATION = "task_misinterpretation"
    NO_FAILURE = "no_goal_failure"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class TokenExhaustionSubtype(str, Enum):
    TOKEN_EXHAUSTION = "token_exhaustion"
    NO_TOKEN_EXHAUSTION = "no_token_exhaustion"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class HallucinationSubtype(str, Enum):
    HALLUCINATION_DETECTED = "hallucination_detected"
    NO_HALLUCINATION = "no_hallucination"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class InfiniteLoopSubtype(str, Enum):
    DEGRADED_SUCCESS = "degraded_success"
    EXACT_REPETITION = "exact_repetition"
    STUCK_ON_FAILURE = "stuck_on_failure"
    REASONING_LOOP = "reasoning_loop"
    NO_INFINITE_LOOP = "no_infinite_loop"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ContextLossSubtype(str, Enum):
    CONTEXT_LOSS_DETECTED = "context_loss_detected"
    NO_CONTEXT_LOSS = "no_context_loss"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class PrematureTerminationSubtype(str, Enum):
    PREMATURE_TERMINATION_DETECTED = "premature_termination_detected"
    NO_PREMATURE_TERMINATION = "no_premature_termination"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ConfidenceBand(str, Enum):
    CONFIRMED = "confirmed"
    LIKELY = "likely"
    MAYBE = "maybe"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# ── Trace / step status (Tier 1–2 trace fields) ─────────────────────────────

class TraceRunStatus(str, Enum):
    SUCCESS = "success"
    COMPLETED = "completed"
    ERROR = "error"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    TIMEOUT = "timeout"


class StepRunStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    FAILED = "failed"
    CONTINUING = "continuing"


TRACE_SUCCESS_STATUSES = frozenset({
    TraceRunStatus.SUCCESS.value,
    TraceRunStatus.COMPLETED.value,
})

STEP_FAILURE_STATUSES = frozenset({
    StepRunStatus.ERROR.value,
    StepRunStatus.FAILED.value,
})

TRACE_FAILURE_STATUSES = frozenset({
    TraceRunStatus.ERROR.value,
    TraceRunStatus.FAILED.value,
    TraceRunStatus.INCOMPLETE.value,
    TraceRunStatus.TIMEOUT.value,
})


# ── LLM judge verdict strings (prompt contract) ─────────────────────────────

class ToolSelectionVerdict(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"


class ParameterStructureVerdict(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"


class ParameterValuesVerdict(str, Enum):
    JUSTIFIED = "justified"
    UNJUSTIFIED = "unjustified"
    UNCERTAIN = "uncertain"


class GoalAlignmentVerdict(str, Enum):
    CORRECT = "correct"
    MISINTERPRETED = "misinterpreted"
    UNCERTAIN = "uncertain"


class ContextLossVerdict(str, Enum):
    CONTEXT_LOST = "context_lost"
    NO_CONTEXT_LOSS = "no_context_loss"
    UNCERTAIN = "uncertain"


class PrematureTerminationVerdict(str, Enum):
    PREMATURE = "premature"
    COMPLETE = "complete"
    UNCERTAIN = "uncertain"


# Canonical insufficient-evidence subtype string (shared by all detector subtypes).
INSUFFICIENT_EVIDENCE_SUBTYPE = ToolUseSubtype.INSUFFICIENT_EVIDENCE.value


# ── Constraint validation result fields ──────────────────────────────────────

class ConstraintValidationField(str, Enum):
    SATISFIED = "satisfied"
    REASON = "reason"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# ── Grounding analysis classifications ───────────────────────────────────────

class GroundingClassification(str, Enum):
    DIRECT = "direct"
    DERIVED = "derived"
    UNGROUNDED = "ungrounded"


# ── Repeated evidence signal labels ─────────────────────────────────────────

class EvidenceSignal(str, Enum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    LLM_UNCERTAIN_VERDICT = "llm_uncertain_verdict"
    LLM_MISINTERPRETED_VERDICT = "llm_misinterpreted_verdict"
    LOW_TASK_OUTPUT_SIMILARITY = "low_task_output_similarity"
    ALL_ENABLED_DETECTORS = "all_enabled_detectors"


# ── Aggregated no-failure subtypes for classifier filtering ─────────────────

NO_FAILURE_SUBTYPE_VALUES = frozenset({
    ToolUseSubtype.NO_FAILURE.value,
    GoalFailureSubtype.NO_FAILURE.value,
    HallucinationSubtype.NO_HALLUCINATION.value,
    TokenExhaustionSubtype.NO_TOKEN_EXHAUSTION.value,
    InfiniteLoopSubtype.NO_INFINITE_LOOP.value,
    ContextLossSubtype.NO_CONTEXT_LOSS.value,
    PrematureTerminationSubtype.NO_PREMATURE_TERMINATION.value,
    FailureType.NONE.value,
    ClassifierSubtype.NO_FAILURE.value,
})
