# Enum definitions for agent diagnostics
from enum import Enum


class FailureType(str, Enum):
    TOOL_USE_FAILURE = "tool_use_failure"
    HALLUCINATION = "hallucination"
    GOAL_SATISFACTION_FAILURE = "goal_satisfaction_failure"
    CONTEXT_LOSS = "context_loss"
    TOKEN_EXHAUSTION = "token_exhaustion"
    PREMATURE_TERMINATION = "premature_termination"
    INFINITE_LOOP = "infinite_loop"
    NONE = "none"  # used when no failure detected at all


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


class ConfidenceBand(str, Enum):
    CONFIRMED = "confirmed"
    LIKELY = "likely"
    POSSIBLE = "possible"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"