# Result model definitions
# result.py
# Defines the output side of the pipeline. Every detector, regardless of
# which failure category it handles, returns a DetectionResult built from
# this same shape. This is what lets classifier.py compare results from
# totally different detectors on equal footing.

from typing import Optional
from pydantic import BaseModel

from agent_diagnostician.models.enums import FailureType, ConfidenceBand


class Evidence(BaseModel):
    """One piece of supporting evidence behind a detector's decision.
    A DetectionResult can carry multiple Evidence entries -- e.g. Tool Use
    Stage 3 might combine a grounding-check signal AND an LLM judge signal."""

    detection_stage: str          # e.g. "1A - Task vs Thought", "2 - Runtime Schema Inference"
    signal: str                   # short label for what fired, e.g. "grounding_check_failed"
    confidence_contribution: float  # 0-1, how much this signal contributed to final confidence
    explanation: str              # human-readable reason this evidence was flagged


class DetectionResult(BaseModel):
    """The final output of any detector's detect(trace) -> DetectionResult call."""

    failure_type: FailureType
    subtype: str                  # left as str, not a single shared enum -- each category
                                   # has its own subtype enum (ToolUseSubtype, GoalFailureSubtype, etc.)

    confidence_score: float       # 0-1, normalized
    confidence_band: ConfidenceBand

    evidence: list[Evidence]      # every signal that contributed to the verdict
    reason: str                   # human-readable summary of why this verdict was reached
    fix_direction: Optional[str] = None  # actionable suggestion, if applicable

    detection_stage: str          # which stage ultimately produced the verdict

    # For categories like Goal Failure where a primary + a weaker secondary
    # signal can both be true (e.g. constraint violation is primary, but a
    # minor intent/execution inconsistency was also observed) -- optional,
    # most detectors will leave this None.
    secondary_evidence: Optional["DetectionResult"] = None