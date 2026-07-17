# Base detector class for agent diagnostics
# base.py
# Shared skeleton every detector (ToolUseDetector, GoalFailureDetector, etc.)
# inherits from. Contains NO failure-detection logic -- only the enforced
# interface (detect) and small shared helper utilities every detector
# would otherwise have to rewrite.

from abc import ABC, abstractmethod

from agent_diagnostician.models.trace import AgentTrace
from agent_diagnostician.models.result import DetectionResult, Evidence
from agent_diagnostician.models.enums import ConfidenceBand


class BaseDetector(ABC):
    """Every concrete detector must inherit from this and implement detect().
    Python will refuse to instantiate a subclass that doesn't override
    detect() -- that's what @abstractmethod enforces."""

    @abstractmethod
    def detect(self, trace: AgentTrace) -> DetectionResult:
        """Given a trace, return a DetectionResult. Every subclass must
        implement this with its own category-specific pipeline logic."""
        raise NotImplementedError

    # --- Shared helpers below. Not abstract -- subclasses may use these
    # as-is, ignore them, or override if a detector genuinely needs
    # different behavior. ---

    @staticmethod
    def confidence_to_band(score: float) -> ConfidenceBand:
        """Converts a raw 0-1 confidence score into a ConfidenceBand.
        Thresholds are a starting point -- tune per detector later if needed."""
        if score >= 0.85:
            return ConfidenceBand.CONFIRMED
        elif score >= 0.6:
            return ConfidenceBand.LIKELY
        elif score >= 0.3:
            return ConfidenceBand.MAYBE
        else:
            return ConfidenceBand.INSUFFICIENT_EVIDENCE

    @staticmethod
    def build_result(
        failure_type,
        subtype: str,
        confidence_score: float,
        evidence: list[Evidence],
        reason: str,
        detection_stage: str,
        fix_direction: str | None = None,
        secondary_evidence: DetectionResult | None = None,
    ) -> DetectionResult:
        """Shared constructor helper so every detector builds its final
        DetectionResult the same way, instead of each one duplicating this
        assembly logic."""
        return DetectionResult(
            failure_type=failure_type,
            subtype=subtype,
            confidence_score=confidence_score,
            confidence_band=BaseDetector.confidence_to_band(confidence_score),
            evidence=evidence,
            reason=reason,
            fix_direction=fix_direction,
            detection_stage=detection_stage,
            secondary_evidence=secondary_evidence,
        )
