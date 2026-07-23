# Agent diagnostic classifier module
# classifier.py
# Runs every detector against a trace and returns the single most
# confident diagnosis. Does NO analysis itself -- only aggregates
# DetectionResult objects from detectors and applies tiebreaking logic.
#
# Priority order (used as tiebreaker when confidence scores are equal):
#   1. Tool Use Failure         (most specific, step-level, deterministic checks)
#   2. Goal Satisfaction Failure (outcome-level, deterministic constraint checks)
#   3. Context Loss             (execution-level, multi-step pattern)
#   4. Token Exhaustion         (execution-level, metric-based)
#   5. Premature Termination    (termination-level)
#   6. Infinite Loop            (termination-level, pattern across steps)
#   7. Hallucination            (last — most LLM-dependent, least deterministic)
#
# Planning failures outrank execution and termination because they are
# root causes -- execution/termination failures are often consequences.

from agent_diagnostician.models.trace import AgentTrace
from agent_diagnostician.models.result import DetectionResult
from agent_diagnostician.models.enums import FailureType, ConfidenceBand
from agent_diagnostician.analysis.llm_judge import LLMJudge, MockLLMJudge
from agent_diagnostician.detectors.planning.tool_use import ToolUseDetector
from agent_diagnostician.detectors.planning.goal_failure import GoalFailureDetector
from agent_diagnostician.detectors.planning.hallucination import HallucinationDetector

# Priority order — lower index = higher priority when scores are tied
DETECTOR_PRIORITY = [
    FailureType.TOOL_USE_FAILURE,
    FailureType.GOAL_SATISFACTION_FAILURE,
    FailureType.CONTEXT_LOSS,
    FailureType.TOKEN_EXHAUSTION,
    FailureType.PREMATURE_TERMINATION,
    FailureType.INFINITE_LOOP,
    FailureType.HALLUCINATION,
]


class Classifier:
    """Runs all detectors and returns the single most confident diagnosis.
    
    Usage:
        classifier = Classifier()
        result = classifier.diagnose(trace)
    
    Inject a real LLMJudge for production:
        classifier = Classifier(llm_judge=GeminiLLMJudge())
    """

    def __init__(self, llm_judge: LLMJudge | None = None):
        """Initialize all detectors with the same LLM judge instance.
        One LLMJudge shared across all detectors — no duplicate model loads.
        
        Args:
            llm_judge: LLM judge implementation. Defaults to MockLLMJudge.
        """
        self.llm_judge = llm_judge or MockLLMJudge()

        # Only Tool Use and Goal Failure are implemented so far.
        # Add remaining detectors here as they are built — no other
        # changes needed in this file.
        self.detectors = [
            ToolUseDetector(llm_judge=self.llm_judge),
            GoalFailureDetector(llm_judge=self.llm_judge),
            HallucinationDetector(llm_judge=self.llm_judge),
            # ContextLossDetector(llm_judge=self.llm_judge),    # not yet built
            # TokenExhaustionDetector(llm_judge=self.llm_judge),# not yet built
            # PrematureTerminationDetector(llm_judge=self.llm_judge), # not yet built
            # InfiniteLoopDetector(llm_judge=self.llm_judge),   # not yet built
        ]

    def diagnose(self, trace: AgentTrace) -> DetectionResult:
        """Run all detectors and return the single best diagnosis.
        
        Selection logic:
        1. Run every detector.
        2. Filter out NO_FAILURE and INSUFFICIENT_EVIDENCE results.
        3. If no failures detected → return NO_FAILURE.
        4. If one failure detected → return it.
        5. If multiple failures detected → pick by highest confidence.
           Tiebreak by DETECTOR_PRIORITY order.
        
        Args:
            trace: AgentTrace to diagnose
            
        Returns:
            Single DetectionResult representing the most likely root cause
        """
        all_results = []

        for detector in self.detectors:
            result = detector.detect(trace)
            all_results.append(result)

        # Filter to only real failures
        failures = [
            r for r in all_results
            if r.subtype not in (
                "no_tool_use_failure",
                "no_goal_failure",
                "none",
                "insufficient_evidence",
            )
            and r.confidence_band != ConfidenceBand.INSUFFICIENT_EVIDENCE
        ]

        # No failures detected
        if not failures:
            return self._no_failure_result()

        # One failure — return it directly
        if len(failures) == 1:
            return failures[0]

        # Multiple failures — pick by highest confidence, tiebreak by priority
        return self._select_primary(failures)

    def _select_primary(self, failures: list[DetectionResult]) -> DetectionResult:
        """Pick primary failure from multiple candidates.
        
        Sort by: confidence score descending, then priority order ascending.
        """
        def sort_key(result: DetectionResult):
            priority = DETECTOR_PRIORITY.index(result.failure_type) \
                if result.failure_type in DETECTOR_PRIORITY \
                else len(DETECTOR_PRIORITY)
            # Negate confidence so higher confidence sorts first
            return (-result.confidence_score, priority)

        failures.sort(key=sort_key)
        return failures[0]

    def _no_failure_result(self) -> DetectionResult:
        """Build a clean NO_FAILURE result when no detector fired."""
        from agent_diagnostician.models.result import Evidence

        return DetectionResult(
            failure_type=FailureType.NONE,
            subtype="no_failure",
            confidence_score=1.0,
            confidence_band=ConfidenceBand.CONFIRMED,
            evidence=[],
            reason="No failure detected across all detectors",
            fix_direction="No fix required",
            detection_stage="none",
            secondary_evidence=None,
        )