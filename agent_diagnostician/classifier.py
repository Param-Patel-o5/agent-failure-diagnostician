# Agent diagnostic classifier module
# classifier.py
# Runs selected detectors against a trace and returns the single most
# confident diagnosis. Does NO analysis itself -- only aggregates
# DetectionResult objects from detectors and applies tiebreaking logic.
#
# Users can control which detectors run via enabled_detectors parameter:
#   - Pass a list of FailureType enums to run only those detectors
#   - Pass None (default) to run all detectors defined in DEFAULT_ENABLED_DETECTORS
#   - This lets users skip expensive LLM-dependent detectors when not needed

from typing import Optional, Set, List
from agent_diagnostician.models.trace import AgentTrace
from agent_diagnostician.models.result import DetectionResult, Evidence
from agent_diagnostician.models.enums import (
    FailureType,
    ConfidenceBand,
    ClassifierSubtype,
    NO_FAILURE_SUBTYPE_VALUES,
    INSUFFICIENT_EVIDENCE_SUBTYPE,
    EvidenceSignal,
)
from agent_diagnostician.analysis.llm import LLMJudge, MockLLMJudge
from agent_diagnostician.analysis.embeddings import EmbeddingMatcher
from agent_diagnostician.config import DEFAULT_ENABLED_DETECTORS, DETECTOR_MAPPING

# Import detectors
from agent_diagnostician.detectors.planning.tool_use import ToolUseDetector
from agent_diagnostician.detectors.planning.goal_failure import GoalFailureDetector
from agent_diagnostician.detectors.planning.hallucination import HallucinationDetector
from agent_diagnostician.detectors.execution.token_exhaustion import TokenExhaustionDetector
from agent_diagnostician.detectors.execution.context_loss import ContextLossDetector
from agent_diagnostician.detectors.termination.infinite_loop import InfiniteLoopDetector
from agent_diagnostician.detectors.termination.premature_termination import PrematureTerminationDetector

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


# Re-export for backwards compatibility (classifier module level).
NO_FAILURE_SUBTYPES = NO_FAILURE_SUBTYPE_VALUES

class Classifier:
    """Runs selected detectors and returns the single most confident diagnosis.
    
    Usage with default detectors:
        classifier = Classifier()
        result = classifier.diagnose(trace)
    
    Usage with selective detectors:
        classifier = Classifier(enabled_detectors=[FailureType.TOKEN_EXHAUSTION])
        result = classifier.diagnose(trace)
    
    Inject a real LLM judge for production:
        from agent_diagnostician.analysis.llm import create_llm_judge_from_env
        classifier = Classifier(llm_judge=create_llm_judge_from_env())
    """

    def __init__(
        self,
        llm_judge: LLMJudge | None = None,
        enabled_detectors: Optional[List[FailureType]] = None,
        embedding_matcher: EmbeddingMatcher | None = None,
    ):
        """Initialize classifier with optional detector selection.
        
        Args:
            llm_judge: LLM judge implementation. Defaults to MockLLMJudge.
            enabled_detectors: Optional list of FailureType enums to run.
                              If None, uses DEFAULT_ENABLED_DETECTORS from config.
            embedding_matcher: Shared embedding model for detectors that need it.
                              Created once if not provided.
        
        Raises:
            ValueError: If any detector type in enabled_detectors is unknown.
        """
        self.llm_judge = llm_judge or MockLLMJudge()
        self.embedding_matcher = embedding_matcher or EmbeddingMatcher()
        self.enabled_detectors = enabled_detectors or DEFAULT_ENABLED_DETECTORS

        # Validate that all requested detectors exist
        for detector_type in self.enabled_detectors:
            if detector_type not in DETECTOR_MAPPING:
                raise ValueError(
                    f"Unknown detector type: {detector_type}. "
                    f"Valid options: {list(DETECTOR_MAPPING.keys())}"
                )

        # Instantiate only the enabled detectors
        self.detectors = self._build_detectors()

        # Track which detectors ran (for debugging/logging)
        self.ran_detectors: Set[str] = set()

    def _build_detectors(self) -> List:
        """Build detector instances based on enabled_detectors list."""
        detectors = []

        if FailureType.TOOL_USE_FAILURE in self.enabled_detectors:
            detectors.append(
                ToolUseDetector(
                    llm_judge=self.llm_judge,
                    embedding_matcher=self.embedding_matcher,
                )
            )

        if FailureType.GOAL_SATISFACTION_FAILURE in self.enabled_detectors:
            detectors.append(
                GoalFailureDetector(
                    llm_judge=self.llm_judge,
                    embedding_matcher=self.embedding_matcher,
                )
            )

        if FailureType.HALLUCINATION in self.enabled_detectors:
            detectors.append(HallucinationDetector(llm_judge=self.llm_judge))

        if FailureType.TOKEN_EXHAUSTION in self.enabled_detectors:
            detectors.append(TokenExhaustionDetector())

        if FailureType.INFINITE_LOOP in self.enabled_detectors:
            detectors.append(
                InfiniteLoopDetector(
                    llm_judge=self.llm_judge,
                    embedding_matcher=self.embedding_matcher,
                )
            )

        if FailureType.CONTEXT_LOSS in self.enabled_detectors:
            detectors.append(
                ContextLossDetector(
                    llm_judge=self.llm_judge,
                    embedding_matcher=self.embedding_matcher,
                )
            )

        if FailureType.PREMATURE_TERMINATION in self.enabled_detectors:
            detectors.append(
                PrematureTerminationDetector(
                    llm_judge=self.llm_judge,
                    embedding_matcher=self.embedding_matcher,
                )
            )

        return detectors

    def diagnose(self, trace: AgentTrace) -> DetectionResult:
        """Run all enabled detectors and return the single best diagnosis.
        
        Selection logic:
        1. Run every enabled detector.
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
        self.ran_detectors.clear()

        for detector in self.detectors:
            detector_name = detector.__class__.__name__
            result = detector.detect(trace)
            all_results.append(result)
            self.ran_detectors.add(detector_name)

        # Filter to real failures only — exclude no-failure and insufficient-evidence
        failures = [
            r for r in all_results
            if r.subtype not in NO_FAILURE_SUBTYPES
            and r.subtype != INSUFFICIENT_EVIDENCE_SUBTYPE
        ]

        # Store all failures for multi-failure summary
        self.all_failures = failures

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

    def get_all_failures(self) -> list:
        """Get all detected failures, not just the primary one.
        
        Returns:
            List of all DetectionResults that indicated failures
        """
        return getattr(self, 'all_failures', [])

    def get_detector_status(self) -> dict:
        """Get status of which detectors ran vs were skipped.
        
        Returns:
            Dict with 'ran' and 'skipped' lists of detector names
        """
        # Get all possible detector names from config mapping
        all_possible = {v for v in DETECTOR_MAPPING.values()}
        
        ran = list(self.ran_detectors)
        skipped = list(all_possible - self.ran_detectors)
        
        return {
            "ran": ran,
            "skipped": skipped,
        }

    def _no_failure_result(self) -> DetectionResult:
        """Build a clean NO_FAILURE result when no detector fired."""
        evidence = []
        if self.ran_detectors:
            evidence.append(
                Evidence(
                    detection_stage="all_detectors_passed",
                    signal=EvidenceSignal.ALL_ENABLED_DETECTORS.value,
                    confidence_contribution=1.0,
                    explanation=f"All {len(self.ran_detectors)} enabled detectors passed: {', '.join(sorted(self.ran_detectors))}",
                )
            )

        return DetectionResult(
            failure_type=FailureType.NONE,
            subtype=ClassifierSubtype.NO_FAILURE.value,
            confidence_score=0.0,  # Zero confidence that there IS a failure
            confidence_band=ConfidenceBand.CONFIRMED,
            evidence=evidence,
            reason="No failure detected across all enabled detectors.",
            fix_direction=None,
            detection_stage="classifier_aggregation",
            secondary_evidence=None,
        )