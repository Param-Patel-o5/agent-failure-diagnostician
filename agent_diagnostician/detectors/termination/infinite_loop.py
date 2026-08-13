# Infinite loop failure detector
# detectors/termination/infinite_loop.py
# Detects when an agent repeats the same tool call pattern excessively.
#
# Loop detection: same tool called ≥3 times AND represents ≥40% of total steps
# Subtype analysis: linear funnel priority
#   1. DEGRADED_SUCCESS     (task completed but via wasteful loops)
#   2. EXACT_REPETITION     (same tool+input called repeatedly)
#   3. STUCK_ON_FAILURE     (tool failing, agent keeps retrying)
#   4. REASONING_LOOP       (agent's thinking stuck in circle)
#   5. INSUFFICIENT_EVIDENCE (loop suspected but can't classify)
#
# Pattern: Linear funnel with stop-on-first-hit subtype selection.

from typing import Any

from agent_diagnostician.detectors.base import BaseDetector
from agent_diagnostician.models.trace import AgentTrace, Step
from agent_diagnostician.models.result import DetectionResult, Evidence
from agent_diagnostician.models.enums import (
    FailureType,
    InfiniteLoopSubtype,
    TRACE_SUCCESS_STATUSES,
    STEP_FAILURE_STATUSES,
)
from agent_diagnostician.config import (
    INFINITE_LOOP_ERROR_RATIO_THRESHOLD,
    INFINITE_LOOP_INPUT_SIMILARITY_THRESHOLD,
    INFINITE_LOOP_MIN_CALLS,
    INFINITE_LOOP_MIN_RATIO,
    INFINITE_LOOP_THOUGHT_SIMILARITY_THRESHOLD,
)
from agent_diagnostician.analysis.llm import LLMJudge, MockLLMJudge
from agent_diagnostician.analysis.embeddings import EmbeddingMatcher
from agent_diagnostician.utils.text import fuzzy_match


class InfiniteLoopDetector(BaseDetector):
    """Detects infinite loop failures in agent execution traces.
    
    Loop detection: same tool called ≥3 times AND represents ≥40% of steps
    Subtype analysis: linear funnel priority
        1. DEGRADED_SUCCESS     (task completed but via wasteful loops)
        2. EXACT_REPETITION     (same tool+input called repeatedly)
        3. STUCK_ON_FAILURE     (tool failing, agent keeps retrying)
        4. REASONING_LOOP       (agent's thinking stuck in circle)
        5. INSUFFICIENT_EVIDENCE (loop suspected but can't classify)
    """

    # Minimum thresholds for loop detection (see config.py)
    MIN_LOOP_CALLS = INFINITE_LOOP_MIN_CALLS
    MIN_LOOP_RATIO = INFINITE_LOOP_MIN_RATIO
    INPUT_SIMILARITY_THRESHOLD = INFINITE_LOOP_INPUT_SIMILARITY_THRESHOLD
    ERROR_RATIO_THRESHOLD = INFINITE_LOOP_ERROR_RATIO_THRESHOLD
    THOUGHT_SIMILARITY_THRESHOLD = INFINITE_LOOP_THOUGHT_SIMILARITY_THRESHOLD

    def __init__(
        self,
        llm_judge: LLMJudge | None = None,
        embedding_matcher: EmbeddingMatcher | None = None,
    ):
        """Initialize detector with optional LLM judge and EmbeddingMatcher.
        
        Args:
            llm_judge: Optional LLM judge implementation. Defaults to MockLLMJudge.
            embedding_matcher: Shared embedding matcher for lazy-loaded analysis.
        """
        self.llm_judge = llm_judge or MockLLMJudge()
        self._embedding_matcher = embedding_matcher

    @property
    def embedding_matcher(self) -> EmbeddingMatcher:
        """Lazy-loaded EmbeddingMatcher for thought similarity analysis."""
        if self._embedding_matcher is None:
            self._embedding_matcher = EmbeddingMatcher()
        return self._embedding_matcher

    def detect(self, trace: AgentTrace) -> DetectionResult:
        """Run infinite loop detection pipeline on a trace.
        
        Args:
            trace: AgentTrace to analyze
            
        Returns:
            DetectionResult with infinite loop classification
        """
        if not trace.steps:
            return self.build_result(
                failure_type=FailureType.INFINITE_LOOP,
                subtype=InfiniteLoopSubtype.INSUFFICIENT_EVIDENCE.value,
                confidence_score=0.0,
                evidence=[],
                reason="No steps found in trace to analyze",
                detection_stage="none",
                fix_direction="Provide a trace with at least one tool invocation step",
            )

        # Step 1: Detect if any tool is being called in a loop pattern
        loop_info = self._detect_loops(trace)
        
        if loop_info is None:
            return self.build_result(
                failure_type=FailureType.INFINITE_LOOP,
                subtype=InfiniteLoopSubtype.NO_INFINITE_LOOP.value,
                confidence_score=1.0,
                evidence=[],
                reason="No infinite loop pattern detected",
                detection_stage="1 - Loop Detection",
                fix_direction="No fix required — agent used tools efficiently",
            )

        # Step 2: Collect evidence signals for the detected loop
        evidence_dict = self._collect_evidence(trace, loop_info)
        
        # Step 3: Decide subtype using linear funnel
        subtype, confidence, reason = self._decide_subtype(evidence_dict, loop_info)

        # Build evidence list for final result
        evidence = [
            Evidence(
                detection_stage="1 - Loop Detection",
                signal="tool_repetition",
                confidence_contribution=0.30,
                explanation=f"Tool '{loop_info['tool_name']}' called {loop_info['count']} out of {len(trace.steps)} steps ({loop_info['ratio']*100:.0f}%)",
            )
        ]

        if evidence_dict.get("input_similarity") is not None:
            evidence.append(
                Evidence(
                    detection_stage="2 - Input Analysis",
                    signal="input_similarity",
                    confidence_contribution=0.40,
                    explanation=f"Average input similarity across loop steps: {evidence_dict['input_similarity']:.2f}",
                )
            )

        if evidence_dict.get("error_ratio") is not None and evidence_dict["error_ratio"] > 0:
            evidence.append(
                Evidence(
                    detection_stage="2 - Error Analysis",
                    signal="error_pattern",
                    confidence_contribution=0.30,
                    explanation=f"Error rate in loop steps: {evidence_dict['error_ratio']*100:.0f}%",
                )
            )

        if evidence_dict.get("thought_similarity") is not None:
            evidence.append(
                Evidence(
                    detection_stage="2 - Thought Analysis",
                    signal="thought_repetition",
                    confidence_contribution=0.20,
                    explanation=f"Average thought similarity across loop steps: {evidence_dict['thought_similarity']:.2f}",
                )
            )

        return self.build_result(
            failure_type=FailureType.INFINITE_LOOP,
            subtype=subtype,
            confidence_score=confidence,
            evidence=evidence,
            reason=reason,
            detection_stage="3 - Subtype Decision",
            fix_direction=self._get_fix_direction(subtype, evidence_dict),
        )

    def _detect_loops(self, trace: AgentTrace) -> dict | None:
        """Detect if any tool is called in a loop pattern.
        
        Algorithm:
        1. Count occurrences of each tool across all steps
        2. For tools called ≥ MIN_LOOP_CALLS times, calculate ratio
        3. Return info for first tool meeting both thresholds, or None
        
        Returns:
            Dict with tool_name, step_indices, count, ratio or None
        """
        # Count tool occurrences and track step indices
        tool_counts = {}
        tool_indices = {}
        
        for step in trace.steps:
            # Validate step has required fields
            if not hasattr(step, 'step_index') or step.step_index is None:
                continue  # Skip steps without valid index
            if not hasattr(step, 'tool_name') or not step.tool_name:
                continue  # Skip steps without tool name
                
            tool_name = step.tool_name
            if tool_name not in tool_counts:
                tool_counts[tool_name] = 0
                tool_indices[tool_name] = []
            tool_counts[tool_name] += 1
            tool_indices[tool_name].append(step.step_index)
        
        # Find first tool meeting both thresholds
        total_steps = len(trace.steps)
        if total_steps == 0:
            return None
            
        for tool_name, count in tool_counts.items():
            if count >= self.MIN_LOOP_CALLS:
                ratio = count / total_steps
                if ratio >= self.MIN_LOOP_RATIO:
                    loop_steps = [
                        s for s in trace.steps
                        if hasattr(s, "tool_name") and s.tool_name == tool_name
                    ]
                    input_sim = self._compute_input_similarity(loop_steps)
                    error_ratio = self._compute_error_ratio(loop_steps)
                    thought_sim = self._compute_thought_similarity_from_steps(loop_steps)
                    if (
                        input_sim >= self.INPUT_SIMILARITY_THRESHOLD
                        or error_ratio >= self.ERROR_RATIO_THRESHOLD
                        or (
                            thought_sim is not None
                            and thought_sim >= self.THOUGHT_SIMILARITY_THRESHOLD
                        )
                    ):
                        return {
                            "tool_name": tool_name,
                            "step_indices": tool_indices[tool_name],
                            "count": count,
                            "ratio": ratio,
                        }
        
        return None

    @staticmethod
    def _step_failed(step: Step) -> bool:
        if hasattr(step, "step_status") and step.step_status in STEP_FAILURE_STATUSES:
            return True
        if hasattr(step, "error_message") and step.error_message is not None:
            return True
        if hasattr(step, "tool_output") and isinstance(step.tool_output, dict):
            if "error" in step.tool_output:
                return True
        return False

    def _compute_error_ratio(self, loop_steps: list[Step]) -> float:
        if not loop_steps:
            return 0.0
        error_count = sum(1 for step in loop_steps if self._step_failed(step))
        return error_count / len(loop_steps)

    def _collect_evidence(self, trace: AgentTrace, loop_info: dict) -> dict:
        """Collect all evidence signals for the detected loop.
        
        Args:
            trace: Full trace for context
            loop_info: Dict from _detect_loops with loop details
            
        Returns:
            Dict with input_similarity, error_ratio, thought_similarity, task_completed
        """
        # Get all steps that are part of the loop
        loop_steps = []
        for step in trace.steps:
            # Validate step index exists before comparison
            if (hasattr(step, 'step_index') and 
                step.step_index is not None and 
                step.step_index in loop_info["step_indices"]):
                loop_steps.append(step)
        
        if not loop_steps:
            # Fallback: if step_index matching fails, use tool_name matching
            for step in trace.steps:
                if hasattr(step, 'tool_name') and step.tool_name == loop_info["tool_name"]:
                    loop_steps.append(step)
        
        # A. Tool Input Similarity
        input_similarity = self._compute_input_similarity(loop_steps)
        
        # B. Error Analysis
        error_ratio = self._compute_error_ratio(loop_steps)
        
        # C. Thought Similarity (Tier 3, only compute if needed and available)
        thoughts = []
        for step in loop_steps:
            if (hasattr(step, 'thought') and 
                step.thought is not None and 
                isinstance(step.thought, str) and 
                step.thought.strip()):
                thoughts.append(step.thought.strip())
        
        thought_similarity = None
        if len(thoughts) >= 2:
            thought_similarity = self._compute_thought_similarity(thoughts)
        
        # D. Task Completion
        task_completed = (
            hasattr(trace, 'status') and trace.status in TRACE_SUCCESS_STATUSES and
            hasattr(trace, 'final_output') and trace.final_output is not None
        )
        
        return {
            "input_similarity": input_similarity,
            "error_ratio": error_ratio,
            "thought_similarity": thought_similarity,
            "task_completed": task_completed,
        }

    def _compute_input_similarity(self, loop_steps: list[Step]) -> float:
        """Compute average fuzzy similarity between tool_inputs in loop steps.
        
        Converts each tool_input dict to string and uses fuzzy_match().
        
        Args:
            loop_steps: Steps that are part of the detected loop
            
        Returns:
            Average similarity score (0.0-1.0)
        """
        if len(loop_steps) < 2:
            return 1.0  # Single call = identical to itself
        
        # Convert tool_inputs to value-focused strings (ignore shared key names).
        input_strings = []
        for step in loop_steps:
            if hasattr(step, "tool_input") and step.tool_input is not None:
                if isinstance(step.tool_input, dict):
                    input_strings.append(
                        " ".join(str(v) for v in step.tool_input.values())
                    )
                else:
                    input_strings.append(str(step.tool_input))
            else:
                input_strings.append("")
        
        if len(input_strings) < 2:
            return 1.0  # Not enough valid inputs to compare
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(input_strings)):
            for j in range(i + 1, len(input_strings)):
                try:
                    sim = fuzzy_match(input_strings[i], input_strings[j])
                    similarities.append(sim)
                except Exception:
                    # If fuzzy_match fails, assume no similarity
                    similarities.append(0.0)
        
        return sum(similarities) / len(similarities) if similarities else 1.0

    def _compute_thought_similarity_from_steps(self, loop_steps: list[Step]) -> float | None:
        thoughts = []
        for step in loop_steps:
            if hasattr(step, "thought") and step.thought and str(step.thought).strip():
                thoughts.append(str(step.thought).strip())
        return self._compute_thought_similarity(thoughts)

    def _compute_thought_similarity(self, thoughts: list[str]) -> float:
        """Compute average semantic similarity between thoughts.
        
        Uses EmbeddingMatcher for semantic comparison with lazy initialization.
        
        Args:
            thoughts: List of thought strings from loop steps
            
        Returns:
            Average similarity score (0.0-1.0) or None if embedding failed
        """
        if len(thoughts) < 2:
            return None
        
        try:
            similarities = []
            
            for i in range(len(thoughts)):
                for j in range(i + 1, len(thoughts)):
                    try:
                        sim = self.embedding_matcher.similarity(thoughts[i], thoughts[j])
                        similarities.append(sim)
                    except Exception as e:
                        # Log specific embedding comparison failure but continue
                        # In production, you might want to log this: f"Failed to compare thoughts {i} and {j}: {e}"
                        continue
            
            if not similarities:
                return None  # All comparisons failed
            
            return sum(similarities) / len(similarities)
            
        except Exception:
            # If EmbeddingMatcher initialization or operation fails completely,
            # return None (signal unavailable) rather than crashing
            return None

    def _decide_subtype(
        self, evidence: dict, loop_info: dict
    ) -> tuple[str, float, str]:
        """Linear funnel to determine subtype and confidence.
        
        Priority order (stop on first match):
        1. DEGRADED_SUCCESS     — task completed but via loops
        2. STUCK_ON_FAILURE     — majority of loop steps failed (even if inputs match)
        3. EXACT_REPETITION     — inputs nearly identical, not failure-dominated
        4. REASONING_LOOP       — thoughts are repetitive
        5. INSUFFICIENT_EVIDENCE — loop suspected but unclear
        
        Args:
            evidence: Dict from _collect_evidence
            loop_info: Dict from _detect_loops
            
        Returns:
            (subtype, confidence, reason) tuple
        """
        # 1. DEGRADED_SUCCESS — completed via wasteful identical retries
        input_sim = evidence.get("input_similarity")
        error_ratio = evidence.get("error_ratio")
        if (
            evidence.get("task_completed")
            and input_sim is not None
            and input_sim >= self.INPUT_SIMILARITY_THRESHOLD
        ):
            return (
                InfiniteLoopSubtype.DEGRADED_SUCCESS.value,
                0.80,
                "Task completed but agent used repetitive identical tool calls, suggesting a loop",
            )

        # 2. STUCK_ON_FAILURE — failure-dominated loop monopolizing the trace
        loop_ratio = loop_info.get("ratio", 0)
        if (
            error_ratio is not None
            and error_ratio >= self.ERROR_RATIO_THRESHOLD
            and loop_ratio >= 0.75
        ):
            return (
                InfiniteLoopSubtype.STUCK_ON_FAILURE.value,
                0.75,
                "Agent kept retrying the same tool despite repeated failures",
            )

        # 3. EXACT_REPETITION — identical inputs without failure-dominated pattern
        if input_sim is not None and input_sim >= self.INPUT_SIMILARITY_THRESHOLD:
            return (
                InfiniteLoopSubtype.EXACT_REPETITION.value,
                0.90,
                "Agent made identical tool calls repeatedly with near-identical inputs",
            )

        # 5. Legitimate multi-call pattern (varied inputs, task completed)
        if evidence.get("task_completed"):
            return (
                InfiniteLoopSubtype.NO_INFINITE_LOOP.value,
                0.75,
                "Repeated tool use appears intentional — inputs varied and task completed",
            )

        # 5. REASONING_LOOP (thoughts are repetitive)
        thought_sim = evidence.get("thought_similarity")
        if thought_sim is not None and thought_sim >= self.THOUGHT_SIMILARITY_THRESHOLD:
            return (
                InfiniteLoopSubtype.REASONING_LOOP.value,
                0.65,
                "Agent's reasoning showed repetitive patterns across loop steps",
            )

        # 6. INSUFFICIENT_EVIDENCE
        return (
            InfiniteLoopSubtype.INSUFFICIENT_EVIDENCE.value,
            0.30,
            "Loop pattern detected but insufficient evidence to classify subtype",
        )

    def _get_fix_direction(self, subtype: str, evidence: dict) -> str:
        """Return actionable fix direction based on subtype."""
        if subtype == InfiniteLoopSubtype.DEGRADED_SUCCESS.value:
            return "Review if all tool calls were necessary; consider task decomposition or early stopping"
        elif subtype == InfiniteLoopSubtype.EXACT_REPETITION.value:
            return "Update agent's prompt to emphasize variation in approach or add termination criteria"
        elif subtype == InfiniteLoopSubtype.STUCK_ON_FAILURE.value:
            return "Fix the failing tool or update agent's error recovery strategy"
        elif subtype == InfiniteLoopSubtype.REASONING_LOOP.value:
            return "Revise agent's reasoning prompt to encourage exploration of alternatives"
        else:
            return "Review loop pattern and identify root cause; consider adding explicit loop limits"
