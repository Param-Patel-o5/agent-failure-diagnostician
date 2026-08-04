# Premature termination failure detector
# detectors/termination/premature_termination.py
# Detects when an agent declares success but hasn't completed all task requirements.
#
# Premature Termination Pipeline
# Evidence sources: task-output similarity, task-inputs coverage,
# failed step ratio, LLM fallback (only when 1+2 inconclusive)

from typing import Any

from agent_diagnostician.detectors.base import BaseDetector
from agent_diagnostician.models.trace import AgentTrace, Step
from agent_diagnostician.models.result import DetectionResult, Evidence
from agent_diagnostician.models.enums import FailureType
from agent_diagnostician.analysis.embeddings import EmbeddingMatcher
from agent_diagnostician.analysis.llm_judge import LLMJudge, MockLLMJudge


class PrematureTerminationDetector(BaseDetector):
    """Detects premature termination failures in agent execution traces.
    
    An agent terminates early when it declares success (status: "success",
    final_output present) but the actual work done is incomplete relative
    to what was asked in the task.
    """

    # Similarity thresholds
    TASK_OUTPUT_SIM_THRESHOLD = 0.45  # Below this = low similarity signal
    TASK_INPUTS_SIM_THRESHOLD = 0.45  # Below this = low coverage signal
    MIN_CONFIDENCE_THRESHOLD = 0.35   # Minimum confidence to report failure
    MAX_CONFIDENCE = 0.92             # Cap for final confidence score

    def __init__(self, llm_judge: LLMJudge | None = None):
        """Initialize detector with LLM judge.
        
        Args:
            llm_judge: Optional LLM judge implementation. Defaults to MockLLMJudge.
        """
        self.llm_judge = llm_judge or MockLLMJudge()
        # Instantiate EmbeddingMatcher once for all similarity checks
        self.embeddings = EmbeddingMatcher()

    def detect(self, trace: AgentTrace) -> DetectionResult:
        """Run premature termination detection pipeline.
        
        Args:
            trace: AgentTrace to analyze
            
        Returns:
            DetectionResult with premature termination classification
        """
        # Handle edge case: no steps to analyze
        if not trace.steps:
            return self.build_result(
                failure_type=FailureType.PREMATURE_TERMINATION,
                subtype="no_premature_termination",
                confidence_score=0.0,
                evidence=[],
                reason="No steps to analyze",
                detection_stage="premature_termination_pipeline",
                fix_direction=None,
            )

        evidence = []
        
        # Step 1 — Task vs Final Output similarity (always runs)
        final_output_str = str(trace.final_output) if trace.final_output else ""
        task_output_sim = self.embeddings.similarity(trace.task, final_output_str)
        
        output_signal_fired = False
        output_confidence = 0.0
        if task_output_sim < self.TASK_OUTPUT_SIM_THRESHOLD:
            output_signal_fired = True
            output_confidence = 0.50
            evidence.append(
                Evidence(
                    detection_stage="1 - Task vs Final Output",
                    signal="low_task_output_similarity",
                    confidence_contribution=0.50,
                    explanation=f"Low semantic similarity between task and final output ({task_output_sim:.2f}) — output may not address all task requirements",
                )
            )

        # Step 2 — Task vs Tool Inputs (always runs)
        tool_inputs_combined = " ".join(str(s.tool_input) if s.tool_input else "{}" for s in trace.steps)
        task_inputs_sim = self.embeddings.similarity(trace.task, tool_inputs_combined)
        
        inputs_signal_fired = False
        inputs_confidence = 0.0
        if task_inputs_sim < self.TASK_INPUTS_SIM_THRESHOLD:
            inputs_signal_fired = True
            inputs_confidence = 0.45
            evidence.append(
                Evidence(
                    detection_stage="2 - Task vs Tool Inputs",
                    signal="low_task_inputs_similarity",
                    confidence_contribution=0.45,
                    explanation=f"Tool inputs across all steps show low coverage of task requirements (similarity={task_inputs_sim:.2f})",
                )
            )

        # Step 3 — Tool Outputs success check (always runs)
        failed_steps = []
        for s in trace.steps:
            # Check step status
            if hasattr(s, 'step_status') and s.step_status in ("error", "failed"):
                failed_steps.append(s)
                continue
            
            # Check error_message field
            if hasattr(s, 'error_message') and s.error_message is not None:
                failed_steps.append(s)
                continue
            
            # Check tool_output for error dict
            if hasattr(s, 'tool_output') and isinstance(s.tool_output, dict) and "error" in s.tool_output:
                failed_steps.append(s)
                continue
        
        failed_confidence = 0.0
        if failed_steps:
            failed_ratio = len(failed_steps) / max(1, len(trace.steps))
            failed_confidence = min(0.40, failed_ratio * 0.60)
            evidence.append(
                Evidence(
                    detection_stage="3 - Tool Output Success Check",
                    signal="failed_steps_detected",
                    confidence_contribution=failed_confidence,
                    explanation=f"{len(failed_steps)} of {len(trace.steps)} steps failed — agent may have terminated after failures without retrying or completing remaining requirements",
                )
            )

        # Step 4 — LLM fallback (only runs when Steps 1 and 2 are both inconclusive)
        llm_confidence = 0.0
        if not output_signal_fired and not inputs_signal_fired:
            # Prepare steps list for LLM
            steps_list = [
                {
                    "step_index": s.step_index,
                    "tool_name": s.tool_name,
                    "tool_input": s.tool_input if s.tool_input else {},
                    "tool_output": s.tool_output,
                }
                for s in trace.steps
            ]
            
            # Get last thought from any step
            last_thought = None
            for s in reversed(trace.steps):
                if hasattr(s, 'thought') and s.thought is not None:
                    last_thought = s.thought
                    break
            
            # Compute embedding score for LLM context
            task_output_sim_for_llm = self.embeddings.similarity(trace.task, final_output_str)
            
            # Run LLM judge
            llm_result = self.llm_judge.evaluate_goal_alignment(
                task=trace.task,
                final_output=trace.final_output,
                steps=steps_list,
                thought=last_thought,
                embedding_score=task_output_sim_for_llm,
            )
            
            if llm_result.get("verdict") == "misinterpreted":
                llm_confidence = 0.55 + llm_result.get("confidence", 0) * 0.20
                evidence.append(
                    Evidence(
                        detection_stage="4 - LLM Fallback",
                        signal="llm_incomplete_verdict",
                        confidence_contribution=llm_confidence,
                        explanation=f"LLM judge determined task was not fully completed: {llm_result.get('reason', 'no reason')}",
                    )
                )
            elif llm_result.get("verdict") == "uncertain":
                llm_confidence = 0.10
                evidence.append(
                    Evidence(
                        detection_stage="4 - LLM Fallback",
                        signal="llm_uncertain_verdict",
                        confidence_contribution=0.10,
                        explanation="LLM judge was uncertain about task completion",
                    )
                )

        # Step 5 — Combine and decide
        final_confidence = output_confidence + inputs_confidence + failed_confidence + llm_confidence
        final_confidence = min(final_confidence, self.MAX_CONFIDENCE)

        if final_confidence >= self.MIN_CONFIDENCE_THRESHOLD:
            return self.build_result(
                failure_type=FailureType.PREMATURE_TERMINATION,
                subtype="premature_termination_detected",
                confidence_score=final_confidence,
                evidence=evidence,
                reason="Agent terminated before completing all task requirements",
                detection_stage="premature_termination_pipeline",
                fix_direction="Ensure agent completes all stated requirements before declaring task complete",
            )
        else:
            return self.build_result(
                failure_type=FailureType.PREMATURE_TERMINATION,
                subtype="no_premature_termination",
                confidence_score=1.0 - final_confidence,
                evidence=evidence,
                reason="No premature termination detected",
                detection_stage="premature_termination_pipeline",
                fix_direction=None,
            )
