# Hallucination failure detector
# detectors/planning/hallucination.py
# Detects whether the agent fabricated values in tool_input or thought
# that cannot be traced to the task or any prior tool output.
#
# Returns one of three outcomes (as plain strings):
#   "hallucination_detected"
#   "no_hallucination"  
#   "insufficient_evidence"
#
# Pipeline: grounding check + LLM judge weighted combination.
# Neither signal alone is sufficient — both must contribute.

from typing import Any

from agent_diagnostician.detectors.base import BaseDetector
from agent_diagnostician.models.trace import AgentTrace, Step
from agent_diagnostician.models.result import DetectionResult, Evidence
from agent_diagnostician.models.enums import FailureType, ConfidenceBand

from agent_diagnostician.analysis.grounding import GroundingAnalyzer
from agent_diagnostician.analysis.llm_judge import LLMJudge, MockLLMJudge


class HallucinationDetector(BaseDetector):
    """Detects hallucination failures in agent execution traces.
    
    Checks for fabricated values in tool_input or thought that cannot
    be traced to the task or any prior tool output.
    
    Hallucination Pipeline — grounding + LLM weighted combination.
    Neither signal alone is sufficient. Both must contribute.
    """

    def __init__(self, llm_judge: LLMJudge | None = None):
        """Initialize detector with LLM judge.
        
        Args:
            llm_judge: LLM judge implementation. If None, uses MockLLMJudge.
        """
        self.llm_judge = llm_judge or MockLLMJudge()

    def detect(self, trace: AgentTrace) -> DetectionResult:
        """Run hallucination detection pipeline on a trace.
        
        Checks every step. Returns the first step where hallucination
        is detected. If no step fires, returns no_hallucination.
        
        Args:
            trace: AgentTrace to analyze
            
        Returns:
            DetectionResult with hallucination classification
        """
        if not trace.steps:
            return self.build_result(
                failure_type=FailureType.HALLUCINATION,
                subtype="insufficient_evidence",
                confidence_score=0.0,
                evidence=[],
                reason="No steps found in trace to analyze",
                detection_stage="none",
                fix_direction="Provide a trace with at least one tool invocation step",
            )

        best_candidate = None  # Store the best insufficient evidence candidate
        best_candidate_confidence = 0.0

        for step in trace.steps:
            # Skip steps that errored — hallucination detection requires
            # analyzing what the agent intended to do, not broken steps
            if step.step_status in ("error", "failed") or step.error_message is not None:
                continue

            # Handle empty tool_input gracefully
            if not step.tool_input:
                continue

            # Run hallucination pipeline for this step
            result = self._detect_step_hallucination(trace, step)
            
            # Check thresholds
            if result.subtype == "hallucination_detected":
                return result
            elif result.subtype == "no_hallucination":
                # Continue to next step
                continue
            elif result.subtype == "insufficient_evidence":
                # Track best candidate (highest confidence insufficient evidence)
                if result.confidence_score > best_candidate_confidence:
                    best_candidate = result
                    best_candidate_confidence = result.confidence_score
                continue

        # No hallucination detected in any step
        if best_candidate is not None:
            # Return best insufficient evidence candidate
            return best_candidate
        
        # All steps passed with confidence < 0.20 → no hallucination
        return self.build_result(
            failure_type=FailureType.HALLUCINATION,
            subtype="no_hallucination",
            confidence_score=1.0,
            evidence=[],
            reason="No hallucination detected across all steps",
            detection_stage="none",
            fix_direction="No fix required — agent used traceable values",
        )

    def _detect_step_hallucination(
        self, trace: AgentTrace, step: Step
    ) -> DetectionResult:
        """Run hallucination pipeline on a single step.
        
        Pipeline:
          1. Grounding check on tool_input (always runs)
          2. Grounding check on thought (if present)
          3. LLM judge (always runs, primary signal)
          4. Weighted combination
          5. Decision based on thresholds
        
        Args:
            trace: Full trace (for context like task, prior outputs)
            step: Step to analyze
            
        Returns:
            DetectionResult for this step's hallucination check
        """
        # Collect prior outputs (all steps before current)
        prior_outputs = []
        for s in trace.steps:
            if s.step_index < step.step_index:
                prior_outputs.append(s.tool_output)

        # ───────────────────────────────────────────────────────────────────
        # Step 1 — Grounding check on tool_input
        # ───────────────────────────────────────────────────────────────────
        
        grounding_results = GroundingAnalyzer.analyze(
            step.tool_input, trace.task, prior_outputs
        )
        summary = GroundingAnalyzer.summarize(grounding_results)
        
        total_fields = summary["total_fields"]
        ungrounded_count = summary["ungrounded"]
        ungrounded_fields = summary["ungrounded_fields"]
        
        # Base grounding score
        grounding_score = 0.0
        
        grounding_evidence = []
        if total_fields > 0 and ungrounded_count > 0:
            # Per ungrounded field: +0.40 / total_fields (cap at 0.60)
            per_field_contribution = 0.40 / total_fields
            grounding_score = min(0.60, per_field_contribution * ungrounded_count)
            
            for field in ungrounded_fields:
                grounding_evidence.append(
                    Evidence(
                        detection_stage="1 - Tool Input Grounding",
                        signal="ungrounded_field",
                        confidence_contribution=per_field_contribution,
                        explanation=f"Field '{field}' contains values that cannot be traced to the task or prior tool outputs",
                    )
                )

        # ───────────────────────────────────────────────────────────────────
        # Step 2 — Grounding check on thought (only if step.thought present)
        # ───────────────────────────────────────────────────────────────────
        
        if step.thought is not None:
            # Convert thought to a fake tool_input dict for grounding analysis
            thought_as_input = {"thought_content": step.thought}
            
            thought_grounding = GroundingAnalyzer.analyze(
                thought_as_input, trace.task, prior_outputs
            )
            thought_summary = GroundingAnalyzer.summarize(thought_grounding)
            
            # If thought_content is ungrounded
            if thought_summary["ungrounded"] > 0:
                grounding_score += 0.15  # Thought fabrication bonus
                grounding_score = min(0.60, grounding_score)  # Cap at 0.60
                
                grounding_evidence.append(
                    Evidence(
                        detection_stage="2 - Thought Grounding",
                        signal="ungrounded_thought",
                        confidence_contribution=0.15,
                        explanation=f"Agent's thought contains values that cannot be traced to the task or prior tool outputs: '{step.thought[:200]}...'",
                    )
                )

        # ───────────────────────────────────────────────────────────────────
        # Step 3 — LLM judge (primary signal)
        # ───────────────────────────────────────────────────────────────────
        
        # Prepare available_tools_list
        available_tools_list = None
        if trace.available_tools is not None:
            available_tools_list = [
                {"name": t.name, "description": t.description or ""}
                for t in trace.available_tools
            ]
        
        llm_result = self.llm_judge.evaluate_hallucination(
            task=trace.task,
            tool_input=step.tool_input,
            prior_outputs=prior_outputs,
            thought=step.thought,
            available_tools=available_tools_list,
        )
        
        llm_score = llm_result.get("confidence", 0.0)
        llm_reason = llm_result.get("reason", "No reason provided")
        
        llm_evidence = [
            Evidence(
                detection_stage="3 - LLM Judge",
                signal="llm_hallucination_confidence",
                confidence_contribution=llm_score,
                explanation=f"LLM judge confidence: {llm_score:.2f} — {llm_reason}",
            )
        ]

        # ───────────────────────────────────────────────────────────────────
        # Step 4 — Weighted combination
        # ───────────────────────────────────────────────────────────────────
        
        # Weighted final: (grounding * 0.40) + (llm * 0.60)
        final_confidence = (grounding_score * 0.40) + (llm_score * 0.60)
        final_confidence = min(0.95, final_confidence)  # Cap at 0.95

        # ───────────────────────────────────────────────────────────────────
        # Step 5 — Decision based on thresholds
        # ───────────────────────────────────────────────────────────────────
        
        # THRESHOLDS:
        # >= 0.50 → HALLUCINATION_DETECTED
        # < 0.20  → NO_HALLUCINATION for this step
        # between → INSUFFICIENT_EVIDENCE (continue checking other steps)
        
        if final_confidence >= 0.50:
            # Hallucination detected
            return self.build_result(
                failure_type=FailureType.HALLUCINATION,
                subtype="hallucination_detected",
                confidence_score=final_confidence,
                evidence=grounding_evidence + llm_evidence,
                reason=llm_reason,
                detection_stage="hallucination_pipeline",
                fix_direction="Ensure all parameter values are traceable to the task or prior tool outputs",
            )
        elif final_confidence < 0.20:
            # No hallucination for this step
            return self.build_result(
                failure_type=FailureType.HALLUCINATION,
                subtype="no_hallucination",
                confidence_score=1.0 - final_confidence,  # Invert: high confidence of no hallucination
                evidence=[],
                reason="No hallucination signals detected in this step",
                detection_stage="hallucination_pipeline",
                fix_direction="No fix required — values appear traceable",
            )
        else:
            # Insufficient evidence — neither strong hallucination nor strong absence
            return self.build_result(
                failure_type=FailureType.HALLUCINATION,
                subtype="insufficient_evidence",
                confidence_score=final_confidence,
                evidence=grounding_evidence + llm_evidence,
                reason="Insufficient evidence to confirm or rule out hallucination",
                detection_stage="hallucination_pipeline",
                fix_direction="Gather more context or provide clearer task requirements",
            )