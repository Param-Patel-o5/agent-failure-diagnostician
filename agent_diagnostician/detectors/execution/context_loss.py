# Context loss failure detector
# detectors/execution/context_loss.py
# Detects when an agent drops or contradicts information established in prior steps.
#
# Context Loss Pipeline
# Checks from step 3 onwards — insufficient context before that.
# Evidence sources: grounding check on tool_input vs running context,
# thought contradiction via semantic similarity, LLM fallback.
# Context Loss differs from Hallucination: agent HAD the value earlier
# but dropped or contradicted it — not invented from nothing.

from typing import Any

from agent_diagnostician.detectors.base import BaseDetector
from agent_diagnostician.models.trace import AgentTrace, Step
from agent_diagnostician.models.result import DetectionResult, Evidence
from agent_diagnostician.models.enums import FailureType
from agent_diagnostician.analysis.embeddings import EmbeddingMatcher
from agent_diagnostician.analysis.grounding import GroundingAnalyzer
from agent_diagnostician.analysis.llm_judge import LLMJudge, MockLLMJudge


class ContextLossDetector(BaseDetector):
    """Detects context loss failures in agent execution traces.
    
    Context loss occurs when an agent drops or contradicts information
    that was established in prior steps. Key distinction from hallucination:
    - Hallucination = inventing a value from nothing
    - Context Loss = having a value earlier, then dropping or contradicting it
    """

    # Thresholds
    LOW_SIMILARITY_THRESHOLD = 0.40   # Thought vs prior context
    MIN_CONFIDENCE_THRESHOLD = 0.40   # Minimum confidence to report failure per step
    MAX_CONFIDENCE = 0.92             # Cap for step-level confidence score

    def __init__(self, llm_judge: LLMJudge | None = None):
        """Initialize detector with LLM judge.
        
        Args:
            llm_judge: Optional LLM judge implementation. Defaults to MockLLMJudge.
        """
        self.llm_judge = llm_judge or MockLLMJudge()
        # Instantiate EmbeddingMatcher once for all similarity checks
        self.embeddings = EmbeddingMatcher()

    def detect(self, trace: AgentTrace) -> DetectionResult:
        """Run context loss detection pipeline.
        
        Args:
            trace: AgentTrace to analyze
            
        Returns:
            DetectionResult with context loss classification
        """
        # Setup — Skip early steps (insufficient context before step 3)
        if len(trace.steps) < 4:
            return self.build_result(
                failure_type=FailureType.CONTEXT_LOSS,
                subtype="no_context_loss",
                confidence_score=1.0,
                evidence=[],
                reason="Trace too short for context loss analysis (need >= 4 steps)",
                detection_stage="context_loss_pipeline",
                fix_direction=None,
            )

        # For each step with step_index >= 3
        for step in trace.steps:
            if step.step_index < 3:
                continue  # Skip first 3 steps (insufficient context)

            evidence = []
            
            # --- Check 1: Tool Input vs Running Context ---
            context_loss_score = self._check_tool_input_grounding(
                step, trace, evidence
            )
            
            # --- Check 2: Thought Contradiction ---
            thought_score = self._check_thought_contradiction(
                step, trace, evidence
            )
            
            # --- Check 3: LLM Fallback (only if Checks 1 & 2 both produced no signal) ---
            llm_score = 0.0
            if context_loss_score == 0.0 and thought_score == 0.0:
                llm_score = self._check_llm_fallback(step, trace, evidence)
            
            # Step-level decision
            step_confidence = context_loss_score + thought_score + llm_score
            step_confidence = min(step_confidence, self.MAX_CONFIDENCE)
            
            if step_confidence >= self.MIN_CONFIDENCE_THRESHOLD:
                return self.build_result(
                    failure_type=FailureType.CONTEXT_LOSS,
                    subtype="context_loss_detected",
                    confidence_score=step_confidence,
                    evidence=evidence,
                    reason=f"Context loss detected at step {step.step_index} — agent dropped or contradicted information from prior steps",
                    detection_stage="context_loss_pipeline",
                    fix_direction="Ensure agent carries forward key values from prior tool outputs — consider summarizing context between steps or using explicit memory mechanisms",
                )

        # No step across the whole trace fired the threshold
        return self.build_result(
            failure_type=FailureType.CONTEXT_LOSS,
            subtype="no_context_loss",
            confidence_score=1.0,
            evidence=[],
            reason="No context loss detected across all steps",
            detection_stage="context_loss_pipeline",
            fix_direction=None,
        )

    def _check_tool_input_grounding(
        self, step: Step, trace: AgentTrace, evidence: list[Evidence]
    ) -> float:
        """Check if tool input contains ungrounded values when prior context exists.
        
        Returns context_loss_score (0.0 or calculated score).
        """
        # Handle None tool_input gracefully
        if not step.tool_input:
            return 0.0
        
        # Build running context from all prior tool_outputs
        prior_outputs = [
            s.tool_output for s in trace.steps
            if s.step_index < step.step_index and s.tool_output is not None
        ]
        
        # Check if prior successful calls exist for same tool_name
        prior_same_tool = [
            s for s in trace.steps
            if s.tool_name == step.tool_name
            and s.step_index < step.step_index
        ]
        
        # Only proceed if this tool was called successfully before
        if not prior_same_tool:
            return 0.0
        
        # Call grounding analyzer
        grounding_results = GroundingAnalyzer.analyze(step.tool_input, trace.task, prior_outputs)
        summary = GroundingAnalyzer.summarize(grounding_results)
        
        # Context loss signal fires when fields are ungrounded AND prior calls exist
        if summary.get('ungrounded', 0) > 0:
            ungrounded_ratio = summary['ungrounded'] / max(1, summary['total_fields'])
            context_loss_score = min(0.50, ungrounded_ratio * 0.60)
            
            ungrounded_fields = summary.get('ungrounded_fields', [])
            evidence.append(
                Evidence(
                    detection_stage="1 - Tool Input vs Running Context",
                    signal="context_dropped_in_tool_input",
                    confidence_contribution=context_loss_score,
                    explanation=f"Step {step.step_index}: {summary['ungrounded']} field(s) ({ungrounded_fields}) cannot be traced to task or prior outputs, but this tool was called successfully before — values may have been dropped from context",
                )
            )
            return context_loss_score
        
        return 0.0

    def _check_thought_contradiction(
        self, step: Step, trace: AgentTrace, evidence: list[Evidence]
    ) -> float:
        """Check if thought contradicts prior context.
        
        Returns thought_score (0.0 or 0.55).
        """
        # Only run if thought is present
        if not step.thought:
            return 0.0
        
        # Build running context string from all prior outputs
        prior_outputs = [
            s.tool_output for s in trace.steps
            if s.step_index < step.step_index and s.tool_output is not None
        ]
        prior_context_str = " ".join(
            GroundingAnalyzer._flatten_to_str(o) for o in prior_outputs
        )
        
        # Compare thought against prior context
        thought_context_sim = self.embeddings.similarity(step.thought, prior_context_str)
        
        if thought_context_sim < self.LOW_SIMILARITY_THRESHOLD:
            thought_score = 0.55
            evidence.append(
                Evidence(
                    detection_stage="2 - Thought vs Prior Context",
                    signal="thought_contradicts_prior_context",
                    confidence_contribution=0.55,
                    explanation=f"Step {step.step_index}: agent's thought shows low alignment with established prior context (similarity={thought_context_sim:.2f}) — agent may have lost track of earlier information",
                )
            )
            return thought_score
        
        return 0.0

    def _check_llm_fallback(
        self, step: Step, trace: AgentTrace, evidence: list[Evidence]
    ) -> float:
        """LLM fallback only runs when Checks 1 & 2 both produced no signal.
        
        Returns llm_score (0.0, 0.10, or calculated misinterpreted score).
        """
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
            if s.thought is not None:
                last_thought = s.thought
                break
        
        # Compute embedding score for LLM context
        final_output_str = str(trace.final_output) if trace.final_output else ""
        task_output_sim = self.embeddings.similarity(trace.task, final_output_str)
        
        # Run LLM judge
        llm_result = self.llm_judge.evaluate_goal_alignment(
            task=trace.task,
            final_output=trace.final_output,
            steps=steps_list,
            thought=last_thought,
            embedding_score=task_output_sim,
        )
        
        if llm_result.get("verdict") == "misinterpreted":
            llm_score = 0.45 + llm_result.get("confidence", 0) * 0.15
            evidence.append(
                Evidence(
                    detection_stage="3 - LLM Fallback",
                    signal="llm_context_loss_verdict",
                    confidence_contribution=llm_score,
                    explanation=f"LLM judge detected context inconsistency: {llm_result.get('reason', 'no reason')}",
                )
            )
            return llm_score
        elif llm_result.get("verdict") == "uncertain":
            llm_score = 0.10
            evidence.append(
                Evidence(
                    detection_stage="3 - LLM Fallback",
                    signal="llm_uncertain_verdict",
                    confidence_contribution=0.10,
                    explanation="LLM judge was uncertain about context consistency",
                )
            )
            return llm_score
        
        return 0.0
