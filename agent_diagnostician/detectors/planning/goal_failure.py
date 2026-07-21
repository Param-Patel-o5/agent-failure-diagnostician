# Goal failure detector
# detectors/planning/goal_failure.py
# Checks whether the agent satisfied the user's actual goal.
# Returns one of four outcomes (GoalFailureSubtype):
#   CONSTRAINT_VIOLATION, TASK_MISINTERPRETATION, NO_FAILURE, INSUFFICIENT_EVIDENCE
#
# CRITICAL: Both Stage 1 (Constraint Validation) and Stage 2 (Task Misinterpretation)
# run independently and produce their own confidence scores. The Aggregator picks
# the higher one as primary. The lower one becomes secondary_evidence if >= 0.30.

from typing import Any

from agent_diagnostician.detectors.base import BaseDetector
from agent_diagnostician.models.trace import AgentTrace, Step
from agent_diagnostician.models.result import DetectionResult, Evidence
from agent_diagnostician.models.enums import FailureType, GoalFailureSubtype, ConfidenceBand

from agent_diagnostician.analysis.embeddings import EmbeddingMatcher
from agent_diagnostician.analysis.llm_judge import LLMJudge, MockLLMJudge
from agent_diagnostician.analysis.constraint_extractor import ConstraintExtractor


class GoalFailureDetector(BaseDetector):
    """Detects goal satisfaction failures in agent execution traces.
    
    Two independent analysis branches:
      - Stage 1: Constraint Validation (did agent respect explicit constraints?)
      - Stage 2: Task Misinterpretation (did agent solve the correct task?)
    
    Both always run. Aggregator picks the higher confidence score as primary,
    with the other as secondary_evidence if >= 0.30.
    """

    def __init__(self, llm_judge: LLMJudge | None = None):
        """Initialize detector with LLM judge and embedding matcher.
        
        EmbeddingMatcher is expensive to initialize — instantiate ONCE
        here, not inside detect() or any stage method.
        
        Args:
            llm_judge: LLM judge implementation. If None, uses MockLLMJudge.
        """
        self.llm_judge = llm_judge or MockLLMJudge()
        self.embeddings = EmbeddingMatcher()

    def detect(self, trace: AgentTrace) -> DetectionResult:
        """Run full goal failure detection pipeline on a trace.
        
        Both Stage 1 and Stage 2 ALWAYS run independently. The Aggregator
        compares their confidence scores and picks the higher one as primary.
        
        Args:
            trace: AgentTrace to analyze
            
        Returns:
            DetectionResult with failure classification, confidence,
            evidence, and fix direction
        """
        # Stage 0 — Constraint Extraction
        # Use pre-computed constraint_list if available (Tier 4), otherwise extract
        if trace.constraint_list is not None and len(trace.constraint_list) > 0:
            extracted_constraints = trace.constraint_list
        else:
            extracted_constraints = self._extract_constraints(trace.task)

        # Stage 1 — Constraint Validation (always runs)
        constraint_result = self._stage_constraint_validation(trace, extracted_constraints)

        # Stage 2 — Task Misinterpretation Analysis (always runs)
        misinterpretation_result = self._stage_misinterpretation(trace)

        # Aggregator — Choose primary result
        return self._aggregate_results(trace, constraint_result, misinterpretation_result, extracted_constraints)

    def _extract_constraints(self, task: str) -> list[dict]:
        """Extract constraints from task text using ConstraintExtractor.
        
        Returns list of constraint dicts with type, subtype, value, unit, raw.
        If no constraints found, returns empty list.
        """
        return ConstraintExtractor.extract(task)

    # ───────────────────────────────────────────────────────────────────────
    # Stage 1 — Constraint Validation
    # ───────────────────────────────────────────────────────────────────────

    def _stage_constraint_validation(
        self, trace: AgentTrace, constraints: list[dict]
    ) -> tuple[float, list[Evidence]]:
        """Stage 1 — Constraint Validation.
        
        Three parallel checks (all three run regardless):
          Check A: Validate each constraint against final_output
          Check B: Thought consistency (supporting signal only)
          Check C: Semantic similarity (supporting signal only)
        
        Returns:
            Tuple of (constraint_confidence, list of constraint_evidence)
        """
        confidence = 0.0
        evidence = []

        # If no constraints extracted, skip to return (confidence = 0.0)
        if not constraints:
            return (0.0, [])

        total_constraints = len(constraints)
        violated_count = 0
        violation_details = []

        # Check A — Validate each constraint against final_output
        final_output_str = str(trace.final_output) if trace.final_output is not None else ""

        for constraint in constraints:
            validation = self._validate_single_constraint(constraint, trace.final_output)

            if not validation["satisfied"]:
                violated_count += 1
                violation_details.append({
                    "constraint": constraint,
                    "reason": validation["reason"],
                })
                evidence.append(
                    Evidence(
                        detection_stage="1A - Constraint Validation",
                        signal="constraint_violated",
                        confidence_contribution=0.80 / total_constraints,
                        explanation=f"Constraint violated: {constraint['raw']} — {validation['reason']}",
                    )
                )

        # Check B — Thought consistency (supporting signal only)
        thought_inconsistency_fired = False
        if trace.steps:
            for step in trace.steps:
                if step.thought is not None:
                    sim = self.embeddings.similarity(step.thought, trace.task)
                    if sim < 0.5:
                        # Thought shows agent may have misread constraints
                        thought_inconsistency_fired = True
                        evidence.append(
                            Evidence(
                                detection_stage="1B - Thought Consistency",
                                signal="thought_inconsistency",
                                confidence_contribution=0.10,
                                explanation=f"Step {step.step_index} thought shows potential constraint misunderstanding (task-thought similarity={sim:.2f})",
                            )
                        )
                        break  # Only add once

        # Check C — Semantic check on final_output vs task (supporting signal only)
        semantic_sim = self.embeddings.similarity(trace.task, final_output_str)
        if semantic_sim < 0.5:
            evidence.append(
                Evidence(
                    detection_stage="1C - Semantic Similarity",
                    signal="low_semantic_similarity",
                    confidence_contribution=0.0,  # Supporting only, no direct contribution
                    explanation=f"Low semantic similarity between task and final output ({semantic_sim:.2f}) — task may not have been understood correctly",
                )
            )

        # Calculate final constraint_confidence
        if violated_count > 0:
            # Base from violations
            confidence = (0.80 / total_constraints) * violated_count
            # Add thought inconsistency bonus if it fired
            if thought_inconsistency_fired:
                confidence += 0.10
        else:
            # No violations, but check for thought inconsistency
            if thought_inconsistency_fired:
                confidence = 0.10

        # Cap at 0.90
        confidence = min(0.90, confidence)

        return (confidence, evidence)

    def _validate_single_constraint(
        self, constraint: dict, actual_value: Any
    ) -> dict:
        """Validate a single constraint against the actual output."""
        return ConstraintExtractor.validate_constraint(constraint, actual_value)

    # ───────────────────────────────────────────────────────────────────────
    # Stage 2 — Task Misinterpretation Analysis
    # ───────────────────────────────────────────────────────────────────────

    def _stage_misinterpretation(
        self, trace: AgentTrace
    ) -> tuple[float, list[Evidence]]:
        """Stage 2 — Task Misinterpretation Analysis.
        
        Steps:
          Step A: Thought vs execution (supporting)
          Step B: Embedding similarity (weak supporting, passed to LLM)
          Step C: LLM judge (core engine)
        
        Returns:
            Tuple of (misinterpretation_confidence, list of misinterpretation_evidence)
        """
        evidence = []
        base_confidence = 0.0

        # Step A — Thought vs execution (supporting signal only)
        thought_execution_mismatch_fired = False
        for step in trace.steps:
            if step.thought is not None:
                thought_str = str(step.thought)
                output_str = str(step.tool_output) if step.tool_output is not None else ""
                sim = self.embeddings.similarity(thought_str, output_str)

                if sim < 0.45:
                    # Execution may not have followed stated intent
                    thought_execution_mismatch_fired = True
                    evidence.append(
                        Evidence(
                            detection_stage="2A - Thought vs Execution",
                            signal="thought_execution_mismatch",
                            confidence_contribution=0.25,
                            explanation=f"Step {step.step_index}: thought does not align with execution (similarity={sim:.2f})",
                        )
                    )
                    break  # Only add once

        # Step B — Embedding similarity (weak supporting, passed to LLM)
        final_output_str = str(trace.final_output) if trace.final_output is not None else ""
        task_output_sim = self.embeddings.similarity(trace.task, final_output_str)
        # Do NOT gate on this score — just record it as weak evidence
        if task_output_sim < 0.5:
            evidence.append(
                Evidence(
                    detection_stage="2B - Task vs Output Similarity",
                    signal="low_task_output_similarity",
                    confidence_contribution=0.0,  # Supporting only
                    explanation=f"Low semantic similarity between task and final output ({task_output_sim:.2f})",
                )
            )

        # Step C — LLM judge (core engine)
        # Prepare steps list for LLM
        steps_list = []
        for s in trace.steps:
            steps_list.append({
                "step_index": s.step_index,
                "tool_name": s.tool_name,
                "tool_input": s.tool_input,
                "tool_output": s.tool_output,
            })

        # Get last step's thought (or None)
        last_thought = next((s.thought for s in reversed(trace.steps) if s.thought), None)

        llm_result = self.llm_judge.evaluate_goal_alignment(
            task=trace.task,
            final_output=trace.final_output,
            steps=steps_list,
            thought=last_thought,
            embedding_score=task_output_sim,
        )

        # Build LLM evidence
        if llm_result.get("verdict") == "misinterpreted":
            base_confidence = 0.55 + llm_result.get("confidence", 0) * 0.20
            evidence.append(
                Evidence(
                    detection_stage="2C - LLM Judge",
                    signal="llm_misinterpreted_verdict",
                    confidence_contribution=0.55 + llm_result.get("confidence", 0) * 0.20,
                    explanation=f"LLM judge determined task was misinterpreted: {llm_result.get('reason', 'no reason provided')}",
                )
            )
        elif llm_result.get("verdict") == "uncertain":
            base_confidence = 0.10
            evidence.append(
                Evidence(
                    detection_stage="2C - LLM Judge",
                    signal="llm_uncertain_verdict",
                    confidence_contribution=0.10,
                    explanation=f"LLM judge was uncertain about task completion: {llm_result.get('reason', 'no reason provided')}",
                )
            )
        # elif llm_result.get("verdict") == "correct":
        #     base_confidence = 0.0 (already set)
        #     No evidence added — no signal fired

        # Add thought-execution mismatch bonus if it fired
        if thought_execution_mismatch_fired:
            base_confidence += 0.25

        # Cap at 0.90
        base_confidence = min(0.90, base_confidence)

        return (base_confidence, evidence)

    # ───────────────────────────────────────────────────────────────────────
    # Aggregator — Choose Primary Result
    # ───────────────────────────────────────────────────────────────────────

    def _aggregate_results(
        self,
        trace: AgentTrace,
        constraint_result: tuple[float, list[Evidence]],
        misinterpretation_result: tuple[float, list[Evidence]],
        extracted_constraints: list[dict],
    ) -> DetectionResult:
        """Aggregator — compare confidence scores and choose primary result.
        
        CASE 1 (check first): if both scores < 0.20 → NO_FAILURE
        CASE 4 (check second): if constraints extracted but both scores very low → INSUFFICIENT_EVIDENCE
        CASE 2: if constraint_confidence >= misinterpretation_confidence → Primary = CONSTRAINT_VIOLATION
        CASE 3: if misinterpretation_confidence > constraint_confidence → Primary = TASK_MISINTERPRETATION
        """
        constraint_confidence, constraint_evidence = constraint_result
        misinterpretation_confidence, misinterpretation_evidence = misinterpretation_result

        # CASE 1: Both are 0.0 or below threshold (< 0.20) → NO_FAILURE
        if constraint_confidence < 0.20 and misinterpretation_confidence < 0.20:
            return self.build_result(
                failure_type=FailureType.GOAL_SATISFACTION_FAILURE,
                subtype=GoalFailureSubtype.NO_FAILURE.value,
                confidence_score=1.0,
                evidence=[],
                reason="No goal satisfaction issues detected — agent respected constraints and solved the correct task",
                detection_stage="none",
                fix_direction="No fix required — goal was satisfied",
            )

        # CASE 4: Constraints were extracted but neither analysis produced a clear verdict
        # Only fires when we had constraints to validate but couldn't reach a verdict
        if len(extracted_constraints) > 0 and constraint_confidence == 0.0 and misinterpretation_confidence <= 0.10:
            llm_uncertain = any(
                e.signal in ("llm_uncertain_verdict", "llm_uncertain_with_ungrounded")
                for e in misinterpretation_evidence
            )
            if llm_uncertain and not constraint_evidence:
                return self.build_result(
                    failure_type=FailureType.GOAL_SATISFACTION_FAILURE,
                    subtype=GoalFailureSubtype.INSUFFICIENT_EVIDENCE.value,
                    confidence_score=0.10,
                    evidence=misinterpretation_evidence,
                    reason="Constraints were present but neither analysis produced a clear verdict",
                    detection_stage="aggregator",
                    fix_direction="Provide clearer task instructions with explicit constraints, or gather more context to evaluate task completion",
                )

        # CASE 2: constraint_confidence >= misinterpretation_confidence
        if constraint_confidence >= misinterpretation_confidence:
            primary_confidence = constraint_confidence
            primary_evidence = constraint_evidence

            # Build secondary_evidence if misinterpretation_confidence >= 0.30
            secondary_evidence = None
            if misinterpretation_confidence >= 0.30:
                secondary_evidence = self.build_result(
                    failure_type=FailureType.GOAL_SATISFACTION_FAILURE,
                    subtype=GoalFailureSubtype.TASK_MISINTERPRETATION.value,
                    confidence_score=misinterpretation_confidence,
                    evidence=misinterpretation_evidence,
                    reason="Task misinterpretation signal detected as secondary support",
                    detection_stage="2",
                    fix_direction="Review agent reasoning to ensure correct task interpretation",
                )

            return self.build_result(
                failure_type=FailureType.GOAL_SATISFACTION_FAILURE,
                subtype=GoalFailureSubtype.CONSTRAINT_VIOLATION.value,
                confidence_score=primary_confidence,
                evidence=primary_evidence,
                reason=f"Agent violated {len([e for e in constraint_evidence if e.signal == 'constraint_violated'])} explicit constraint(s) from the task",
                detection_stage="1",
                fix_direction="Ensure agent respects all explicit constraints (numeric, categorical, structural) specified in the task",
                secondary_evidence=secondary_evidence,
            )

        # CASE 3: misinterpretation_confidence > constraint_confidence
        primary_confidence = misinterpretation_confidence
        primary_evidence = misinterpretation_evidence

        # Build secondary_evidence if constraint_confidence >= 0.30
        secondary_evidence = None
        if constraint_confidence >= 0.30:
            secondary_evidence = self.build_result(
                failure_type=FailureType.GOAL_SATISFACTION_FAILURE,
                subtype=GoalFailureSubtype.CONSTRAINT_VIOLATION.value,
                confidence_score=constraint_confidence,
                evidence=constraint_evidence,
                reason="Constraint violation signal detected as secondary support",
                detection_stage="1",
                fix_direction="Review explicit constraints from task and ensure they are respected",
            )

        return self.build_result(
            failure_type=FailureType.GOAL_SATISFACTION_FAILURE,
            subtype=GoalFailureSubtype.TASK_MISINTERPRETATION.value,
            confidence_score=primary_confidence,
            evidence=primary_evidence,
            reason="Agent misinterpreted the task — execution did not align with the stated goal",
            detection_stage="2",
            fix_direction="Clarify task instructions or improve agent's task comprehension",
            secondary_evidence=secondary_evidence,
        )
        