# Tool use failure detector
# detectors/planning/tool_use.py
# Checks every step in a trace for tool use failures.
# Returns one of five outcomes (ToolUseSubtype):
#   WRONG_TOOL_SELECTED, INVALID_PARAMETERS, INCORRECT_PARAMETER_VALUES,
#   NO_FAILURE, INSUFFICIENT_EVIDENCE
#
# Pipeline is stop-on-first-hit: if Stage 1 fires, never run Stage 2.
# If Stage 2 fires, never run Stage 3.
#
# Only ONE subtype is returned per step. The detector checks every step
# in trace.steps and returns the first failure (stop-on-first-hit across
# steps too).

from typing import Any

from agent_diagnostician.detectors.base import BaseDetector
from agent_diagnostician.models.trace import AgentTrace, Step
from agent_diagnostician.models.result import DetectionResult, Evidence
from agent_diagnostician.models.enums import FailureType, ToolUseSubtype, ConfidenceBand

from agent_diagnostician.analysis.embeddings import EmbeddingMatcher
from agent_diagnostician.analysis.schema import SchemaValidator
from agent_diagnostician.analysis.grounding import GroundingAnalyzer
from agent_diagnostician.analysis.llm_judge import LLMJudge, MockLLMJudge


class ToolUseDetector(BaseDetector):
    """Detects tool use failures in agent execution traces.
    
    Uses a fallback pipeline:
      Rules > Runtime Inference > Embeddings > LLM
    
    Instantiate with an optional LLM judge. If none provided, uses
    MockLLMJudge for development/testing.
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
        """Run full tool use failure detection pipeline on a trace.
        
        Checks every step. Returns the first failure found (stop-on-first-hit
        across steps too). If no step fails, returns NO_FAILURE.
        
        Args:
            trace: AgentTrace to analyze
            
        Returns:
            DetectionResult with failure classification, confidence,
            evidence, and fix direction
        """
        # If trace has no steps, return insufficient evidence
        if not trace.steps:
            return self.build_result(
                failure_type=FailureType.TOOL_USE_FAILURE,
                subtype=ToolUseSubtype.INSUFFICIENT_EVIDENCE.value,
                confidence_score=0.0,
                evidence=[],
                reason="No steps found in trace to analyze",
                detection_stage="none",
                fix_direction="Provide a trace with at least one tool invocation step",
            )

        # Check each step in order
        for step in trace.steps:
            result = self._detect_step(trace, step)
            # If this step produced a failure (not NO_FAILURE or INSUFFICIENT_EVIDENCE), return it
            if result.subtype not in (
                ToolUseSubtype.NO_FAILURE.value,
                ToolUseSubtype.INSUFFICIENT_EVIDENCE.value,
            ):
                return result

        # No step failed
        return self.build_result(
            failure_type=FailureType.TOOL_USE_FAILURE,
            subtype=ToolUseSubtype.NO_FAILURE.value,
            confidence_score=1.0,
            evidence=[],
            reason="All steps passed tool use validation checks",
            detection_stage="none",
            fix_direction="No fix required — agent used tools correctly",
        )

    def _detect_step(self, trace: AgentTrace, step: Step) -> DetectionResult:
        """Run full pipeline on a single step.
        
        Args:
            trace: Full trace (for context like available_tools)
            step: Step to analyze
            
        Returns:
            DetectionResult for this step
        """
        # Stage 1 — Wrong Tool Selected
        result = self._stage_wrong_tool(trace, step)
        if result is not None:
            return result

        # Stage 2 — Invalid Parameters
        result = self._stage_invalid_parameters(trace, step)
        if result is not None:
            return result

        # Stage 3 — Incorrect Parameter Values
        result = self._stage_incorrect_values(trace, step)
        if result is not None:
            return result

        # All stages passed
        return self.build_result(
            failure_type=FailureType.TOOL_USE_FAILURE,
            subtype=ToolUseSubtype.NO_FAILURE.value,
            confidence_score=1.0,
            evidence=[],
            reason="Step passed all tool use validation checks",
            detection_stage="none",
            fix_direction="No fix required — tool use was correct",
        )

    # ───────────────────────────────────────────────────────────────────────
    # Stage 1 — Wrong Tool Selected
    # ───────────────────────────────────────────────────────────────────────

    def _stage_wrong_tool(
        self, trace: AgentTrace, step: Step
    ) -> DetectionResult | None:
        """Stage 1 — Wrong Tool Selected?
        
        Pipeline (stop-on-first-hit):
          1A. If thought present: task vs thought (comprehension check)
          1B. If 1A passes: thought vs tool_name (selection check)
          2.  If available_tools with descriptions present: rank tools by
              similarity to task, check if called tool is rank-1 or close.
          3.  LLM fallback if 1/2 unavailable or inconclusive.
        
        Returns:
            DetectionResult if WRONG_TOOL_SELECTED, None if pass/must fall through
        """
        # Step 1A — Task vs Thought comparison (if thought present)
        if step.thought is not None:
            task_thought_sim = self.embeddings.similarity(trace.task, step.thought)
            if task_thought_sim < 0.5:
                # Thought misunderstands the task → wrong tool will be selected
                return self.build_result(
                    failure_type=FailureType.TOOL_USE_FAILURE,
                    subtype=ToolUseSubtype.WRONG_TOOL_SELECTED.value,
                    confidence_score=0.75,
                    evidence=[
                        Evidence(
                            detection_stage="1A - Task vs Thought",
                            signal="thought_task_mismatch",
                            confidence_contribution=0.75,
                            explanation=f"Agent's thought demonstrates misunderstanding of the task "
                                        f"(similarity={task_thought_sim:.2f}). Task: '{trace.task[:100]}...'. "
                                        f"Thought: '{step.thought[:100]}...'",
                        )
                    ],
                    reason="Agent's thought shows it misunderstood the task, leading to wrong tool selection",
                    detection_stage="1A",
                    fix_direction="Revise the agent's instructions or prompt to clarify the task intent",
                )

        # Step 1B — Thought vs Tool Name (if thought present and 1A passed)
        if step.thought is not None:
            thought_tool_match = self._match_thought_to_tool(
                thought=step.thought,
                tool_name=step.tool_name,
                available_tools=trace.available_tools,
            )
            if not thought_tool_match:
                return self.build_result(
                    failure_type=FailureType.TOOL_USE_FAILURE,
                    subtype=ToolUseSubtype.WRONG_TOOL_SELECTED.value,
                    confidence_score=0.70,
                    evidence=[
                        Evidence(
                            detection_stage="1B - Thought vs Tool",
                            signal="thought_tool_mismatch",
                            confidence_contribution=0.70,
                            explanation=f"Agent's thought describes actions inconsistent with the selected tool "
                                        f"'{step.tool_name}'. Thought: '{step.thought[:100]}...'",
                        )
                    ],
                    reason="Agent's stated intent (in thought) does not match the selected tool",
                    detection_stage="1B",
                    fix_direction="Align the agent's reasoning with the tool it selects, or select a tool that matches its reasoning",
                )

        # Step 2 — Tool ranking via embedding (if available_tools present)
        if trace.available_tools is not None:
            # Build tool descriptions list
            tool_descriptions = []
            for tool in trace.available_tools:
                desc = tool.description if tool.description else ""
                tool_descriptions.append(f"{tool.name}: {desc}".strip())

            if tool_descriptions:
                # Rank tools by similarity to task
                ranked = self.embeddings.rank_by_similarity(trace.task, tool_descriptions)

                # Find rank of the actually called tool
                called_rank = None
                called_rank_index = None
                for idx, item in enumerate(ranked):
                    # Extract tool name from "name: description" format
                    rank_name = item["text"].split(":")[0].strip()
                    if rank_name == step.tool_name:
                        called_rank = idx + 1  # 1-indexed rank
                        called_rank_index = idx
                        break

                if called_rank is not None and called_rank_index is not None:
                    # Compute similarity gap between rank-1 and rank-2
                    gap = self.embeddings.similarity_gap(ranked)

                    # If called tool is NOT rank-1 AND gap > 0.15: wrong tool
                    if called_rank > 1 and gap > 0.15:
                        top_tool_name = ranked[0]["text"].split(":")[0].strip()
                        return self.build_result(
                            failure_type=FailureType.TOOL_USE_FAILURE,
                            subtype=ToolUseSubtype.WRONG_TOOL_SELECTED.value,
                            confidence_score=0.60,
                            evidence=[
                                Evidence(
                                    detection_stage="2 - Tool Ranking",
                                    signal="tool_ranking_mismatch",
                                    confidence_contribution=0.60,
                                    explanation=f"Called tool '{step.tool_name}' ranked {called_rank} (out of {len(ranked)}), "
                                                f"but rank-1 tool '{top_tool_name}' is clearly better (gap={gap:.2f}).",
                                )
                            ],
                            reason=f"Agent selected tool '{step.tool_name}' when a clearly better tool '{top_tool_name}' was available",
                            detection_stage="2",
                            fix_direction="Select the top-ranked tool based on semantic similarity to the task",
                        )

        # Step 3 — LLM fallback (if steps 1/2 were unavailable or inconclusive)
        available_tools_list = []
        if trace.available_tools is not None:
            for tool in trace.available_tools:
                available_tools_list.append({
                    "name": tool.name,
                    "description": tool.description or "",
                })

        llm_result = self.llm_judge.evaluate_wrong_tool(
            task=trace.task,
            selected_tool=step.tool_name,
            available_tools=available_tools_list,
            thought=step.thought,
        )

        if llm_result.get("verdict") == "incorrect":
            return self.build_result(
                failure_type=FailureType.TOOL_USE_FAILURE,
                subtype=ToolUseSubtype.WRONG_TOOL_SELECTED.value,
                confidence_score=0.55,
                evidence=[
                    Evidence(
                        detection_stage="3 - LLM Fallback",
                        signal="llm_wrong_tool_verdict",
                        confidence_contribution=0.55,
                        explanation=f"LLM judge judged tool selection incorrect: {llm_result.get('reason', 'no reason provided')}",
                    )
                ],
                reason=f"LLM judge determined the selected tool was incorrect: {llm_result.get('reason', 'uncertain')}",
                detection_stage="3",
                fix_direction="Choose a different tool that better matches the task requirements",
            )

        # LLM said "uncertain" — fall through to Stage 2
        return None

    def _match_thought_to_tool(
        self, thought: str, tool_name: str, available_tools: list | None
    ) -> bool:
        """Check if thought describes actions consistent with the selected tool.
        
        Returns True if thought appears consistent, False if mismatch detected.
        """
        # If available_tools present, embed thought against tool description
        if available_tools is not None:
            for tool in available_tools:
                if tool.name == tool_name:
                    if tool.description:
                        sim = self.embeddings.similarity(thought, tool.description)
                        # High similarity suggests thought matches tool purpose
                        return sim >= 0.4

        # Fallback: simple keyword heuristic
        # Look for tool-related keywords in thought
        tool_lower = tool_name.lower()
        thought_lower = thought.lower()

        # If thought contains the tool name or strong related terms, assume match
        if tool_lower in thought_lower:
            return True

        # Simple heuristic: if thought mentions relevant actions, assume OK
        common_verbs = ["search", "calculate", "write", "read", "analyze", "compute"]
        for verb in common_verbs:
            if verb in thought_lower and verb in tool_lower:
                return True

        # If we can't find evidence of mismatch, assume OK (conservative)
        return True

    # ───────────────────────────────────────────────────────────────────────
    # Stage 2 — Invalid Parameters
    # ───────────────────────────────────────────────────────────────────────

    def _stage_invalid_parameters(
        self, trace: AgentTrace, step: Step
    ) -> DetectionResult | None:
        """Stage 2 — Invalid Parameters? (assumes tool selection is correct)
        
        Pipeline (stop-on-first-hit):
          1. If ToolSpec with matching name exists AND schema_ not None:
             validate against official schema.
          2. If schema unavailable: infer from ≥2 prior successful calls.
          3. LLM fallback — judges structure only, not values.
        
        Returns:
            DetectionResult if INVALID_PARAMETERS, None if pass/must fall through
        """
        # Step 1 — Validate against official schema (if available)
        if trace.available_tools is not None:
            toolspec = self._find_toolspec(trace.available_tools, step.tool_name)
            if toolspec is not None and toolspec.schema_ is not None:
                validation = SchemaValidator.validate_against_schema(
                    step.tool_input, toolspec.schema_
                )

                if not validation["valid"]:
                    errors = validation.get("errors", [])
                    return self.build_result(
                        failure_type=FailureType.TOOL_USE_FAILURE,
                        subtype=ToolUseSubtype.INVALID_PARAMETERS.value,
                        confidence_score=0.90,
                        evidence=[
                            Evidence(
                                detection_stage="1 - Official Schema Validation",
                                signal="schema_validation_failed",
                                confidence_contribution=0.90,
                                explanation=f"Tool input failed schema validation: {'; '.join(errors)}",
                            )
                        ],
                        reason=f"Tool input does not match the official schema for '{step.tool_name}': {'; '.join(errors)}",
                        detection_stage="1",
                        fix_direction=f"Fix parameter structure to match the official schema for '{step.tool_name}'",
                    )

        # Step 2 — Infer schema from prior successful calls (if official schema unavailable)
        prior_successful_calls = self._get_prior_successful_calls(trace, step)
        if len(prior_successful_calls) >= 2:
            inferred_schema = SchemaValidator.infer_schema_from_calls(
                prior_successful_calls
            )

            if inferred_schema is not None:
                validation = SchemaValidator.validate_against_schema(
                    step.tool_input, inferred_schema
                )

                if not validation["valid"]:
                    errors = validation.get("errors", [])
                    return self.build_result(
                        failure_type=FailureType.TOOL_USE_FAILURE,
                        subtype=ToolUseSubtype.INVALID_PARAMETERS.value,
                        confidence_score=0.65,
                        evidence=[
                            Evidence(
                                detection_stage="2 - Inferred Schema Validation",
                                signal="inferred_schema_validation_failed",
                                confidence_contribution=0.65,
                                explanation=f"Tool input failed validation against inferred schema: {'; '.join(errors)}",
                            )
                        ],
                        reason=f"Tool input does not match the inferred schema (based on {len(prior_successful_calls)} prior successful calls): {'; '.join(errors)}",
                        detection_stage="2",
                        fix_direction=f"Match the parameter structure used in prior successful calls to '{step.tool_name}'",
                    )

        # Step 3 — LLM fallback
        llm_result = self.llm_judge.evaluate_parameter_structure(
            tool_name=step.tool_name,
            tool_input=step.tool_input,
            task=trace.task,
        )

        if llm_result.get("verdict") == "invalid":
            return self.build_result(
                failure_type=FailureType.TOOL_USE_FAILURE,
                subtype=ToolUseSubtype.INVALID_PARAMETERS.value,
                confidence_score=0.55,
                evidence=[
                    Evidence(
                        detection_stage="3 - LLM Fallback",
                        signal="llm_invalid_structure_verdict",
                        confidence_contribution=0.55,
                        explanation=f"LLM judge judged parameter structure invalid: {llm_result.get('reason', 'no reason provided')}",
                    )
                ],
                reason=f"LLM judge determined the parameter structure is invalid: {llm_result.get('reason', 'uncertain')}",
                detection_stage="3",
                fix_direction="Fix the parameter structure to match what's expected for this tool",
            )

        # LLM said "uncertain" — fall through to Stage 3
        return None

    def _find_toolspec(
        self, available_tools: list, tool_name: str
    ) -> object | None:
        """Find ToolSpec with matching name from available_tools list."""
        for tool in available_tools:
            if hasattr(tool, "name") and tool.name == tool_name:
                return tool
        return None

    def _get_prior_successful_calls(
        self, trace: AgentTrace, current_step: Step
    ) -> list[dict]:
        """Get prior steps with same tool_name that succeeded.
        
        Prior means step_index < current step_index.
        Success means step_status != "error" and step_status != "failed".
        """
        prior_calls = []
        for step in trace.steps:
            if step.step_index >= current_step.step_index:
                break
            if step.tool_name == current_step.tool_name:
                status = step.step_status
                if status is None or status not in ("error", "failed"):
                    prior_calls.append(step.tool_input)
        return prior_calls

    # ───────────────────────────────────────────────────────────────────────
    # Stage 3 — Incorrect Parameter Values
    # ───────────────────────────────────────────────────────────────────────

    def _stage_incorrect_values(
        self, trace: AgentTrace, step: Step
    ) -> DetectionResult | None:
        """Stage 3 — Incorrect Parameter Values? (assumes tool and schema correct)
        
        Pipeline:
          1. Grounding check (default, zero-cost) — trace values to task/prior outputs.
          2. If thought present: compare thought vs actual tool_input values.
          3. LLM fallback — judge whether values are justified.
        
        Returns:
            DetectionResult if INCORRECT_PARAMETER_VALUES, None if pass
        """
        # Collect prior outputs (all steps before current)
        prior_outputs = []
        for s in trace.steps:
            if s.step_index < step.step_index:
                prior_outputs.append(s.tool_output)

        # Step 1 — Grounding analysis (always runs, zero-cost default)
        grounding_results = GroundingAnalyzer.analyze(
            step.tool_input, trace.task, prior_outputs
        )
        summary = GroundingAnalyzer.summarize(grounding_results)

        ungrounded_count = summary["ungrounded"]
        ungrounded_fields = summary["ungrounded_fields"]

        # If there are ungrounded fields, we'll carry this forward as evidence
        # but don't return yet — this alone isn't enough to confirm failure
        ungrounded_evidence = []
        if ungrounded_count > 0:
            # Weight per ungrounded field, cap at 0.80 total
            per_field_weight = min(0.50, 0.80 / max(1, ungrounded_count))
            total_weight = min(0.80, per_field_weight * ungrounded_count)

            for field in ungrounded_fields:
                ungrounded_evidence.append(
                    Evidence(
                        detection_stage="3.1 - Grounding Analysis",
                        signal="grounding_check_failed",
                        confidence_contribution=per_field_weight,
                        explanation=f"Value for field '{field}' cannot be traced back to the task or prior tool outputs",
                    )
                )

        # Step 2 — Thought consistency check (if thought present)
        if step.thought is not None:
            thought_str = str(step.thought)
            tool_input_str = str(step.tool_input)
            thought_input_sim = self.embeddings.similarity(thought_str, tool_input_str)

            if thought_input_sim < 0.5 and ungrounded_count > 0:
                # Both conditions met: thought doesn't match input, AND values are ungrounded
                combined_confidence = min(
                    0.80,
                    0.70 + (0.50 * min(2, ungrounded_count))  # thought mismatch + grounding failures
                )

                return self.build_result(
                    failure_type=FailureType.TOOL_USE_FAILURE,
                    subtype=ToolUseSubtype.INCORRECT_PARAMETER_VALUES.value,
                    confidence_score=combined_confidence,
                    evidence=ungrounded_evidence + [
                        Evidence(
                            detection_stage="3.2 - Thought vs Input",
                            signal="thought_input_mismatch",
                            confidence_contribution=0.70,
                            explanation=f"Agent's thought ({thought_str[:100]}...) does not match the values it actually passed ({tool_input_str[:100]}...)",
                        )
                    ],
                    reason=f"Agent's stated intent (in thought) doesn't match the values it passed, and values cannot be traced to task/prior outputs",
                    detection_stage="3.2",
                    fix_direction="Ensure the tool input values match the agent's stated intent and can be traced to the task or prior outputs",
                )

        # Step 3 — LLM fallback
        llm_result = self.llm_judge.evaluate_parameter_values(
            task=trace.task,
            tool_input=step.tool_input,
            prior_outputs=prior_outputs,
            thought=step.thought,
        )

        if llm_result.get("verdict") == "unjustified":
            return self.build_result(
                failure_type=FailureType.TOOL_USE_FAILURE,
                subtype=ToolUseSubtype.INCORRECT_PARAMETER_VALUES.value,
                confidence_score=0.55,
                evidence=ungrounded_evidence + [
                    Evidence(
                        detection_stage="3.3 - LLM Fallback",
                        signal="llm_unjustified_values_verdict",
                        confidence_contribution=0.55,
                        explanation=f"LLM judge judged parameter values unjustified: {llm_result.get('reason', 'no reason provided')}",
                    )
                ],
                reason=f"LLM judge determined the parameter values are not logically justified: {llm_result.get('reason', 'uncertain')}",
                detection_stage="3.3",
                fix_direction="Ensure parameter values can be traced back to the task or prior tool outputs",
            )

        # Verdict == "uncertain"
        if ungrounded_count > 0:
            # Some ungrounded fields exist, LLM is uncertain
            return self.build_result(
                failure_type=FailureType.TOOL_USE_FAILURE,
                subtype=ToolUseSubtype.INCORRECT_PARAMETER_VALUES.value,
                confidence_score=0.35,  # MAYBE band
                evidence=ungrounded_evidence + [
                    Evidence(
                        detection_stage="3.3 - LLM Fallback",
                        signal="llm_uncertain_with_ungrounded",
                        confidence_contribution=0.10,
                        explanation=f"LLM judge was uncertain, but {ungrounded_count} field(s) could not be grounded",
                    )
                ],
                reason=f"LLM judge was uncertain, but {ungrounded_count} parameter(s) could not be traced to the task or prior outputs",
                detection_stage="3.3",
                fix_direction="Review parameter values and ensure they can be traced to the task or prior tool outputs",
            )

        # LLM uncertain and no ungrounded fields → Insufficient Evidence
        return self.build_result(
            failure_type=FailureType.TOOL_USE_FAILURE,
            subtype=ToolUseSubtype.INSUFFICIENT_EVIDENCE.value,
            confidence_score=0.10,
            evidence=[
                Evidence(
                    detection_stage="3.3 - LLM Fallback",
                    signal="llm_uncertain_no_evidence",
                    confidence_contribution=0.10,
                    explanation="LLM judge could not determine if values were correct",
                )
            ],
            reason="LLM judge was uncertain and no ungrounded fields were detected",
            detection_stage="3.3",
            fix_direction="Gather more information or context to evaluate parameter value correctness",
        )
