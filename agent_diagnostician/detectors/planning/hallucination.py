# Hallucination failure detector
# detectors/planning/hallucination.py
# Detects whether the agent fabricated values in tool_input or thought
# that cannot be traced to the task or any prior tool output.

import re
from typing import Any

from agent_diagnostician.detectors.base import BaseDetector
from agent_diagnostician.models.trace import AgentTrace, Step
from agent_diagnostician.models.result import DetectionResult, Evidence
from agent_diagnostician.models.enums import FailureType, HallucinationSubtype, STEP_FAILURE_STATUSES

from agent_diagnostician.config import (
    HALLUCINATION_CLEAR_PASS_THRESHOLD,
    HALLUCINATION_DETECT_THRESHOLD,
    HALLUCINATION_GROUNDING_CAP,
    HALLUCINATION_GROUNDING_WEIGHT,
    HALLUCINATION_IDENTIFIER_BONUS,
    HALLUCINATION_LLM_WEIGHT,
    HALLUCINATION_MAX_CONFIDENCE,
    HALLUCINATION_MIN_UNGROUNDED_FOR_DETECT,
    HALLUCINATION_PER_FIELD_WEIGHT,
    HALLUCINATION_STRONG_GROUNDING_THRESHOLD,
    HALLUCINATION_THOUGHT_BONUS,
)
from agent_diagnostician.analysis.grounding import GroundingAnalyzer
from agent_diagnostician.analysis.llm.parser import is_llm_response_ok
from agent_diagnostician.analysis.llm import LLMJudge, MockLLMJudge

_IDENTIFIER_RE = re.compile(
    r"^[A-Z]{1,4}[-_]\d+|[A-Z]{2,}-\d+|\d+[A-Z]-\d+$",
    re.IGNORECASE,
)
_PRICE_FIELD_HINTS = ("price", "amount", "limit", "cost", "total", "fee")
_NUMERIC_CONFLICT_BONUS = 0.25


class HallucinationDetector(BaseDetector):
    """Detects hallucination failures in agent execution traces."""

    def __init__(self, llm_judge: LLMJudge | None = None):
        self.llm_judge = llm_judge or MockLLMJudge()

    def detect(self, trace: AgentTrace) -> DetectionResult:
        if not trace.steps:
            return self.build_result(
                failure_type=FailureType.HALLUCINATION,
                subtype=HallucinationSubtype.INSUFFICIENT_EVIDENCE.value,
                confidence_score=0.0,
                evidence=[],
                reason="No steps found in trace to analyze",
                detection_stage="none",
                fix_direction="Provide a trace with at least one tool invocation step",
            )

        best_candidate = None
        best_candidate_confidence = 0.0

        for step in trace.steps:
            if step.step_status in STEP_FAILURE_STATUSES or step.error_message is not None:
                continue
            if not step.tool_input:
                continue

            result = self._detect_step_hallucination(trace, step)

            if result.subtype == HallucinationSubtype.HALLUCINATION_DETECTED.value:
                return result
            if result.subtype == HallucinationSubtype.NO_HALLUCINATION.value:
                continue
            if result.subtype == HallucinationSubtype.INSUFFICIENT_EVIDENCE.value:
                if result.confidence_score > best_candidate_confidence:
                    best_candidate = result
                    best_candidate_confidence = result.confidence_score

        if best_candidate is not None:
            return best_candidate

        return self.build_result(
            failure_type=FailureType.HALLUCINATION,
            subtype=HallucinationSubtype.NO_HALLUCINATION.value,
            confidence_score=1.0,
            evidence=[],
            reason="No hallucination detected across all steps",
            detection_stage="none",
            fix_direction="No fix required — agent used traceable values",
        )

    def _detect_step_hallucination(self, trace: AgentTrace, step: Step) -> DetectionResult:
        prior_outputs = [
            s.tool_output for s in trace.steps if s.step_index < step.step_index
        ]

        grounding_results = GroundingAnalyzer.analyze(
            step.tool_input, trace.task, prior_outputs
        )
        summary = GroundingAnalyzer.summarize(grounding_results)

        grounding_score, grounding_evidence, conflict_bonus, identifier_ungrounded, thought_ungrounded = self._score_grounding(
            summary,
            grounding_results,
            trace.task,
            prior_outputs,
            step,
        )

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

        llm_ok = is_llm_response_ok(llm_result)
        llm_score = llm_result.get("confidence", 0.0) if llm_ok else 0.0
        llm_reason = llm_result.get("reason", "No reason provided")

        llm_evidence = [
            Evidence(
                detection_stage="3 - LLM Judge",
                signal="llm_hallucination_confidence",
                confidence_contribution=llm_score,
                explanation=f"LLM judge confidence: {llm_score:.2f} — {llm_reason}",
            )
        ]

        final_confidence = (
            grounding_score * HALLUCINATION_GROUNDING_WEIGHT
            + llm_score * HALLUCINATION_LLM_WEIGHT
        )
        final_confidence = min(HALLUCINATION_MAX_CONFIDENCE, final_confidence)

        # Grounding-only escalation only when the LLM call failed (API/parse error).
        # Do not bypass a successful uncertain/low-confidence LLM verdict — those cases
        # must flow through the blended score and insufficient_evidence funnel below.
        if not llm_ok:
            if self._grounding_only_detect(
                summary,
                grounding_score,
                prior_outputs,
                conflict_bonus,
                identifier_ungrounded,
            ):
                final_confidence = max(
                    final_confidence,
                    min(
                        HALLUCINATION_MAX_CONFIDENCE,
                        grounding_score * HALLUCINATION_GROUNDING_WEIGHT + 0.35,
                    ),
                )
                llm_reason = (
                    "Strong grounding signal: fabricated identifiers or conflicting "
                    "values not traceable to task or prior outputs"
                )

        if final_confidence >= HALLUCINATION_DETECT_THRESHOLD:
            return self.build_result(
                failure_type=FailureType.HALLUCINATION,
                subtype=HallucinationSubtype.HALLUCINATION_DETECTED.value,
                confidence_score=final_confidence,
                evidence=grounding_evidence + llm_evidence,
                reason=llm_reason,
                detection_stage="hallucination_pipeline",
                fix_direction="Ensure all parameter values are traceable to the task or prior tool outputs",
            )

        if final_confidence < HALLUCINATION_CLEAR_PASS_THRESHOLD:
            if thought_ungrounded and summary["ungrounded"] == 0:
                return self.build_result(
                    failure_type=FailureType.HALLUCINATION,
                    subtype=HallucinationSubtype.INSUFFICIENT_EVIDENCE.value,
                    confidence_score=max(final_confidence, 0.15),
                    evidence=grounding_evidence + llm_evidence,
                    reason="Thought contains ungrounded assumptions but tool input appears traceable",
                    detection_stage="hallucination_pipeline",
                    fix_direction="Verify agent reasoning matches available context",
                )
            return self.build_result(
                failure_type=FailureType.HALLUCINATION,
                subtype=HallucinationSubtype.NO_HALLUCINATION.value,
                confidence_score=1.0 - final_confidence,
                evidence=[],
                reason="No hallucination signals detected in this step",
                detection_stage="hallucination_pipeline",
                fix_direction="No fix required — values appear traceable",
            )

        return self.build_result(
            failure_type=FailureType.HALLUCINATION,
            subtype=HallucinationSubtype.INSUFFICIENT_EVIDENCE.value,
            confidence_score=final_confidence,
            evidence=grounding_evidence + llm_evidence,
            reason="Insufficient evidence to confirm or rule out hallucination",
            detection_stage="hallucination_pipeline",
            fix_direction="Gather more context or provide clearer task requirements",
        )

    def _score_grounding(
        self,
        summary: dict,
        grounding_results: dict,
        task: str,
        prior_outputs: list[Any],
        step: Step,
    ) -> tuple[float, list, float, int, bool]:
        evidence: list = []
        total_fields = summary["total_fields"]
        ungrounded_fields = summary["ungrounded_fields"]
        ungrounded_count = summary["ungrounded"]

        grounding_score = 0.0
        if total_fields > 0 and ungrounded_count > 0:
            per_field = HALLUCINATION_PER_FIELD_WEIGHT / total_fields
            grounding_score = min(
                HALLUCINATION_GROUNDING_CAP,
                per_field * ungrounded_count,
            )
            for field in ungrounded_fields:
                evidence.append(
                    Evidence(
                        detection_stage="1 - Tool Input Grounding",
                        signal="ungrounded_field",
                        confidence_contribution=per_field,
                        explanation=(
                            f"Field '{field}' contains values that cannot be traced "
                            "to the task or prior tool outputs"
                        ),
                    )
                )

        conflict_bonus = 0.0
        identifier_bonus = 0.0
        identifier_ungrounded = 0
        thought_ungrounded = False
        for field in ungrounded_fields:
            value = step.tool_input.get(field)
            if self._looks_like_identifier(value) and not self._value_in_text(value, task):
                identifier_ungrounded += 1
                identifier_bonus += HALLUCINATION_IDENTIFIER_BONUS
                evidence.append(
                    Evidence(
                        detection_stage="1b - Identifier Check",
                        signal="fabricated_identifier",
                        confidence_contribution=HALLUCINATION_IDENTIFIER_BONUS,
                        explanation=(
                            f"Field '{field}' looks like a specific identifier "
                            f"('{value}') absent from task and prior outputs"
                        ),
                    )
                )

        conflict_bonus = self._numeric_conflict_bonus(step.tool_input, prior_outputs)
        if conflict_bonus > 0:
            grounding_score += conflict_bonus
            evidence.append(
                Evidence(
                    detection_stage="1c - Numeric Conflict",
                    signal="value_conflicts_with_prior_output",
                    confidence_contribution=conflict_bonus,
                    explanation="Numeric value contradicts a price/amount from a prior tool output",
                )
            )

        grounding_score = min(
            HALLUCINATION_GROUNDING_CAP,
            grounding_score + identifier_bonus,
        )

        if step.thought is not None:
            thought_grounding = GroundingAnalyzer.analyze(
                {"thought_content": step.thought}, task, prior_outputs
            )
            thought_summary = GroundingAnalyzer.summarize(thought_grounding)
            thought_ungrounded = thought_summary["ungrounded"] > 0
            if thought_ungrounded:
                grounding_score = min(
                    HALLUCINATION_GROUNDING_CAP,
                    grounding_score + HALLUCINATION_THOUGHT_BONUS,
                )
                evidence.append(
                    Evidence(
                        detection_stage="2 - Thought Grounding",
                        signal="ungrounded_thought",
                        confidence_contribution=HALLUCINATION_THOUGHT_BONUS,
                        explanation=(
                            "Agent's thought contains values that cannot be traced "
                            f"to the task or prior tool outputs: '{step.thought[:200]}...'"
                        ),
                    )
                )
        else:
            thought_ungrounded = False

        return grounding_score, evidence, conflict_bonus, identifier_ungrounded, thought_ungrounded

    def _grounding_only_detect(
        self,
        summary: dict,
        grounding_score: float,
        prior_outputs: list[Any],
        conflict_bonus: float,
        identifier_ungrounded: int,
    ) -> bool:
        ungrounded = summary["ungrounded"]
        if conflict_bonus > 0 and ungrounded >= 1:
            return True
        if identifier_ungrounded >= 2:
            return grounding_score >= HALLUCINATION_STRONG_GROUNDING_THRESHOLD
        if identifier_ungrounded >= 1 and not prior_outputs:
            return grounding_score >= HALLUCINATION_STRONG_GROUNDING_THRESHOLD
        return False

    @staticmethod
    def _looks_like_identifier(value: Any) -> bool:
        if value is None:
            return False
        text = str(value).strip()
        if len(text) < 3:
            return False
        if _IDENTIFIER_RE.match(text):
            return True
        upper_prefixes = ("ORD", "USR", "EMP", "DR", "P", "APT", "BK", "MTG", "DOC", "CONF")
        upper_text = text.upper()
        for prefix in upper_prefixes:
            if upper_text.startswith(prefix) and any(ch.isdigit() for ch in text):
                return True
        return False

    @staticmethod
    def _value_in_text(value: Any, text: str) -> bool:
        if value is None:
            return False
        return str(value).lower() in text.lower()

    @staticmethod
    def _numeric_conflict_bonus(tool_input: dict, prior_outputs: list[Any]) -> float:
        prior_numbers: list[float] = []
        for output in prior_outputs:
            output_str = GroundingAnalyzer._flatten_to_str(output)
            for match in re.findall(r"\d+\.?\d*", output_str):
                prior_numbers.append(float(match))

        if not prior_numbers:
            return 0.0

        for field, value in tool_input.items():
            field_lower = field.lower()
            if not any(hint in field_lower for hint in _PRICE_FIELD_HINTS):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            for prior in prior_numbers:
                if abs(numeric - prior) > 1.0:
                    return _NUMERIC_CONFLICT_BONUS
        return 0.0
