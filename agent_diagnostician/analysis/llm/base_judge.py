# Provider-backed LLM judge — shared prompt, call, parse, validate logic.

from __future__ import annotations

import json
import time
from typing import Any, Type

from pydantic import BaseModel

from agent_diagnostician.analysis.llm.config import LLMJudgeConfig
from agent_diagnostician.analysis.llm.judge import LLMJudge
from agent_diagnostician.analysis.llm.parser import (
    classify_provider_error,
    error_dict,
    parse_failed_dict,
    parse_json_object,
    success_dict,
    validate_response,
)
from agent_diagnostician.analysis.llm.providers.base import LLMProviderClient
from agent_diagnostician.analysis.llm.schemas import (
    ContextLossResponse,
    GoalAlignmentResponse,
    HallucinationResponse,
    ParameterStructureResponse,
    ParameterValuesResponse,
    PrematureTerminationResponse,
    ToolSelectionResponse,
)
from agent_diagnostician.models.enums import (
    ContextLossVerdict,
    GoalAlignmentVerdict,
    ParameterStructureVerdict,
    ParameterValuesVerdict,
    PrematureTerminationVerdict,
    ToolSelectionVerdict,
)


class ProviderLLMJudge(LLMJudge):
    """LLM judge that delegates raw completion to a provider client."""

    def __init__(self, config: LLMJudgeConfig, provider: LLMProviderClient):
        self._config = config
        self._provider = provider

    def _run_prompt(
        self,
        prompt: str,
        schema: Type[BaseModel],
        *,
        fallback_verdict: str = ToolSelectionVerdict.UNCERTAIN.value,
        extra_defaults: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(self._config.max_retries):
            try:
                raw = self._provider.complete(
                    prompt,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                    timeout_sec=self._config.timeout_sec,
                )
                data = parse_json_object(raw)
                if data is None:
                    if attempt + 1 < self._config.max_retries:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    result = parse_failed_dict(
                        f"failed to parse LLM response: {raw[:120]}",
                        fallback_verdict=fallback_verdict,
                    )
                    if extra_defaults:
                        result.update(extra_defaults)
                    return result

                validated = validate_response(schema, data)
                if validated is None:
                    if attempt + 1 < self._config.max_retries:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    result = parse_failed_dict(
                        "LLM response did not match expected schema",
                        fallback_verdict=fallback_verdict,
                    )
                    if extra_defaults:
                        result.update(extra_defaults)
                    return result

                return success_dict(validated)

            except Exception as exc:
                last_error = exc
                error_type = classify_provider_error(exc)
                if error_type.value in {
                    "quota_exceeded",
                    "authentication",
                    "model_not_found",
                }:
                    break
                if attempt + 1 < self._config.max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                break

        if self._config.strict and last_error is not None:
            raise last_error

        error_type = classify_provider_error(last_error) if last_error else "unknown"
        result = error_dict(
            error_type,
            str(last_error) if last_error else "unknown provider error",
            fallback_verdict=fallback_verdict,
            extra=extra_defaults,
        )
        return result

    def evaluate_wrong_tool(
        self,
        task: str,
        selected_tool: str,
        available_tools: list[dict],
        thought: str | None = None,
    ) -> dict[str, Any]:
        tools_str = "\n".join(
            f"- {tool['name']}: {tool.get('description', 'no description')}"
            for tool in available_tools
        )
        thought_str = f"\nAgent's reasoning: {thought}" if thought else ""
        template = self._load_prompt("tool_selection")
        prompt = template.format(
            task=task,
            thought_str=thought_str,
            selected_tool=selected_tool,
            tools_str=tools_str,
        )
        return self._run_prompt(
            prompt,
            ToolSelectionResponse,
            fallback_verdict=ToolSelectionVerdict.UNCERTAIN.value,
        )

    def evaluate_parameter_structure(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        task: str,
    ) -> dict[str, Any]:
        template = self._load_prompt("parameter_structure")
        prompt = template.format(
            task=task,
            tool_name=tool_name,
            tool_input=json.dumps(tool_input, indent=2),
        )
        return self._run_prompt(
            prompt,
            ParameterStructureResponse,
            fallback_verdict=ParameterStructureVerdict.UNCERTAIN.value,
            extra_defaults={"issues": []},
        )

    def evaluate_parameter_values(
        self,
        task: str,
        tool_input: dict[str, Any],
        prior_outputs: list[Any],
        thought: str | None = None,
    ) -> dict[str, Any]:
        prior_str = "\n".join(
            f"Step {index} output: {json.dumps(output)}"
            for index, output in enumerate(prior_outputs)
        ) or "None"
        thought_str = f"\nAgent's reasoning: {thought}" if thought else ""
        template = self._load_prompt("parameter_values")
        prompt = template.format(
            task=task,
            thought_str=thought_str,
            prior_str=prior_str,
            tool_input=json.dumps(tool_input, indent=2),
        )
        return self._run_prompt(
            prompt,
            ParameterValuesResponse,
            fallback_verdict=ParameterValuesVerdict.UNCERTAIN.value,
            extra_defaults={"suspicious_fields": []},
        )

    def evaluate_goal_alignment(
        self,
        task: str,
        final_output: Any,
        steps: list[dict],
        thought: str | None = None,
        embedding_score: float | None = None,
    ) -> dict[str, Any]:
        steps_str = "\n".join(
            f"Step {step.get('step_index', index)}: {step.get('tool_name')} "
            f"input={json.dumps(step.get('tool_input'))} "
            f"output={json.dumps(step.get('tool_output'))}"
            for index, step in enumerate(steps)
        )
        thought_str = f"\nAgent's reasoning: {thought}" if thought else ""
        score_str = (
            f"\nSemantic similarity (task vs output): {embedding_score:.2f}"
            if embedding_score is not None
            else ""
        )
        template = self._load_prompt("goal_alignment")
        prompt = template.format(
            task=task,
            thought_str=thought_str,
            score_str=score_str,
            steps_str=steps_str,
            final_output=json.dumps(final_output),
        )
        return self._run_prompt(
            prompt,
            GoalAlignmentResponse,
            fallback_verdict=GoalAlignmentVerdict.UNCERTAIN.value,
        )

    def evaluate_hallucination(
        self,
        task: str,
        tool_input: dict,
        prior_outputs: list[Any],
        thought: str | None = None,
        available_tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        prior_str = "\n".join(
            f"Step {index} output: {json.dumps(output)}"
            for index, output in enumerate(prior_outputs)
        ) or "None"
        thought_str = f"\nAgent's reasoning: {thought}" if thought else ""
        tools_str = "\n".join(
            f"- {tool['name']}: {tool.get('description', '')}"
            for tool in (available_tools or [])
        )
        tools_section = f"\nAvailable tools:\n{tools_str}" if tools_str else ""
        template = self._load_prompt("hallucination_detection")
        prompt = template.format(
            task=task,
            thought_str=thought_str,
            tools_section=tools_section,
            prior_str=prior_str,
            tool_input=json.dumps(tool_input, indent=2),
        )
        return self._run_prompt(prompt, HallucinationResponse)

    def evaluate_context_loss(
        self,
        task: str,
        step_index: int,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: Any,
        prior_outputs: list[Any],
        steps: list[dict],
        thought: str | None = None,
    ) -> dict[str, Any]:
        steps_str = "\n".join(
            f"Step {step.get('step_index', index)}: {step.get('tool_name')} "
            f"input={json.dumps(step.get('tool_input'))} "
            f"output={json.dumps(step.get('tool_output'))}"
            for index, step in enumerate(steps)
        )
        prior_str = "\n".join(
            f"Step {index} output: {json.dumps(output)}"
            for index, output in enumerate(prior_outputs)
        ) or "None"
        thought_str = f"\nAgent's reasoning: {thought}" if thought else ""
        template = self._load_prompt("context_loss")
        prompt = template.format(
            task=task,
            thought_str=thought_str,
            steps_str=steps_str,
            step_index=step_index,
            tool_name=tool_name,
            tool_input=json.dumps(tool_input, indent=2),
            tool_output=json.dumps(tool_output),
            prior_str=prior_str,
        )
        return self._run_prompt(
            prompt,
            ContextLossResponse,
            fallback_verdict=ContextLossVerdict.UNCERTAIN.value,
        )

    def evaluate_premature_termination(
        self,
        task: str,
        final_output: Any,
        steps: list[dict],
        thought: str | None = None,
        embedding_score: float | None = None,
    ) -> dict[str, Any]:
        steps_str = "\n".join(
            f"Step {step.get('step_index', index)}: {step.get('tool_name')} "
            f"input={json.dumps(step.get('tool_input'))} "
            f"output={json.dumps(step.get('tool_output'))}"
            for index, step in enumerate(steps)
        )
        thought_str = f"\nAgent's reasoning: {thought}" if thought else ""
        score_str = (
            f"\nSemantic similarity (task vs output): {embedding_score:.2f}"
            if embedding_score is not None
            else ""
        )
        template = self._load_prompt("premature_termination")
        prompt = template.format(
            task=task,
            thought_str=thought_str,
            score_str=score_str,
            steps_str=steps_str,
            final_output=json.dumps(final_output),
        )
        return self._run_prompt(
            prompt,
            PrematureTerminationResponse,
            fallback_verdict=PrematureTerminationVerdict.UNCERTAIN.value,
        )
