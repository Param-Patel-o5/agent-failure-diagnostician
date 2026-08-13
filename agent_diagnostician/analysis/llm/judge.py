# LLM judge abstract interface and mock implementation.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMJudge(ABC):
    """Task-specific judge methods used by detectors."""

    @staticmethod
    def _load_prompt(prompt_name: str) -> str:
        import os

        llm_dir = os.path.dirname(os.path.abspath(__file__))
        package_dir = os.path.dirname(os.path.dirname(llm_dir))
        prompt_path = os.path.join(package_dir, "prompts", f"{prompt_name}.txt")
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        with open(prompt_path, encoding="utf-8") as handle:
            return handle.read().strip()

    @abstractmethod
    def evaluate_wrong_tool(
        self,
        task: str,
        selected_tool: str,
        available_tools: list[dict],
        thought: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluate_parameter_structure(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        task: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluate_parameter_values(
        self,
        task: str,
        tool_input: dict[str, Any],
        prior_outputs: list[Any],
        thought: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluate_goal_alignment(
        self,
        task: str,
        final_output: Any,
        steps: list[dict],
        thought: str | None = None,
        embedding_score: float | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluate_hallucination(
        self,
        task: str,
        tool_input: dict,
        prior_outputs: list[Any],
        thought: str | None = None,
        available_tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def evaluate_premature_termination(
        self,
        task: str,
        final_output: Any,
        steps: list[dict],
        thought: str | None = None,
        embedding_score: float | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


class MockLLMJudge(LLMJudge):
    """Deterministic judge for development without an API key."""

    def evaluate_wrong_tool(self, task, selected_tool, available_tools, thought=None):
        from agent_diagnostician.models.enums import LLMResponseStatus, ToolSelectionVerdict

        return {
            "status": LLMResponseStatus.OK.value,
            "verdict": ToolSelectionVerdict.UNCERTAIN.value,
            "confidence": 0.0,
            "reason": "mock judge",
        }

    def evaluate_parameter_structure(self, tool_name, tool_input, task):
        from agent_diagnostician.models.enums import LLMResponseStatus, ParameterStructureVerdict

        return {
            "status": LLMResponseStatus.OK.value,
            "verdict": ParameterStructureVerdict.UNCERTAIN.value,
            "confidence": 0.0,
            "reason": "mock judge",
            "issues": [],
        }

    def evaluate_parameter_values(self, task, tool_input, prior_outputs, thought=None):
        from agent_diagnostician.models.enums import LLMResponseStatus, ParameterValuesVerdict

        return {
            "status": LLMResponseStatus.OK.value,
            "verdict": ParameterValuesVerdict.UNCERTAIN.value,
            "confidence": 0.0,
            "reason": "mock judge",
            "suspicious_fields": [],
        }

    def evaluate_goal_alignment(self, task, final_output, steps, thought=None, embedding_score=None):
        from agent_diagnostician.models.enums import GoalAlignmentVerdict, LLMResponseStatus

        return {
            "status": LLMResponseStatus.OK.value,
            "verdict": GoalAlignmentVerdict.UNCERTAIN.value,
            "confidence": 0.0,
            "reason": "mock judge",
        }

    def evaluate_hallucination(self, task, tool_input, prior_outputs, thought=None, available_tools=None):
        from agent_diagnostician.models.enums import LLMResponseStatus

        return {
            "status": LLMResponseStatus.OK.value,
            "confidence": 0.0,
            "reason": "mock judge",
        }

    def evaluate_context_loss(
        self,
        task,
        step_index,
        tool_name,
        tool_input,
        tool_output,
        prior_outputs,
        steps,
        thought=None,
    ):
        from agent_diagnostician.models.enums import ContextLossVerdict, LLMResponseStatus

        return {
            "status": LLMResponseStatus.OK.value,
            "verdict": ContextLossVerdict.UNCERTAIN.value,
            "confidence": 0.0,
            "reason": "mock judge",
        }

    def evaluate_premature_termination(
        self, task, final_output, steps, thought=None, embedding_score=None
    ):
        from agent_diagnostician.models.enums import LLMResponseStatus, PrematureTerminationVerdict

        return {
            "status": LLMResponseStatus.OK.value,
            "verdict": PrematureTerminationVerdict.UNCERTAIN.value,
            "confidence": 0.0,
            "reason": "mock judge",
        }
