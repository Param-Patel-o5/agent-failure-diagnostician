# LLM judge evaluation utilities
# analysis/llm_judge.py
# LLM-based fallback judge. Not a generic prompt executor -- exposes
# task-specific methods, each with its own prompt and structured output.
# Abstract base + Gemini implementation. Swap providers without touching
# any detector code.

from abc import ABC, abstractmethod
from typing import Any
import json


# ─── Abstract Interface ────────────────────────────────────────────────────────

class LLMJudge(ABC):
    """Abstract interface every LLM provider must implement.
    Detectors only ever call these methods -- never the provider directly.
    Swapping Gemini for GPT or Claude means writing a new subclass here,
    nothing else changes."""

    @abstractmethod
    def evaluate_wrong_tool(
        self,
        task: str,
        selected_tool: str,
        available_tools: list[dict],
        thought: str | None = None,
    ) -> dict:
        """Judge whether the selected tool was appropriate for the task.
        
        Returns:
            {
                'verdict': 'correct' | 'incorrect' | 'uncertain',
                'confidence': float (0-1),
                'reason': str
            }
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_parameter_structure(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        task: str,
    ) -> dict:
        """Judge whether tool_input is structurally valid for this tool.
        Judges structure only -- not whether values are correct.
        
        Returns:
            {
                'verdict': 'valid' | 'invalid' | 'uncertain',
                'confidence': float,
                'reason': str,
                'issues': list of str
            }
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_parameter_values(
        self,
        task: str,
        tool_input: dict[str, Any],
        prior_outputs: list[Any],
        thought: str | None = None,
    ) -> dict:
        """Judge whether parameter values are logically justified.
        
        Returns:
            {
                'verdict': 'justified' | 'unjustified' | 'uncertain',
                'confidence': float,
                'reason': str,
                'suspicious_fields': list of str
            }
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_goal_alignment(
        self,
        task: str,
        final_output: Any,
        steps: list[dict],
        thought: str | None = None,
        embedding_score: float | None = None,
    ) -> dict:
        """Judge whether the agent actually solved the stated task.
        
        Returns:
            {
                'verdict': 'correct' | 'misinterpreted' | 'uncertain',
                'confidence': float,
                'reason': str
            }
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_hallucination(
        self,
        task: str,
        tool_input: dict,
        prior_outputs: list[Any],
        thought: str | None = None,
        available_tools: list[dict] | None = None,
    ) -> dict:
        """Judge whether the agent hallucinated values in tool_input
        or thought field.
        
        Returns:
            {
                'confidence': float (0-1),
                'reason': str
            }
        """
        raise NotImplementedError


# ─── Mock Implementation (for development/testing without API) ─────────────────

class MockLLMJudge(LLMJudge):
    """Returns hardcoded 'uncertain' for every call.
    Use during development so detectors can be tested end-to-end
    without a real API key. Replace with GeminiLLMJudge for real runs."""

    def evaluate_wrong_tool(self, task, selected_tool, available_tools, thought=None):
        return {"verdict": "uncertain", "confidence": 0.0, "reason": "mock judge"}

    def evaluate_parameter_structure(self, tool_name, tool_input, task):
        return {"verdict": "uncertain", "confidence": 0.0, "reason": "mock judge", "issues": []}

    def evaluate_parameter_values(self, task, tool_input, prior_outputs, thought=None):
        return {"verdict": "uncertain", "confidence": 0.0, "reason": "mock judge", "suspicious_fields": []}

    def evaluate_goal_alignment(self, task, final_output, steps, thought=None, embedding_score=None):
        return {"verdict": "uncertain", "confidence": 0.0, "reason": "mock judge"}

    def evaluate_hallucination(self, task, tool_input, prior_outputs, thought=None, available_tools=None):
        return {"confidence": 0.0, "reason": "mock judge"}


# ─── Gemini Implementation ─────────────────────────────────────────────────────

class GeminiLLMJudge(LLMJudge):
    """Real LLM judge using Google Gemini API.
    Set your API key via environment variable GEMINI_API_KEY before using."""

    def __init__(self, model_name: str = "gemini-1.5-flash-latest"):
        import google.generativeai as genai
        import os

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Get a free key from https://aistudio.google.com/app/apikey"
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def _call(self, prompt: str) -> str:
        """Send prompt to Gemini, return raw text response."""
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def _parse_json(self, raw: str) -> dict:
        """Parse JSON from LLM response. Strips markdown fences if present."""
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # If JSON parsing fails, return uncertain so detector can fall back
            return {"verdict": "uncertain", "confidence": 0.0, "reason": f"failed to parse LLM response: {raw[:100]}"}

    def evaluate_wrong_tool(
        self,
        task: str,
        selected_tool: str,
        available_tools: list[dict],
        thought: str | None = None,
    ) -> dict:
        tools_str = "\n".join(
            f"- {t['name']}: {t.get('description', 'no description')}"
            for t in available_tools
        )
        thought_str = f"\nAgent's reasoning: {thought}" if thought else ""

        prompt = f"""You are evaluating whether an AI agent selected the correct tool for a task.

Task: {task}{thought_str}
Tool selected: {selected_tool}

Available tools:
{tools_str}

Was the selected tool appropriate for this task?

Respond ONLY with a JSON object, no explanation outside it:
{{
  "verdict": "correct" or "incorrect" or "uncertain",
  "confidence": <float between 0 and 1>,
  "reason": "<one sentence explanation>"
}}"""

        raw = self._call(prompt)
        return self._parse_json(raw)

    def evaluate_parameter_structure(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        task: str,
    ) -> dict:
        prompt = f"""You are evaluating whether parameters passed to a tool are structurally valid.

Task: {task}
Tool: {tool_name}
Parameters passed:
{json.dumps(tool_input, indent=2)}

Judge ONLY structural validity -- are the field names reasonable for this tool?
Are the types sensible? Do not judge whether the values are correct.

Respond ONLY with a JSON object:
{{
  "verdict": "valid" or "invalid" or "uncertain",
  "confidence": <float between 0 and 1>,
  "reason": "<one sentence>",
  "issues": ["<issue 1>", "<issue 2>"]
}}"""

        raw = self._call(prompt)
        return self._parse_json(raw)

    def evaluate_parameter_values(
        self,
        task: str,
        tool_input: dict[str, Any],
        prior_outputs: list[Any],
        thought: str | None = None,
    ) -> dict:
        prior_str = "\n".join(
            f"Step {i} output: {json.dumps(o)}"
            for i, o in enumerate(prior_outputs)
        )
        thought_str = f"\nAgent's reasoning: {thought}" if thought else ""

        prompt = f"""You are evaluating whether parameter values passed to a tool are logically justified.

Task: {task}{thought_str}

Prior tool outputs available to the agent:
{prior_str if prior_str else "None"}

Parameters the agent passed:
{json.dumps(tool_input, indent=2)}

Can each parameter value be traced back to the task or prior outputs?
Are the values logically correct given the available information?

Respond ONLY with a JSON object:
{{
  "verdict": "justified" or "unjustified" or "uncertain",
  "confidence": <float between 0 and 1>,
  "reason": "<one sentence>",
  "suspicious_fields": ["<field name>", ...]
}}"""

        raw = self._call(prompt)
        return self._parse_json(raw)

    def evaluate_goal_alignment(
        self,
        task: str,
        final_output: Any,
        steps: list[dict],
        thought: str | None = None,
        embedding_score: float | None = None,
    ) -> dict:
        steps_str = "\n".join(
            f"Step {s.get('step_index', i)}: {s.get('tool_name')} "
            f"input={json.dumps(s.get('tool_input'))} "
            f"output={json.dumps(s.get('tool_output'))}"
            for i, s in enumerate(steps)
        )
        thought_str = f"\nAgent's reasoning: {thought}" if thought else ""
        score_str = f"\nSemantic similarity (task vs output): {embedding_score:.2f}" if embedding_score is not None else ""

        prompt = f"""You are evaluating whether an AI agent successfully completed the task it was given.

Task: {task}{thought_str}{score_str}

Steps taken:
{steps_str}

Final output: {json.dumps(final_output)}

Did the agent solve the correct task? Or did it solve a different problem,
misinterpret the task, or produce a logically incorrect result?

Respond ONLY with a JSON object:
{{
  "verdict": "correct" or "misinterpreted" or "uncertain",
  "confidence": <float between 0 and 1>,
  "reason": "<one sentence>"
}}"""

        raw = self._call(prompt)
        return self._parse_json(raw)

    def evaluate_hallucination(
        self,
        task: str,
        tool_input: dict[str, Any],
        prior_outputs: list[Any],
        thought: str | None = None,
        available_tools: list[dict] | None = None,
    ) -> dict:
        prior_str = "\n".join(
            f"Step {i} output: {json.dumps(o)}"
            for i, o in enumerate(prior_outputs)
        )
        thought_str = f"\nAgent's reasoning: {thought}" if thought else ""
        tools_str = "\n".join(
            f"- {t['name']}: {t.get('description', '')}"
            for t in (available_tools or [])
        )
        tools_section = f"\nAvailable tools:\n{tools_str}" if tools_str else ""

        prompt = f"""You are evaluating whether an AI agent hallucinated values in its tool call or reasoning.

Task: {task}{thought_str}{tools_section}

Prior tool outputs available to the agent:
{prior_str if prior_str else "None"}

Parameters the agent passed to the tool:
{json.dumps(tool_input, indent=2)}

Hallucination means the agent used specific values (IDs, names, numbers,
facts) that cannot be traced to the task or any prior tool output, and
are not reasonable defaults for the tool type.

Note: common defaults like "metric", "json", "true", "en" are not
hallucinations even if not in the task. Specific identifiers like
flight IDs, order numbers, user IDs that appear nowhere in context
ARE hallucinations.

How confident are you (0.0 to 1.0) that the agent hallucinated?
0.0 = definitely not hallucinated
1.0 = definitely hallucinated

Respond ONLY with a JSON object:
{{
  "confidence": <float between 0 and 1>,
  "reason": "<one sentence explanation>"
}}"""

        raw = self._call(prompt)
        return self._parse_json(raw)