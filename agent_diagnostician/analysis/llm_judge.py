# LLM judge evaluation utilities
# analysis/llm_judge.py
# LLM-based fallback judge. Not a generic prompt executor -- exposes
# task-specific methods, each with its own prompt and structured output.
# Abstract base + Gemini implementation. Swap providers without touching
# any detector code.

from abc import ABC, abstractmethod
from typing import Any
import json
import os


# ─── Abstract Interface ────────────────────────────────────────────────────────

class LLMJudge(ABC):
    """Abstract interface every LLM provider must implement.
    Detectors only ever call these methods -- never the provider directly.
    Swapping Gemini for GPT or Claude means writing a new subclass here,
    nothing else changes."""

    @staticmethod
    def _load_prompt(prompt_name: str) -> str:
        """Load a prompt template from the prompts directory.
        
        Args:
            prompt_name: Name of prompt file (without .txt extension)
            
        Returns:
            Prompt template string
            
        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(os.path.dirname(current_dir), "prompts")
        prompt_path = os.path.join(prompts_dir, f"{prompt_name}.txt")
        
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

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

    def __init__(self, model_name: str = None):
        import google.generativeai as genai
        import os
        
        # Use model from config if not specified
        if model_name is None:
            from agent_diagnostician.config import LLM_MODEL
            model_name = LLM_MODEL

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

        # Load prompt template and format it
        template = self._load_prompt("tool_selection")
        prompt = template.format(
            task=task,
            thought_str=thought_str,
            selected_tool=selected_tool,
            tools_str=tools_str
        )

        raw = self._call(prompt)
        return self._parse_json(raw)

    def evaluate_parameter_structure(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        task: str,
    ) -> dict:
        # Load prompt template and format it
        template = self._load_prompt("parameter_structure")
        prompt = template.format(
            task=task,
            tool_name=tool_name,
            tool_input=json.dumps(tool_input, indent=2)
        )

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
        if not prior_str:
            prior_str = "None"
        thought_str = f"\nAgent's reasoning: {thought}" if thought else ""

        # Load prompt template and format it
        template = self._load_prompt("parameter_values")
        prompt = template.format(
            task=task,
            thought_str=thought_str,
            prior_str=prior_str,
            tool_input=json.dumps(tool_input, indent=2)
        )

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

        # Load prompt template and format it
        template = self._load_prompt("goal_alignment")
        prompt = template.format(
            task=task,
            thought_str=thought_str,
            score_str=score_str,
            steps_str=steps_str,
            final_output=json.dumps(final_output)
        )

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
        if not prior_str:
            prior_str = "None"
            
        thought_str = f"\nAgent's reasoning: {thought}" if thought else ""
        tools_str = "\n".join(
            f"- {t['name']}: {t.get('description', '')}"
            for t in (available_tools or [])
        )
        tools_section = f"\nAvailable tools:\n{tools_str}" if tools_str else ""

        # Load prompt template and format it
        template = self._load_prompt("hallucination_detection")
        prompt = template.format(
            task=task,
            thought_str=thought_str,
            tools_section=tools_section,
            prior_str=prior_str,
            tool_input=json.dumps(tool_input, indent=2)
        )

        raw = self._call(prompt)
        return self._parse_json(raw)