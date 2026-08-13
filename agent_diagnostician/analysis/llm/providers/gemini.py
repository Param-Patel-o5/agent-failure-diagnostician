# Google Gemini provider implementation.

from __future__ import annotations

import google.generativeai as genai

from agent_diagnostician.analysis.llm.config import LLMJudgeConfig
from agent_diagnostician.analysis.llm.providers.base import LLMProviderClient
from agent_diagnostician.models.enums import LLMProvider


class GeminiProvider(LLMProviderClient):
    """Gemini backend for the LLM judge."""

    def __init__(self, config: LLMJudgeConfig):
        self._config = config
        self._model_name = config.resolved_model()
        genai.configure(api_key=config.resolved_api_key())
        self._model = genai.GenerativeModel(self._model_name)

    @property
    def provider_name(self) -> str:
        return LLMProvider.GEMINI.value

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        timeout_sec: float,
    ) -> str:
        del timeout_sec  # google-generativeai uses request-level timeout differently
        response = self._model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        return response.text.strip()
