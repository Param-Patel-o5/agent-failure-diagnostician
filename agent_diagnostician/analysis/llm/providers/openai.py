# OpenAI provider implementation.

from __future__ import annotations

from agent_diagnostician.analysis.llm.config import LLMJudgeConfig
from agent_diagnostician.analysis.llm.providers.base import LLMProviderClient
from agent_diagnostician.models.enums import LLMProvider


class OpenAIProvider(LLMProviderClient):
    """OpenAI chat-completions backend for the LLM judge."""

    def __init__(self, config: LLMJudgeConfig):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI provider requires the openai package. "
                "Install with: pip install openai"
            ) from exc

        self._config = config
        self._model_name = config.resolved_model()
        self._client = OpenAI(api_key=config.resolved_api_key())

    @property
    def provider_name(self) -> str:
        return LLMProvider.OPENAI.value

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
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout_sec,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenAI returned an empty completion")
        return content.strip()
