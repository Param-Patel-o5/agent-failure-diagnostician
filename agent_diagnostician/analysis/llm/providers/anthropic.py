# Anthropic provider implementation.

from __future__ import annotations

from agent_diagnostician.analysis.llm.config import LLMJudgeConfig
from agent_diagnostician.analysis.llm.providers.base import LLMProviderClient
from agent_diagnostician.models.enums import LLMProvider


class AnthropicProvider(LLMProviderClient):
    """Anthropic messages API backend for the LLM judge."""

    def __init__(self, config: LLMJudgeConfig):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic provider requires the anthropic package. "
                "Install with: pip install anthropic"
            ) from exc

        self._config = config
        self._model_name = config.resolved_model()
        self._client = anthropic.Anthropic(api_key=config.resolved_api_key())

    @property
    def provider_name(self) -> str:
        return LLMProvider.ANTHROPIC.value

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
        message = self._client.messages.create(
            model=self._model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout_sec,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = []
        for block in message.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        if not parts:
            raise RuntimeError("Anthropic returned an empty completion")
        return "\n".join(parts).strip()
