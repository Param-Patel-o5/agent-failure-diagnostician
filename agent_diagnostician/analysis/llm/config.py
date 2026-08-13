# LLM judge configuration — provider-agnostic settings.

from __future__ import annotations

import os
from dataclasses import dataclass, field

from agent_diagnostician.models.enums import LLMProvider


DEFAULT_MODELS: dict[str, str] = {
    LLMProvider.GEMINI.value: "gemini-flash-latest",
    LLMProvider.OPENAI.value: "gpt-4o-mini",
    LLMProvider.ANTHROPIC.value: "claude-3-5-haiku-latest",
}

PROVIDER_ENV_KEYS: dict[str, str] = {
    LLMProvider.GEMINI.value: "GEMINI_API_KEY",
    LLMProvider.OPENAI.value: "OPENAI_API_KEY",
    LLMProvider.ANTHROPIC.value: "ANTHROPIC_API_KEY",
}


@dataclass
class LLMJudgeConfig:
    """Provider-agnostic configuration for the LLM judge."""

    provider: str = LLMProvider.GEMINI.value
    api_key: str | None = None
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    max_retries: int = 3
    timeout_sec: float = 30.0
    strict: bool = False

    def resolved_model(self) -> str:
        if self.model:
            return self.model
        return DEFAULT_MODELS.get(self.provider, DEFAULT_MODELS[LLMProvider.GEMINI.value])

    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key

        generic = os.getenv("LLM_API_KEY")
        if generic:
            return generic

        env_key = PROVIDER_ENV_KEYS.get(self.provider)
        if env_key:
            provider_key = os.getenv(env_key)
            if provider_key:
                return provider_key

        raise ValueError(
            f"No API key found for provider '{self.provider}'. "
            f"Set LLM_API_KEY or {PROVIDER_ENV_KEYS.get(self.provider, 'PROVIDER_API_KEY')}."
        )


def config_from_env() -> LLMJudgeConfig:
    """Build config from environment variables."""
    provider = os.getenv("LLM_PROVIDER", LLMProvider.GEMINI.value).lower()
    return LLMJudgeConfig(
        provider=provider,
        api_key=os.getenv("LLM_API_KEY") or _provider_key_from_env(provider),
        model=os.getenv("LLM_MODEL"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
        timeout_sec=float(os.getenv("LLM_TIMEOUT_SEC", "30")),
    )


def _provider_key_from_env(provider: str) -> str | None:
    env_key = PROVIDER_ENV_KEYS.get(provider)
    return os.getenv(env_key) if env_key else None
