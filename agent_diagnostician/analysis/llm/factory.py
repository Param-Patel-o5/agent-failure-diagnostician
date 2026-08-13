# Factory for provider-agnostic LLM judge instances.

from __future__ import annotations

from agent_diagnostician.analysis.llm.base_judge import ProviderLLMJudge
from agent_diagnostician.analysis.llm.config import LLMJudgeConfig, config_from_env
from agent_diagnostician.analysis.llm.judge import LLMJudge, MockLLMJudge
from agent_diagnostician.analysis.llm.providers.anthropic import AnthropicProvider
from agent_diagnostician.analysis.llm.providers.gemini import GeminiProvider
from agent_diagnostician.analysis.llm.providers.openai import OpenAIProvider
from agent_diagnostician.models.enums import LLMProvider


def create_llm_judge(config: LLMJudgeConfig | None = None) -> LLMJudge:
    """Create an LLM judge for the configured provider."""
    resolved = config or config_from_env()

    if resolved.provider == LLMProvider.MOCK.value:
        return MockLLMJudge()

    provider = _build_provider(resolved)
    return ProviderLLMJudge(resolved, provider)


def create_llm_judge_from_env() -> LLMJudge:
    """Convenience wrapper using environment variables."""
    return create_llm_judge(config_from_env())


def _build_provider(config: LLMJudgeConfig):
    if config.provider == LLMProvider.GEMINI.value:
        return GeminiProvider(config)

    if config.provider == LLMProvider.OPENAI.value:
        return OpenAIProvider(config)

    if config.provider == LLMProvider.ANTHROPIC.value:
        return AnthropicProvider(config)

    supported = ", ".join(p.value for p in LLMProvider if p != LLMProvider.MOCK)
    raise ValueError(f"Unsupported LLM provider '{config.provider}'. Supported: {supported}")


def GeminiLLMJudge(model_name: str | None = None, api_key: str | None = None) -> LLMJudge:
    """Deprecated convenience constructor — prefer create_llm_judge()."""
    config = LLMJudgeConfig(
        provider=LLMProvider.GEMINI.value,
        api_key=api_key,
        model=model_name,
    )
    return create_llm_judge(config)
