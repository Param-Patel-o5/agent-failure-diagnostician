# Provider abstraction — raw completion only.

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProviderClient(ABC):
    """Send a prompt to an LLM backend and return raw text."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        timeout_sec: float,
    ) -> str:
        raise NotImplementedError
