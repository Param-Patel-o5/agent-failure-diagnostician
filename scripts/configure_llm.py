#!/usr/bin/env python3
"""Configure and validate the LLM judge for Agent Diagnostician.

Examples:
    python scripts/configure_llm.py --show
    python scripts/configure_llm.py --provider gemini --api-key YOUR_KEY
    python scripts/configure_llm.py --test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as `python scripts/configure_llm.py` from repo root without install.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_diagnostician.analysis.llm.config import (
    DEFAULT_MODELS,
    LLMJudgeConfig,
    PROVIDER_ENV_KEYS,
    config_from_env,
)
from agent_diagnostician.analysis.llm.factory import create_llm_judge
from agent_diagnostician.models.enums import LLMProvider


def _mask_key(key: str | None) -> str:
    if not key:
        return "(not set)"
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def _resolve_api_key(provider: str) -> str | None:
    generic = os.getenv("LLM_API_KEY")
    if generic:
        return generic
    env_key = PROVIDER_ENV_KEYS.get(provider)
    return os.getenv(env_key) if env_key else None


def show_config() -> int:
    provider = os.getenv("LLM_PROVIDER", LLMProvider.GEMINI.value).lower()
    api_key = _resolve_api_key(provider)
    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider, DEFAULT_MODELS[LLMProvider.GEMINI.value])

    print("Current LLM configuration (from environment):")
    print(f"  LLM_PROVIDER={provider}")
    print(f"  LLM_API_KEY={_mask_key(api_key)}")
    print(f"  LLM_MODEL={model}")
    for name, env_key in PROVIDER_ENV_KEYS.items():
        value = os.getenv(env_key)
        if value:
            print(f"  {env_key}={_mask_key(value)} (fallback for provider={name})")
    return 0


def print_setup_commands(provider: str, api_key: str | None, model: str | None) -> int:
    model = model or DEFAULT_MODELS.get(provider, DEFAULT_MODELS[LLMProvider.GEMINI.value])
    env_key = PROVIDER_ENV_KEYS.get(provider, "LLM_API_KEY")

    print(f"Set these environment variables for provider '{provider}':")
    print()
    print("PowerShell:")
    print(f'  $env:LLM_PROVIDER = "{provider}"')
    if api_key:
        print(f'  $env:LLM_API_KEY = "{api_key}"')
        print(f'  $env:{env_key} = "{api_key}"')
    print(f'  $env:LLM_MODEL = "{model}"')
    print()
    print("bash/zsh:")
    print(f'  export LLM_PROVIDER="{provider}"')
    if api_key:
        print(f'  export LLM_API_KEY="{api_key}"')
        print(f'  export {env_key}="{api_key}"')
    print(f'  export LLM_MODEL="{model}"')
    print()
    print("Optional provider packages:")
    if provider == LLMProvider.OPENAI.value:
        print("  pip install openai")
    elif provider == LLMProvider.ANTHROPIC.value:
        print("  pip install anthropic")
    elif provider == LLMProvider.GEMINI.value:
        print("  pip install google-generativeai")
    return 0


def test_config(use_mock: bool) -> int:
    if use_mock:
        judge = create_llm_judge(LLMJudgeConfig(provider=LLMProvider.MOCK.value))
        print("Mock judge created successfully.")
    else:
        try:
            judge = create_llm_judge(config_from_env())
        except ValueError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            print("Run with --mock to verify the package without an API key.", file=sys.stderr)
            return 1

    result = judge.evaluate_wrong_tool(
        task="Calculate 2+2",
        selected_tool="web_search",
        available_tools=[{"name": "calculator", "description": "Performs math"}],
    )
    print(f"Smoke test evaluate_wrong_tool: status={result.get('status')} verdict={result.get('verdict')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Configure LLM judge for Agent Diagnostician")
    parser.add_argument(
        "--provider",
        choices=[p.value for p in LLMProvider if p != LLMProvider.MOCK],
        help="LLM provider to configure",
    )
    parser.add_argument("--api-key", help="API key (printed in setup commands; not stored)")
    parser.add_argument("--model", help="Model override (defaults per provider)")
    parser.add_argument("--show", action="store_true", help="Show current environment configuration")
    parser.add_argument("--test", action="store_true", help="Create a judge and run a smoke test")
    parser.add_argument("--mock", action="store_true", help="Use mock judge for --test (no API key needed)")
    args = parser.parse_args()

    if args.show:
        return show_config()

    if args.provider:
        return print_setup_commands(args.provider, args.api_key, args.model)

    if args.test:
        return test_config(use_mock=args.mock)

    parser.print_help()
    print()
    print("Quick start:")
    print("  python scripts/configure_llm.py --show")
    print("  python scripts/configure_llm.py --provider gemini --api-key YOUR_KEY")
    print("  python scripts/configure_llm.py --test --mock")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
