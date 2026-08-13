# Parse and validate LLM JSON responses.

from __future__ import annotations

import json
import re
from typing import Any, Type

from pydantic import BaseModel, ValidationError

from agent_diagnostician.models.enums import (
    LLMErrorType,
    LLMResponseStatus,
    ToolSelectionVerdict,
)


def strip_json_fences(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_json_object(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(strip_json_fences(raw))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def validate_response(schema: Type[BaseModel], data: dict[str, Any]) -> BaseModel | None:
    try:
        return schema.model_validate(data)
    except ValidationError:
        return None


def success_dict(model: BaseModel) -> dict[str, Any]:
    payload = model.model_dump()
    payload["status"] = LLMResponseStatus.OK.value
    return payload


def error_dict(
    error_type: LLMErrorType | str,
    reason: str,
    *,
    fallback_verdict: str = ToolSelectionVerdict.UNCERTAIN.value,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": LLMResponseStatus.ERROR.value,
        "error_type": error_type.value if isinstance(error_type, LLMErrorType) else error_type,
        "verdict": fallback_verdict,
        "confidence": 0.0,
        "reason": reason,
    }
    if extra:
        payload.update(extra)
    return payload


def parse_failed_dict(reason: str, *, fallback_verdict: str = ToolSelectionVerdict.UNCERTAIN.value) -> dict[str, Any]:
    return {
        "status": LLMResponseStatus.PARSE_FAILED.value,
        "verdict": fallback_verdict,
        "confidence": 0.0,
        "reason": reason,
    }


def is_llm_response_ok(result: dict[str, Any]) -> bool:
    return result.get("status") == LLMResponseStatus.OK.value


def classify_provider_error(exc: Exception) -> LLMErrorType:
    message = str(exc).lower()
    if "429" in message or "quota" in message or "rate limit" in message:
        return LLMErrorType.QUOTA_EXCEEDED
    if "401" in message or "403" in message or "api key" in message or "authentication" in message or "unauthorized" in message:
        return LLMErrorType.AUTHENTICATION
    if "404" in message or "not found" in message or "model" in message:
        return LLMErrorType.MODEL_NOT_FOUND
    if "timeout" in message or "timed out" in message:
        return LLMErrorType.TIMEOUT
    if "connection" in message or "network" in message:
        return LLMErrorType.CONNECTION
    return LLMErrorType.UNKNOWN
