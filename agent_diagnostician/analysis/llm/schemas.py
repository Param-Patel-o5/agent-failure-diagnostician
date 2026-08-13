# Pydantic schemas for LLM judge responses.

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from agent_diagnostician.models.enums import (
    ContextLossVerdict,
    GoalAlignmentVerdict,
    ParameterStructureVerdict,
    ParameterValuesVerdict,
    PrematureTerminationVerdict,
    ToolSelectionVerdict,
)


class ToolSelectionResponse(BaseModel):
    verdict: Literal[
        ToolSelectionVerdict.CORRECT.value,
        ToolSelectionVerdict.INCORRECT.value,
        ToolSelectionVerdict.UNCERTAIN.value,
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class ParameterStructureResponse(BaseModel):
    verdict: Literal[
        ParameterStructureVerdict.VALID.value,
        ParameterStructureVerdict.INVALID.value,
        ParameterStructureVerdict.UNCERTAIN.value,
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    issues: list[str] = Field(default_factory=list)


class ParameterValuesResponse(BaseModel):
    verdict: Literal[
        ParameterValuesVerdict.JUSTIFIED.value,
        ParameterValuesVerdict.UNJUSTIFIED.value,
        ParameterValuesVerdict.UNCERTAIN.value,
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    suspicious_fields: list[str] = Field(default_factory=list)


class GoalAlignmentResponse(BaseModel):
    verdict: Literal[
        GoalAlignmentVerdict.CORRECT.value,
        GoalAlignmentVerdict.MISINTERPRETED.value,
        GoalAlignmentVerdict.UNCERTAIN.value,
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class HallucinationResponse(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, value):
        if value is None:
            return 0.0
        return max(0.0, min(1.0, float(value)))


class ContextLossResponse(BaseModel):
    verdict: Literal[
        ContextLossVerdict.CONTEXT_LOST.value,
        ContextLossVerdict.NO_CONTEXT_LOSS.value,
        ContextLossVerdict.UNCERTAIN.value,
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class PrematureTerminationResponse(BaseModel):
    verdict: Literal[
        PrematureTerminationVerdict.PREMATURE.value,
        PrematureTerminationVerdict.COMPLETE.value,
        PrematureTerminationVerdict.UNCERTAIN.value,
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
