# Trace model definitions
# trace.py
# Defines the internal, framework-agnostic shape of an agent trace.
# tracer.py is responsible for converting raw framework logs (LangChain,
# LangGraph, AutoGen, custom, etc.) into these models. Every detector and
# analysis module downstream only ever sees THIS shape -- never raw JSON.

from typing import Any, Optional
from pydantic import BaseModel


class ToolSpec(BaseModel):
    """One entry in available_tools (Tier 3, optional).
    Describes a tool the agent could have called, used for embedding-based
    ranking (Stage 2 of Wrong Tool Selected) and schema validation
    (Stage 1 of Invalid Parameters)."""

    name: str
    description: str
    schema_: Optional[dict[str, Any]] = None  # trailing underscore, "schema" clashes with Pydantic internals


class Step(BaseModel):
    """One tool invocation within a trace. Tier 1 fields are required,
    Tier 2/3 fields are optional and default to None -- detectors must
    handle their absence gracefully, that's the whole point of the
    fallback-level pipelines."""

    # Tier 1 -- always present
    step_index: int
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: Optional[Any] = None  # optional because a failed call may have no output

    # Tier 2 -- common, not guaranteed
    timestamp: Optional[str] = None
    error_message: Optional[str] = None
    step_status: Optional[str] = None

    # Tier 3 -- rare, needs explicit instrumentation
    thought: Optional[str] = None
    retry_count: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class AgentTrace(BaseModel):
    """Run-level container. This is the object every detector receives
    as input to detect(trace) -> DetectionResult."""

    # Tier 1 -- always present
    run_id: str
    task: str
    status: str
    total_steps: int
    final_output: Optional[Any] = None
    steps: list[Step]

    # Tier 2 -- common, not guaranteed
    total_tokens: Optional[int] = None

    # Tier 3 -- rare
    available_tools: Optional[list[ToolSpec]] = None
    constraints: Optional[list[str]] = None  # raw constraints, if framework provides them directly

    # Tier 4 -- derived, computed later by analysis modules, not from raw JSON
    constraint_list: Optional[list[dict[str, Any]]] = None