# tracer.py
# Temporary implementation to load test fixture JSON files into AgentTrace.
#
# FUTURE EXTENSION NOTE:
# This file will be extended to support:
# - LangChain callback handler traces
# - LangGraph state traces
# - AutoGen conversation traces
# - OpenAI Responses API traces
# - Custom framework traces
#
# Each framework will get its own load_<framework>() function following
# the same pattern as load_fixture(). The internal AgentTrace shape
# will not change — only the ingestion logic changes per framework.

import json
import os
from typing import Any

from agent_diagnostician.models.trace import AgentTrace, Step, ToolSpec


def load_fixture(fixture_path: str) -> AgentTrace:
    """Load a test fixture JSON file and convert it to an AgentTrace.
    
    Args:
        fixture_path: Path to the JSON fixture file
        
    Returns:
        AgentTrace Pydantic model
        
    Raises:
        FileNotFoundError: If fixture file or referenced available_tools file not found
        ValueError: If JSON is malformed
    """
    # Load fixture JSON
    if not os.path.exists(fixture_path):
        raise FileNotFoundError(f"Fixture file not found: {fixture_path}")
    
    try:
        with open(fixture_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in fixture file {fixture_path}: {e}")
    
    # Strip fields that are not part of AgentTrace schema
    # These are test metadata, not trace data
    data.pop("expected_diagnosis", None)
    data.pop("domain", None)
    
    # Handle available_tools_file (separate file reference)
    available_tools_file = data.pop("available_tools_file", None)
    if available_tools_file:
        # Load tools from the referenced file in same directory as fixture
        fixture_dir = os.path.dirname(fixture_path)
        tools_path = os.path.join(fixture_dir, available_tools_file)
        
        if not os.path.exists(tools_path):
            raise FileNotFoundError(
                f"Available tools file referenced in fixture not found: {tools_path}\n"
                f"Fixture path: {fixture_path}"
            )
        
        try:
            with open(tools_path, 'r', encoding='utf-8') as f:
                tools_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in tools file {tools_path}: {e}")
        
        # Convert tools list to ToolSpec models
        # JSON uses "schema" key, but Pydantic model uses "schema_" (trailing underscore)
        tools = []
        for tool in tools_data.get("available_tools", []):
            # Map "schema" -> "schema_" for Pydantic compatibility
            tool_spec = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "schema_": tool.get("schema"),
            }
            tools.append(tool_spec)
        
        data["available_tools"] = tools
    
    # Convert steps to Step models
    # Handle missing optional fields gracefully
    steps = []
    for step in data.get("steps", []):
        step_data = {
            "step_index": step["step_index"],
            "tool_name": step["tool_name"],
            "tool_input": step["tool_input"],
            "tool_output": step.get("tool_output"),  # May be missing
            "thought": step.get("thought"),  # May be missing
            "timestamp": step.get("timestamp"),  # May be missing
            "error_message": step.get("error_message"),  # May be missing
            "step_status": step.get("step_status"),  # May be missing
            "retry_count": step.get("retry_count"),  # May be missing
            "prompt_tokens": step.get("prompt_tokens"),  # May be missing
            "completion_tokens": step.get("completion_tokens"),  # May be missing
        }
        steps.append(step_data)
    
    data["steps"] = steps
    
    # Build AgentTrace using model_validate() for validation
    return AgentTrace.model_validate(data)