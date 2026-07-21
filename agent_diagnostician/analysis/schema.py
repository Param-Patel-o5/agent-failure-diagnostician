# Schema definitions for analysis
# analysis/schema.py
# Validates a tool_input against a schema (official or inferred from prior calls).
# Answers one narrow question: does this tool_input match this schema?
# Returns what's wrong (if anything) without deciding anything about failures.

from typing import Any, Optional


class SchemaValidator:
    """Validates tool parameters against a schema. Two modes:
    1. Official schema present → validate directly.
    2. Official schema absent → infer from ≥2 prior successful calls."""

    @staticmethod
    def validate_against_schema(
        tool_input: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Validates tool_input against an official schema.
        
        Args:
            tool_input: The actual parameters passed to the tool
            schema: JSON schema (assumes Pydantic-like structure with 
                    'type', 'required', 'properties' keys)
        
        Returns:
            {
                'valid': bool,
                'errors': list of str (what's wrong, if anything),
                'missing_fields': list of str,
                'unexpected_fields': list of str,
                'type_mismatches': list of {field: str, expected: str, actual: str}
            }
        """
        errors = []
        missing_fields = []
        unexpected_fields = []
        type_mismatches = []

        # Extract schema metadata
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})

        # Check 1: Required fields present
        for field in required_fields:
            if field not in tool_input:
                missing_fields.append(field)
                errors.append(f"Missing required field: {field}")

        # Check 2: Unexpected fields (if schema is strict)
        for field in tool_input.keys():
            if field not in properties:
                unexpected_fields.append(field)
                errors.append(f"Unexpected field: {field}")

        # Check 3: Type mismatches (simplified type checking)
        for field, value in tool_input.items():
            if field in properties:
                expected_type = properties[field].get("type")
                actual_type = SchemaValidator._get_python_type(value)

                if expected_type and not SchemaValidator._type_matches(
                    actual_type, expected_type
                ):
                    type_mismatches.append(
                        {
                            "field": field,
                            "expected": expected_type,
                            "actual": actual_type,
                        }
                    )
                    errors.append(
                        f"Type mismatch on field '{field}': "
                        f"expected {expected_type}, got {actual_type}"
                    )

        # Check 4: Enum validation (if applicable)
        for field, value in tool_input.items():
            if field in properties:
                enum_values = properties[field].get("enum")
                if enum_values and value not in enum_values:
                    errors.append(
                        f"Invalid enum value for '{field}': "
                        f"'{value}' not in {enum_values}"
                    )

        # Check 5: Numeric constraints (min/max)
        for field, value in tool_input.items():
            if field in properties:
                if isinstance(value, (int, float)):
                    minimum = properties[field].get("minimum")
                    maximum = properties[field].get("maximum")
                    if minimum is not None and value < minimum:
                        errors.append(
                            f"Value {value} for '{field}' is below minimum {minimum}"
                        )
                    if maximum is not None and value > maximum:
                        errors.append(
                            f"Value {value} for '{field}' exceeds maximum {maximum}"
                        )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "missing_fields": missing_fields,
            "unexpected_fields": unexpected_fields,
            "type_mismatches": type_mismatches,
        }

    @staticmethod
    def infer_schema_from_calls(
        prior_successful_calls: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Infers a schema from prior successful calls to the same tool.
        
        Requires ≥2 successful examples to be confident. Extracts:
        - field names that appear consistently
        - types for each field
        - which fields appeared in all calls (likely required)
        
        Args:
            prior_successful_calls: List of tool_input dicts from successful calls
        
        Returns:
            Inferred schema dict, or None if insufficient examples (<2)
        """
        if len(prior_successful_calls) < 2:
            return None

        # Track field names and types across all calls
        field_types = {}  # field_name -> set of types seen
        field_counts = {}  # field_name -> how many calls included it

        for call in prior_successful_calls:
            for field, value in call.items():
                if field not in field_types:
                    field_types[field] = set()
                    field_counts[field] = 0

                field_types[field].add(SchemaValidator._get_python_type(value))
                field_counts[field] += 1

        # Infer required fields (appeared in ALL calls)
        total_calls = len(prior_successful_calls)
        required_fields = [
            field
            for field, count in field_counts.items()
            if count == total_calls
        ]

        # Build inferred schema
        properties = {}
        for field, types_seen in field_types.items():
            # If multiple types seen, be permissive (type: any)
            if len(types_seen) == 1:
                inferred_type = list(types_seen)[0]
            else:
                inferred_type = "any"

            properties[field] = {"type": inferred_type}

        return {
            "type": "object",
            "required": required_fields,
            "properties": properties,
            "inferred": True,  # marker that this was inferred, not official
        }

    @staticmethod
    def _get_python_type(value: Any) -> str:
        """Map Python type to JSON schema type string."""
        if isinstance(value, bool):  # must check bool before int (bool is subclass of int)
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        elif value is None:
            return "null"
        else:
            return "any"

    @staticmethod
    def _type_matches(actual: str, expected: str) -> bool:
        """Check if actual type matches expected type.
        Handles some flexibility: 'integer' is acceptable for 'number', 'any' matches anything."""
        if expected == "any":
            return True
        if actual == expected:
            return True
        # integer can pass as number
        if expected == "number" and actual == "integer":
            return True
        return False