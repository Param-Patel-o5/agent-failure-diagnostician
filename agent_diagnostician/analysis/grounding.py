# Grounding analysis utilities
# analysis/grounding.py
# Checks whether values in tool_input can be traced back to the task
# or prior tool outputs. Answers: where did this value come from?
# Returns Direct / Derived / Ungrounded per value -- never decides
# if it's a failure, that's the detector's job.

from typing import Any
from difflib import SequenceMatcher


class GroundingAnalyzer:
    """Traces parameter values back to their origin.
    Used in Stage 3 of Tool Use (Incorrect Parameter Values) to determine
    whether each value in tool_input is justifiable."""

    # Fuzzy match threshold -- how similar a value needs to be to a
    # source string to count as "found". 0.8 = 80% string similarity.
    # Loose enough to catch reformatted/partial matches, tight enough
    # to avoid false positives.
    FUZZY_THRESHOLD = 0.8

    # How close a numeric value needs to be to count as derived
    # (handles floating point rounding, e.g. 54.36 vs 54.3599...)
    NUMERIC_TOLERANCE = 0.01

    @staticmethod
    def analyze(
        tool_input: dict[str, Any],
        task: str,
        prior_outputs: list[Any],
    ) -> dict[str, dict]:
        """For each value in tool_input, determine its origin.
        
        Args:
            tool_input: the parameters passed to the current tool call
            task: the original run-level task string
            prior_outputs: list of tool_output values from all prior steps
        
        Returns:
            dict mapping each field name to its grounding result:
            {
                'field_name': {
                    'classification': 'direct' | 'derived' | 'ungrounded',
                    'source': str (where it was found, if grounded),
                    'confidence': float (0-1)
                },
                ...
            }
        """
        results = {}

        for field, value in tool_input.items():
            result = GroundingAnalyzer._classify_value(
                value=value,
                task=task,
                prior_outputs=prior_outputs,
            )
            results[field] = result

        return results

    @staticmethod
    def _classify_value(
        value: Any,
        task: str,
        prior_outputs: list[Any],
    ) -> dict:
        """Classify one value as direct, derived, or ungrounded."""

        str_value = str(value)

        # Step 1: Check direct match against task string
        task_score = GroundingAnalyzer._fuzzy_match(str_value, task)
        if task_score >= GroundingAnalyzer.FUZZY_THRESHOLD:
            return {
                "classification": "direct",
                "source": "task",
                "confidence": task_score,
            }

        # Step 2: Check direct match against any prior tool output
        for i, output in enumerate(prior_outputs):
            output_str = GroundingAnalyzer._flatten_to_str(output)
            output_score = GroundingAnalyzer._fuzzy_match(str_value, output_str)
            if output_score >= GroundingAnalyzer.FUZZY_THRESHOLD:
                return {
                    "classification": "direct",
                    "source": f"step_{i}_tool_output",
                    "confidence": output_score,
                }

        # Step 3: Check if value is numerically derived from task or prior outputs
        if GroundingAnalyzer._is_numeric(value):
            all_sources = [task] + [
                GroundingAnalyzer._flatten_to_str(o) for o in prior_outputs
            ]
            is_derived, source = GroundingAnalyzer._check_numeric_derivation(
                float(value), all_sources
            )
            if is_derived:
                return {
                    "classification": "derived",
                    "source": source,
                    "confidence": 0.75,  # derived values carry slightly less confidence
                }

        # Step 4: Nothing found -- ungrounded
        return {
            "classification": "ungrounded",
            "source": None,
            "confidence": 0.0,
        }

    @staticmethod
    def _fuzzy_match(value: str, source: str) -> float:
        """Compute fuzzy string similarity between a value and a source string.
        Uses SequenceMatcher -- good for catching partial matches and
        reformatted versions of the same value."""
        if not value or not source:
            return 0.0
        # Check if value appears as substring first (fast path)
        if value.lower() in source.lower():
            return 1.0
        # Otherwise compute full similarity ratio
        return SequenceMatcher(None, value.lower(), source.lower()).ratio()

    @staticmethod
    def _is_numeric(value: Any) -> bool:
        """Check if a value can be treated as a number."""
        try:
            float(str(value))
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _check_numeric_derivation(
        value: float, sources: list[str]
    ) -> tuple[bool, str]:
        """Check if a numeric value could have been computed from numbers
        found in the source texts. Extracts all numbers from all sources
        and checks simple arithmetic combinations.
        
        Not exhaustive -- catches the most common derivation patterns:
        multiplication, addition, subtraction, division, percentage.
        """
        import re

        # Extract all numbers from all source strings
        all_numbers = []
        for source in sources:
            found = re.findall(r"\d+\.?\d*", source)
            all_numbers.extend([float(n) for n in found])

        if not all_numbers:
            return False, ""

        # Check if value itself appears numerically in sources
        for n in all_numbers:
            if abs(n - value) <= GroundingAnalyzer.NUMERIC_TOLERANCE:
                return True, "direct numeric match in context"

        # Check simple two-number arithmetic combinations
        for i, a in enumerate(all_numbers):
            for b in all_numbers[i:]:
                candidates = {
                    "multiplication": a * b,
                    "addition": a + b,
                    "subtraction": abs(a - b),
                    "division": a / b if b != 0 else None,
                    "percentage": (a * b) / 100,
                }
                for operation, result in candidates.items():
                    if result is not None and abs(result - value) <= GroundingAnalyzer.NUMERIC_TOLERANCE:
                        return True, f"derived via {operation} of values in context"

        return False, ""

    @staticmethod
    def _flatten_to_str(output: Any) -> str:
        """Convert any tool_output shape (dict, list, string, number)
        into a flat string for fuzzy matching against."""
        if isinstance(output, dict):
            return " ".join(str(v) for v in output.values())
        elif isinstance(output, list):
            return " ".join(str(item) for item in output)
        elif output is None:
            return ""
        else:
            return str(output)

    @staticmethod
    def summarize(grounding_results: dict[str, dict]) -> dict:
        """Summarize grounding results across all fields.
        Useful for the detector to get a quick overview before
        deciding confidence.
        
        Returns:
            {
                'total_fields': int,
                'direct': int,
                'derived': int,
                'ungrounded': int,
                'ungrounded_fields': list of str (field names that failed)
            }
        """
        summary = {
            "total_fields": len(grounding_results),
            "direct": 0,
            "derived": 0,
            "ungrounded": 0,
            "ungrounded_fields": [],
        }

        for field, result in grounding_results.items():
            classification = result["classification"]
            summary[classification] += 1
            if classification == "ungrounded":
                summary["ungrounded_fields"].append(field)

        return summary