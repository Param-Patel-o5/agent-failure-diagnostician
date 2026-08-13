# Grounding analysis utilities
# analysis/grounding.py
# Checks whether values in tool_input can be traced back to the task
# or prior tool outputs. Answers: where did this value come from?
# Returns Direct / Derived / Ungrounded per value -- never decides
# if it's a failure, that's the detector's job.

from typing import Any

from agent_diagnostician.config import GROUNDING_FUZZY_THRESHOLD
from agent_diagnostician.models.enums import GroundingClassification
from agent_diagnostician.utils.text import fuzzy_match as _fuzzy_match_fn


class GroundingAnalyzer:
    """Traces parameter values back to their origin.
    Used in Stage 3 of Tool Use (Incorrect Parameter Values) to determine
    whether each value in tool_input is justifiable."""

    FUZZY_THRESHOLD = GROUNDING_FUZZY_THRESHOLD

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
                field=field,
                value=value,
                task=task,
                prior_outputs=prior_outputs,
            )
            results[field] = result

        return results

    @staticmethod
    def _classify_value(
        field: str,
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
                "classification": GroundingClassification.DIRECT.value,
                "source": "task",
                "confidence": task_score,
            }

        # Normalized match (underscores/hyphens → spaces) for report names etc.
        normalized_value = str_value.replace("_", " ").replace("-", " ")
        task_lower = task.lower()
        if normalized_value.lower() in task_lower:
            return {
                "classification": GroundingClassification.DIRECT.value,
                "source": "task_normalized",
                "confidence": 0.90,
            }
        # All tokens from normalized value appear in task (e.g. sales report Q2 2026).
        tokens = [t for t in normalized_value.lower().split() if len(t) > 1]
        if tokens and all(token in task_lower for token in tokens):
            return {
                "classification": GroundingClassification.DIRECT.value,
                "source": "task_token_match",
                "confidence": 0.85,
            }

        # Step 2: Check direct match against any prior tool output
        for i, output in enumerate(prior_outputs):
            output_str = GroundingAnalyzer._flatten_to_str(output)
            output_score = GroundingAnalyzer._fuzzy_match(str_value, output_str)
            if output_score >= GroundingAnalyzer.FUZZY_THRESHOLD:
                return {
                    "classification": GroundingClassification.DIRECT.value,
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
                    "classification": GroundingClassification.DERIVED.value,
                    "source": source,
                    "confidence": 0.75,
                }

        # Step 3.5: Date/time fields often use a different format than the task text.
        field_hint = str(field).lower()
        if "date" in field_hint and GroundingAnalyzer._date_referenced_in_task(str_value, task):
            return {
                "classification": GroundingClassification.DERIVED.value,
                "source": "task_date_format",
                "confidence": 0.80,
            }
        if "time" in field_hint and GroundingAnalyzer._time_referenced_in_task(str_value, task):
            return {
                "classification": GroundingClassification.DERIVED.value,
                "source": "task_time_format",
                "confidence": 0.80,
            }

        # Step 3.6: Check if value is a reasonable default for hallucination detection
        if GroundingAnalyzer._is_reasonable_default(str_value):
            return {
                "classification": GroundingClassification.DERIVED.value,
                "source": "reasonable_default",
                "confidence": 0.85,
            }

        # Step 4: Nothing found -- ungrounded
        return {
            "classification": GroundingClassification.UNGROUNDED.value,
            "source": None,
            "confidence": 0.0,
        }

    @staticmethod
    def _date_referenced_in_task(value: str, task: str) -> bool:
        import re

        parts = value.split("-")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            year, month, day = parts
            if year in task and day in task:
                return True
        # Also accept if all numeric components from value appear in task.
        nums = re.findall(r"\d+", value)
        return bool(nums) and all(n in task for n in nums)

    @staticmethod
    def _time_referenced_in_task(value: str, task: str) -> bool:
        import re

        if value in task:
            return True
        match = re.match(r"^(\d{1,2}):(\d{2})$", value.strip())
        if not match:
            return False
        hour = int(match.group(1))
        minute = match.group(2)
        twelve_hour = hour - 12 if hour > 12 else hour
        patterns = {
            f"{hour:02d}:{minute}",
            f"{hour}:{minute}",
            f"{twelve_hour}:{minute}",
            f"{twelve_hour}{minute}",
        }
        if hour == 15:
            patterns.update({"3pm", "3 pm", "3:00pm", "15:00"})
        task_lower = task.lower()
        return any(p.lower() in task_lower for p in patterns)

    @staticmethod
    def _is_reasonable_default(value: str) -> bool:
        """Check if a value is a reasonable default that shouldn't be considered hallucination"""
        value_lower = value.lower()
        
        # Common status/reason values
        reasonable_defaults = {
            # Request/action reasons
            'customer_request', 'user_request', 'manual_request', 'admin_request',
            'system_request', 'automatic', 'scheduled', 'maintenance',
            
            # Common statuses
            'active', 'inactive', 'pending', 'completed', 'failed', 'success',
            'enabled', 'disabled', 'true', 'false', 'yes', 'no',
            
            # Common formats/units
            'json', 'xml', 'csv', 'pdf', 'txt', 'html', 'utf-8', 'utf8',
            'metric', 'imperial', 'celsius', 'fahrenheit', 'usd', 'eur', 'inr',
            
            # Language codes  
            'en', 'english', 'us', 'uk', 'in',
            
            # Common defaults
            'default', 'standard', 'normal', 'basic', 'premium',

            # Appointment / document aliases
            'follow_up', 'followup', 'follow-up',
            'nda', 'non_disclosure_agreement',
            'performance_review', 'performance review',
        }
        
        return value_lower in reasonable_defaults

    @staticmethod
    def _fuzzy_match(value: str, source: str) -> float:
        """Delegates to utils.text.fuzzy_match — shared helper, no local copy."""
        return _fuzzy_match_fn(value, source)

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
            if classification == GroundingClassification.DIRECT.value:
                summary["direct"] += 1
            elif classification == GroundingClassification.DERIVED.value:
                summary["derived"] += 1
            elif classification == GroundingClassification.UNGROUNDED.value:
                summary["ungrounded"] += 1
                summary["ungrounded_fields"].append(field)

        return summary