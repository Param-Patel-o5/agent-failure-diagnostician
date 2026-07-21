# analysis/constraint_extractor.py
# Extracts explicit constraints from task text.
# Three types only: numeric, categorical, structural.
# Semantic constraints (tone, style) are intentionally excluded --
# too ambiguous to extract deterministically and too unreliable to judge.
# Returns a normalized constraint list (Tier 4 derived field).

import re
from typing import Any


class ConstraintExtractor:
    """Extracts numeric, categorical, and structural constraints from task text.
    Called once at the start of GoalFailureDetector -- output is the
    constraint_list Tier 4 field that Stage 1 validates against."""

    # ── Numeric patterns ───────────────────────────────────────────────────
    # Matches: "under 5000", "below $100", "max 3", "less than 50",
    #          "within 2 days", "at least 10", "more than 500", "exactly 5"
    NUMERIC_PATTERNS = [
        (r"under\s+[\$₹€£]?([\d,]+\.?\d*)\s*(\w+)?",       "less_than"),
        (r"below\s+[\$₹€£]?([\d,]+\.?\d*)\s*(\w+)?",       "less_than"),
        (r"less\s+than\s+[\$₹€£]?([\d,]+\.?\d*)\s*(\w+)?", "less_than"),
        (r"max(?:imum)?\s+[\$₹€£]?([\d,]+\.?\d*)\s*(\w+)?","less_than_equal"),
        (r"no\s+more\s+than\s+[\$₹€£]?([\d,]+\.?\d*)\s*(\w+)?", "less_than_equal"),
        (r"within\s+([\d,]+\.?\d*)\s*(\w+)?",               "less_than_equal"),
        (r"at\s+least\s+[\$₹€£]?([\d,]+\.?\d*)\s*(\w+)?",  "greater_than_equal"),
        (r"more\s+than\s+[\$₹€£]?([\d,]+\.?\d*)\s*(\w+)?", "greater_than"),
        (r"min(?:imum)?\s+[\$₹€£]?([\d,]+\.?\d*)\s*(\w+)?","greater_than_equal"),
        (r"exactly\s+([\d,]+\.?\d*)\s*(\w+)?",              "equal"),
    ]

    # ── Categorical patterns ───────────────────────────────────────────────
    # Matches: "use Python", "don't use recursion", "only use pandas",
    #          "must use X", "avoid X", "do not use X"
    CATEGORICAL_PATTERNS = [
        (r"(?:use|using)\s+([A-Za-z][\w\s\+\#\.]*?)(?:\s+(?:only|exclusively))?(?:\.|,|$)", "must_use"),
        (r"only\s+(?:use|using)\s+([A-Za-z][\w\s\+\#\.]*?)(?:\.|,|$)",                      "must_use"),
        (r"must\s+use\s+([A-Za-z][\w\s\+\#\.]*?)(?:\.|,|$)",                                "must_use"),
        (r"(?:don't|do\s+not|avoid|without|no)\s+(?:use\s+)?([A-Za-z][\w\s\+\#\.]*?)(?:\.|,|$)", "must_not_use"),
        (r"keep\s+(?:the\s+)?([A-Za-z][\w\s_]*?)\s+(?:name|signature|interface)\s+unchanged", "keep_unchanged"),
        (r"do\s+not\s+(?:change|modify|rename)\s+(?:the\s+)?([A-Za-z][\w\s_]*?)(?:\.|,|$)", "keep_unchanged"),
    ]

    # ── Structural patterns ────────────────────────────────────────────────
    # Matches: "return JSON", "as a table", "in CSV format",
    #          "as markdown", "in bullet points"
    STRUCTURAL_KEYWORDS = {
        "json":           "return_json",
        "csv":            "return_csv",
        "markdown":       "return_markdown",
        "table":          "return_table",
        "bullet points":  "return_bullets",
        "numbered list":  "return_numbered_list",
        "plain text":     "return_plain_text",
        "html":           "return_html",
        "xml":            "return_xml",
        "yaml":           "return_yaml",
    }

    @staticmethod
    def extract(task: str) -> list[dict[str, Any]]:
        """Extract all constraints from a task string.

        Args:
            task: the run-level task string

        Returns:
            List of constraint dicts, each with:
            {
                'type': 'numeric' | 'categorical' | 'structural',
                'subtype': str (e.g. 'less_than', 'must_use', 'return_json'),
                'value': str | float (the actual constraint value),
                'unit': str | None (e.g. 'rupees', 'days' for numeric),
                'raw': str (the original matched text)
            }
        """
        constraints = []
        constraints.extend(ConstraintExtractor._extract_numeric(task))
        constraints.extend(ConstraintExtractor._extract_categorical(task))
        constraints.extend(ConstraintExtractor._extract_structural(task))
        return constraints

    @staticmethod
    def _extract_numeric(task: str) -> list[dict]:
        """Extract numeric constraints using regex patterns."""
        results = []
        task_lower = task.lower()

        for pattern, subtype in ConstraintExtractor.NUMERIC_PATTERNS:
            for match in re.finditer(pattern, task_lower):
                raw_value = match.group(1).replace(",", "")  # remove commas from "1,000"
                try:
                    value = float(raw_value)
                except ValueError:
                    continue

                unit = match.group(2).strip() if match.lastindex >= 2 and match.group(2) else None

                results.append({
                    "type": "numeric",
                    "subtype": subtype,
                    "value": value,
                    "unit": unit,
                    "raw": match.group(0).strip(),
                })

        return results

    @staticmethod
    def _extract_categorical(task: str) -> list[dict]:
        """Extract categorical constraints using regex patterns."""
        results = []

        for pattern, subtype in ConstraintExtractor.CATEGORICAL_PATTERNS:
            for match in re.finditer(pattern, task, re.IGNORECASE):
                value = match.group(1).strip().rstrip(".,")
                if len(value) < 2 or len(value) > 50:
                    continue  # skip garbage matches

                results.append({
                    "type": "categorical",
                    "subtype": subtype,
                    "value": value,
                    "unit": None,
                    "raw": match.group(0).strip(),
                })

        return results

    @staticmethod
    def _extract_structural(task: str) -> list[dict]:
        """Extract structural constraints using keyword matching."""
        results = []
        task_lower = task.lower()

        for keyword, subtype in ConstraintExtractor.STRUCTURAL_KEYWORDS.items():
            if keyword in task_lower:
                results.append({
                    "type": "structural",
                    "subtype": subtype,
                    "value": keyword,
                    "unit": None,
                    "raw": keyword,
                })

        return results

    @staticmethod
    def validate_constraint(
        constraint: dict[str, Any],
        actual_value: Any,
    ) -> dict[str, Any]:
        """Check whether a single extracted constraint is satisfied.
        Used by GoalFailureDetector Stage 1 to validate each constraint
        against the actual output.

        Args:
            constraint: one constraint dict from extract()
            actual_value: the value from final_output to check against

        Returns:
            {
                'satisfied': bool,
                'reason': str
            }
        """
        subtype = constraint["subtype"]
        expected = constraint["value"]

        # Numeric validation
        if constraint["type"] == "numeric":
            try:
                actual = float(str(actual_value).replace(",", ""))
            except (ValueError, TypeError):
                return {
                    "satisfied": False,
                    "reason": f"Could not extract numeric value from actual output '{actual_value}'",
                }

            checks = {
                "less_than":          actual < expected,
                "less_than_equal":    actual <= expected,
                "greater_than":       actual > expected,
                "greater_than_equal": actual >= expected,
                "equal":              abs(actual - expected) < 0.01,
            }
            passed = checks.get(subtype, True)
            return {
                "satisfied": passed,
                "reason": f"Numeric constraint '{constraint['raw']}': actual={actual}, expected {subtype} {expected}",
            }

        # Categorical validation -- check if value/keyword appears in output
        if constraint["type"] == "categorical":
            actual_str = str(actual_value).lower()
            value_lower = str(expected).lower()

            if subtype == "must_use":
                satisfied = value_lower in actual_str
                return {
                    "satisfied": satisfied,
                    "reason": f"Categorical constraint 'must use {expected}': {'found' if satisfied else 'not found'} in output",
                }
            elif subtype == "must_not_use":
                satisfied = value_lower not in actual_str
                return {
                    "satisfied": satisfied,
                    "reason": f"Categorical constraint 'must not use {expected}': {'not found (ok)' if satisfied else 'found in output (violation)'}",
                }
            elif subtype == "keep_unchanged":
                satisfied = value_lower in actual_str
                return {
                    "satisfied": satisfied,
                    "reason": f"Categorical constraint 'keep {expected} unchanged': {'present' if satisfied else 'missing or changed'} in output",
                }

        # Structural validation -- check if format keyword appears in output
        if constraint["type"] == "structural":
            actual_str = str(actual_value).lower()
            value_lower = str(expected).lower()
            satisfied = value_lower in actual_str
            return {
                "satisfied": satisfied,
                "reason": f"Structural constraint '{constraint['raw']}': {'detected' if satisfied else 'not detected'} in output",
            }

        return {"satisfied": True, "reason": "Unknown constraint type — skipped"}