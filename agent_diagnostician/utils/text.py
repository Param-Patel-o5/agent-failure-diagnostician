# Text processing utilities
# utils/text.py
# Shared string helpers used by multiple analysis modules.
# Lives here so GroundingAnalyzer, TokenExhaustionDetector, and any future
# module that needs fuzzy matching can import one function instead of each
# reimplementing SequenceMatcher locally.

from difflib import SequenceMatcher


def fuzzy_match(value: str, source: str) -> float:
    """Compute fuzzy string similarity between two strings.

    Uses SequenceMatcher — good for catching partial matches and
    reformatted versions of the same value.

    Returns a score between 0.0 and 1.0:
        1.0 = identical / substring match
        0.0 = completely different

    Args:
        value:  the shorter candidate string
        source: the larger context string to search within
    """
    if not value or not source:
        return 0.0
    # Fast path: exact substring match
    if value.lower() in source.lower():
        return 1.0
    # Full similarity ratio
    return SequenceMatcher(None, value.lower(), source.lower()).ratio()
