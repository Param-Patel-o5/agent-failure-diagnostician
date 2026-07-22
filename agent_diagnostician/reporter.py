# Agent diagnostic reporter module
# reporter.py
# Formats DetectionResult into human-readable output.
# No detection logic here — only presentation.
# Supports: CLI (colored terminal), JSON, Markdown.

import json
from agent_diagnostician.models.result import DetectionResult
from agent_diagnostician.models.enums import ConfidenceBand


# ── ANSI color codes for CLI output ───────────────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

BAND_COLORS = {
    ConfidenceBand.CONFIRMED:            RED,
    ConfidenceBand.LIKELY:               YELLOW,
    ConfidenceBand.MAYBE:                CYAN,
    ConfidenceBand.INSUFFICIENT_EVIDENCE: GREEN,
}


class Reporter:
    """Formats and prints DetectionResult in various output formats."""

    @staticmethod
    def report(result: DetectionResult, format: str = "cli") -> str:
        """Format a DetectionResult into a string.
        
        Args:
            result: DetectionResult from any detector or classifier
            format: "cli" | "json" | "markdown"
        
        Returns:
            Formatted string ready to print or write to file
        """
        if format == "json":
            return Reporter.to_json(result)
        elif format == "markdown":
            return Reporter.to_markdown(result)
        else:
            return Reporter.to_cli(result)

    @staticmethod
    def print(result: DetectionResult, format: str = "cli") -> None:
        """Format and print a DetectionResult to stdout."""
        print(Reporter.report(result, format))

    @staticmethod
    def to_cli(result: DetectionResult) -> str:
        """Format result as colored CLI output."""
        color = BAND_COLORS.get(result.confidence_band, RESET)
        lines = []

        lines.append(f"\n{BOLD}{'─' * 60}{RESET}")
        lines.append(f"{BOLD}Agent Failure Diagnostician{RESET}")
        lines.append(f"{'─' * 60}")

        # Primary verdict
        lines.append(f"{BOLD}Failure Type:{RESET}  {color}{result.failure_type.value}{RESET}")
        lines.append(f"{BOLD}Subtype:{RESET}       {color}{result.subtype}{RESET}")
        lines.append(f"{BOLD}Confidence:{RESET}    {color}{result.confidence_score:.2f} ({result.confidence_band.value}){RESET}")
        lines.append(f"{BOLD}Stage:{RESET}         {result.detection_stage}")
        lines.append("")

        # Reason and fix
        lines.append(f"{BOLD}Reason:{RESET}")
        lines.append(f"  {result.reason}")
        lines.append("")

        if result.fix_direction:
            lines.append(f"{BOLD}Fix Direction:{RESET}")
            lines.append(f"  {result.fix_direction}")
            lines.append("")

        # Evidence
        if result.evidence:
            lines.append(f"{BOLD}Evidence:{RESET}")
            for i, ev in enumerate(result.evidence, 1):
                lines.append(f"  {i}. [{ev.detection_stage}] {ev.signal}")
                lines.append(f"     {ev.explanation}")
                lines.append(f"     Contribution: {ev.confidence_contribution:.2f}")
            lines.append("")

        # Secondary evidence
        if result.secondary_evidence:
            sec = result.secondary_evidence
            lines.append(f"{BOLD}Secondary Signal:{RESET}")
            lines.append(f"  {sec.subtype} (confidence: {sec.confidence_score:.2f})")
            lines.append(f"  {sec.reason}")
            lines.append("")

        lines.append(f"{'─' * 60}\n")
        return "\n".join(lines)

    @staticmethod
    def to_json(result: DetectionResult) -> str:
        """Format result as JSON string."""
        data = {
            "failure_type": result.failure_type.value,
            "subtype": result.subtype,
            "confidence_score": result.confidence_score,
            "confidence_band": result.confidence_band.value,
            "detection_stage": result.detection_stage,
            "reason": result.reason,
            "fix_direction": result.fix_direction,
            "evidence": [
                {
                    "stage": ev.detection_stage,
                    "signal": ev.signal,
                    "confidence_contribution": ev.confidence_contribution,
                    "explanation": ev.explanation,
                }
                for ev in result.evidence
            ],
            "secondary_evidence": {
                "subtype": result.secondary_evidence.subtype,
                "confidence_score": result.secondary_evidence.confidence_score,
                "reason": result.secondary_evidence.reason,
            } if result.secondary_evidence else None,
        }
        return json.dumps(data, indent=2)

    @staticmethod
    def to_markdown(result: DetectionResult) -> str:
        """Format result as Markdown — useful for reports or GitHub issues."""
        lines = []

        lines.append("## Agent Failure Diagnosis\n")
        lines.append(f"| Field | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| **Failure Type** | `{result.failure_type.value}` |")
        lines.append(f"| **Subtype** | `{result.subtype}` |")
        lines.append(f"| **Confidence** | {result.confidence_score:.2f} — *{result.confidence_band.value}* |")
        lines.append(f"| **Detection Stage** | {result.detection_stage} |")
        lines.append("")

        lines.append(f"### Reason\n{result.reason}\n")

        if result.fix_direction:
            lines.append(f"### Fix Direction\n{result.fix_direction}\n")

        if result.evidence:
            lines.append("### Evidence\n")
            for i, ev in enumerate(result.evidence, 1):
                lines.append(f"{i}. **[{ev.detection_stage}]** `{ev.signal}`")
                lines.append(f"   - {ev.explanation}")
                lines.append(f"   - Contribution: `{ev.confidence_contribution:.2f}`")
            lines.append("")

        if result.secondary_evidence:
            sec = result.secondary_evidence
            lines.append("### Secondary Signal\n")
            lines.append(f"- **{sec.subtype}** (confidence: {sec.confidence_score:.2f})")
            lines.append(f"- {sec.reason}")

        return "\n".join(lines)