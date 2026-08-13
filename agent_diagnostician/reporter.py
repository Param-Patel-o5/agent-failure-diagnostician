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
    def confidence_explanation(confidence_score: float, confidence_band: str) -> str:
        """Convert confidence score to plain English explanation.
        
        Args:
            confidence_score: Numeric confidence (0.0-1.0)
            confidence_band: ConfidenceBand enum value
            
        Returns:
            Human-readable explanation of what this confidence means
        """
        if confidence_band == ConfidenceBand.CONFIRMED.value:
            return f"Very high confidence ({confidence_score:.0%}) — multiple strong signals detected"
        elif confidence_band == ConfidenceBand.LIKELY.value:
            return f"High confidence ({confidence_score:.0%}) — clear evidence found"
        elif confidence_band == ConfidenceBand.MAYBE.value:
            return f"Moderate confidence ({confidence_score:.0%}) — some evidence found, but not definitive"
        elif confidence_band == ConfidenceBand.INSUFFICIENT_EVIDENCE.value:
            return f"Low confidence ({confidence_score:.0%}) — limited evidence available"
        else:
            return f"Confidence: {confidence_score:.0%}"

    @staticmethod
    def report(result: DetectionResult, format: str = "cli", detector_status: dict = None, all_failures: list = None) -> str:
        """Format a DetectionResult into a string.
        
        Args:
            result: DetectionResult from any detector or classifier
            format: "cli" | "json" | "markdown"
            detector_status: Optional dict with 'ran' and 'skipped' lists of detector names
            all_failures: Optional list of all DetectionResults that detected failures
        
        Returns:
            Formatted string ready to print or write to file
        """
        if format == "json":
            return Reporter.to_json(result, detector_status, all_failures)
        elif format == "markdown":
            return Reporter.to_markdown(result, detector_status, all_failures)
        else:
            return Reporter.to_cli(result, detector_status, all_failures)

    @staticmethod
    def print(result: DetectionResult, format: str = "cli", detector_status: dict = None, all_failures: list = None) -> None:
        """Format and print a DetectionResult to stdout."""
        print(Reporter.report(result, format, detector_status, all_failures))

    @staticmethod
    def to_cli(result: DetectionResult, detector_status: dict = None, all_failures: list = None) -> str:
        """Format result as colored CLI output."""
        color = BAND_COLORS.get(result.confidence_band, RESET)
        lines = []

        lines.append(f"\n{BOLD}{'─' * 60}{RESET}")
        lines.append(f"{BOLD}Agent Failure Diagnostician{RESET}")
        lines.append(f"{'─' * 60}")

        # Primary verdict
        lines.append(f"{BOLD}Failure Type:{RESET}  {color}{result.failure_type.value}{RESET}")
        lines.append(f"{BOLD}Subtype:{RESET}       {color}{result.subtype}{RESET}")
        
        # Confidence with explanation
        confidence_explanation = Reporter.confidence_explanation(result.confidence_score, result.confidence_band.value)
        lines.append(f"{BOLD}Confidence:{RESET}    {color}{confidence_explanation}{RESET}")
        lines.append(f"{BOLD}Stage:{RESET}         {result.detection_stage}")
        lines.append("")

        # Multi-failure summary (if multiple failures detected)
        if all_failures and len(all_failures) > 1:
            lines.append(f"{BOLD}Multiple Failures Detected:{RESET}")
            # Sort by confidence, show all above threshold
            sorted_failures = sorted(all_failures, key=lambda f: f.confidence_score, reverse=True)
            for i, failure in enumerate(sorted_failures):
                is_primary = failure.failure_type == result.failure_type and failure.subtype == result.subtype
                marker = f"{GREEN}→{RESET}" if is_primary else f"{YELLOW}•{RESET}"
                lines.append(f"  {marker} {failure.failure_type.value} - {failure.subtype} ({failure.confidence_score:.2f})")
            lines.append("")

        # Detector status (if provided)
        if detector_status:
            ran = detector_status.get('ran', [])
            skipped = detector_status.get('skipped', [])
            
            if ran:
                lines.append(f"{BOLD}Detectors Ran:{RESET}")
                lines.append(f"  {GREEN}✓{RESET} " + f", {GREEN}✓{RESET} ".join(ran))
                lines.append("")
            
            if skipped:
                lines.append(f"{BOLD}Detectors Skipped:{RESET}")
                lines.append(f"  {YELLOW}−{RESET} " + f", {YELLOW}−{RESET} ".join(skipped))
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
    def to_json(result: DetectionResult, detector_status: dict = None, all_failures: list = None) -> str:
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
        
        # Add detector status if provided
        if detector_status:
            data["detector_status"] = detector_status
        
        # Add all failures if provided
        if all_failures:
            data["all_failures"] = [
                {
                    "failure_type": f.failure_type.value,
                    "subtype": f.subtype,
                    "confidence_score": f.confidence_score,
                    "confidence_band": f.confidence_band.value,
                }
                for f in all_failures
            ]
        
        return json.dumps(data, indent=2)

    @staticmethod
    def to_markdown(result: DetectionResult, detector_status: dict = None, all_failures: list = None) -> str:
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

        # Multi-failure summary (if multiple failures detected)
        if all_failures and len(all_failures) > 1:
            lines.append("### All Detected Failures\n")
            sorted_failures = sorted(all_failures, key=lambda f: f.confidence_score, reverse=True)
            lines.append("| Failure Type | Subtype | Confidence | Primary |")
            lines.append("|---|---|---|---|")
            for failure in sorted_failures:
                is_primary = failure.failure_type == result.failure_type and failure.subtype == result.subtype
                primary_marker = "✓" if is_primary else ""
                lines.append(f"| `{failure.failure_type.value}` | `{failure.subtype}` | {failure.confidence_score:.2f} | {primary_marker} |")
            lines.append("")

        # Detector status (if provided)
        if detector_status:
            ran = detector_status.get('ran', [])
            skipped = detector_status.get('skipped', [])
            
            lines.append("### Detector Status\n")
            if ran:
                lines.append("**Ran:** " + ", ".join(f"`{d}`" for d in ran) + "\n")
            if skipped:
                lines.append("**Skipped:** " + ", ".join(f"`{d}`" for d in skipped) + "\n")

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