#!/usr/bin/env python3
"""Run all local JSON fixtures and save evaluation metrics for README/docs.

Usage (from repo root):
    python scripts/run_fixture_evaluation.py
    python scripts/run_fixture_evaluation.py --output docs/evaluation

Requires LLM_PROVIDER and LLM_API_KEY (or provider-specific key) in environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_diagnostician.analysis.embeddings import EmbeddingMatcher
from agent_diagnostician.analysis.llm import create_llm_judge_from_env
from agent_diagnostician.analysis.llm.config import config_from_env
from agent_diagnostician.classifier import Classifier
from agent_diagnostician.models.enums import (
    FailureType,
    INSUFFICIENT_EVIDENCE_SUBTYPE,
    NO_FAILURE_SUBTYPE_VALUES,
    ClassifierSubtype,
)
from agent_diagnostician.tracer import load_fixture

FIXTURE_ROOT = _ROOT / "test cases"

# Map top-level fixture folders to a single detector (or None = all detectors).
FOLDER_DETECTORS: dict[str, list[FailureType] | None] = {
    "TOOL_USE FAILURE": [FailureType.TOOL_USE_FAILURE],
    "Goal Statisfiction": [FailureType.GOAL_SATISFACTION_FAILURE],
    "HALLUCINATION": [FailureType.HALLUCINATION],
    "context_loss": [FailureType.CONTEXT_LOSS],
    "premature_termination": [FailureType.PREMATURE_TERMINATION],
}

TOOL_USE_SIDEcars = {
    "available_tools__ecommerce_domain.json",
    "available_tools__document_domain.json",
    "available_tools__knowledge_domain.json",
}

SUBTYPE_ALIASES: dict[str, set[str]] = {
    "wrong_tool_selected": {"wrong_tool_selected", "wrong tool selected"},
    "invalid_parameters": {
        "invalid_parameters",
        "invalid parameters",
        "valid tool, invalid parameters",
    },
    "incorrect_parameter_values": {
        "incorrect_parameter_values",
        "incorrect parameter values",
        "valid tool, incorrect parameter values",
        "valid parameters wrong values",
    },
    "constraint_violation": {"constraint_violation", "constraint violation"},
    "task_misinterpretation": {"task_misinterpretation", "task misinterpretation"},
    "hallucination_detected": {"hallucination_detected", "hallucination detected"},
    "no_hallucination": {"no_hallucination", "no hallucination"},
    "context_loss_detected": {"context_loss_detected", "context loss detected"},
    "no_context_loss": {"no_context_loss", "no context loss"},
    "premature_termination_detected": {
        "premature_termination_detected",
        "premature termination detected",
    },
    "no_premature_termination": {"no_premature_termination", "no premature termination"},
    "exact_repetition": {"exact_repetition", "exact repetition"},
    "stuck_on_failure": {"stuck_on_failure", "stuck on failure"},
    "reasoning_loop": {"reasoning_loop", "reasoning loop"},
    "degraded_success": {"degraded_success", "degraded success"},
    "no_infinite_loop": {"no_infinite_loop", "no infinite loop"},
    "token_exhaustion": {"token_exhaustion", "token_exhaustion_detected"},
    "insufficient_evidence": {"insufficient_evidence", "insufficient evidence"},
    "no_failure": {
        "no_failure",
        "no_tool_use_failure",
        "no tool use failure",
        "no_goal_failure",
        "no_hallucination",
        "no_context_loss",
        "no_premature_termination",
        "no_infinite_loop",
        "no_token_exhaustion",
    },
}

FAILURE_TYPE_ALIASES: dict[str, str] = {
    "tool use failure": "tool_use_failure",
    "goal satisfaction failure": "goal_satisfaction_failure",
    "hallucination": "hallucination",
    "context loss": "context_loss",
    "token exhaustion": "token_exhaustion",
    "premature termination": "premature_termination",
    "infinite loop": "infinite_loop",
    "none": "none",
    "insufficient_evidence": "insufficient_evidence",
}


def normalize(value: str | None) -> str:
    if not value:
        return ""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_failure_type(value: str | None) -> str:
    raw = normalize(value)
    return FAILURE_TYPE_ALIASES.get(raw.replace("_", " "), raw)


def subtype_matches(expected: str | None, actual: str | None) -> bool:
    exp = normalize(expected)
    act = normalize(actual)
    if not exp:
        return True
    if exp == act:
        return True
    for canonical, aliases in SUBTYPE_ALIASES.items():
        exp_norm = {normalize(a) for a in aliases}
        act_norm = {normalize(a) for a in aliases}
        if exp in exp_norm and act in act_norm:
            return True
    if exp == "no_failure":
        return act in NO_FAILURE_SUBTYPE_VALUES or act == ClassifierSubtype.NO_FAILURE.value
    return False


def case_passes(expected: dict, actual_ft: str | None, actual_st: str | None) -> tuple[bool, bool, bool]:
    exp_ft = expected.get("failure_type")
    exp_st = expected.get("subtype")
    ft_ok = failure_type_matches(exp_ft, actual_ft)
    st_ok = subtype_matches(exp_st, actual_st)

    if ft_ok and st_ok:
        return True, ft_ok, st_ok

    # Detectors return their own failure_type even on a clean pass (e.g. context_loss/no_context_loss).
    exp_ft_norm = normalize_failure_type(exp_ft)
    if st_ok and exp_ft_norm in {"none", "insufficient_evidence"}:
        if actual_st in NO_FAILURE_SUBTYPE_VALUES or actual_st == ClassifierSubtype.NO_FAILURE.value:
            return True, True, st_ok
        if exp_ft_norm == "insufficient_evidence" and actual_st == INSUFFICIENT_EVIDENCE_SUBTYPE:
            return True, True, st_ok

    return ft_ok and st_ok, ft_ok, st_ok


def failure_type_matches(expected: str | None, actual: str | None) -> bool:
    exp = normalize_failure_type(expected)
    act = normalize_failure_type(actual)
    if exp == act:
        return True
    if exp == "none" and act == "none":
        return True
    # Fixture uses failure_type=insufficient_evidence for ambiguous cases.
    if exp == "insufficient_evidence":
        return act in {
            "context_loss",
            "hallucination",
            "goal_satisfaction_failure",
            "premature_termination",
            "infinite_loop",
            "tool_use_failure",
            "none",
        }
    return False


def load_fixture_with_expected(path: Path) -> tuple[object, dict | None]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    expected = data.get("expected_diagnosis")
    trace = load_fixture(str(path))
    return trace, expected


def resolve_enabled_detectors(path: Path) -> list[FailureType] | None:
    rel = path.relative_to(FIXTURE_ROOT)
    parts = rel.parts

    if "classifier_e2e" in parts:
        return None  # all detectors

    if "infinite_loop_and_e2e_traces" in parts and "infinite_loop" in parts:
        return [FailureType.INFINITE_LOOP]

    top = parts[0]
    if top in FOLDER_DETECTORS:
        return FOLDER_DETECTORS[top]

    return None


def is_trace_fixture(path: Path) -> bool:
    if path.name in TOOL_USE_SIDEcars:
        return False
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return isinstance(data, dict) and "expected_diagnosis" in data and "steps" in data


def category_label(path: Path) -> str:
    rel = path.relative_to(FIXTURE_ROOT)
    if "classifier_e2e" in rel.parts:
        return "classifier_e2e"
    if "infinite_loop" in rel.parts:
        return "infinite_loop"
    return rel.parts[0]


def diagnose_trace(classifier: Classifier, trace: object, enabled: list[FailureType] | None) -> object:
    """Use classifier aggregation for e2e; direct detector output for single-detector suites."""
    if enabled is None:
        return classifier.diagnose(trace)
    if len(classifier.detectors) != 1:
        return classifier.diagnose(trace)
    return classifier.detectors[0].detect(trace)


def run_evaluation(output_dir: Path) -> dict:
    fixtures = sorted(p for p in FIXTURE_ROOT.rglob("*.json") if is_trace_fixture(p))

    llm_config = config_from_env()
    embedding_matcher = EmbeddingMatcher()
    llm_judge = create_llm_judge_from_env()

    # Reuse one classifier per detector preset to avoid reloading embeddings.
    classifiers: dict[str, Classifier] = {}
    results: list[dict] = []
    start_all = time.perf_counter()

    for index, path in enumerate(fixtures, start=1):
        enabled = resolve_enabled_detectors(path)
        preset_key = "all" if enabled is None else ",".join(d.value for d in enabled)
        if preset_key not in classifiers:
            classifiers[preset_key] = Classifier(
                llm_judge=llm_judge,
                enabled_detectors=enabled,
                embedding_matcher=embedding_matcher,
            )

        classifier = classifiers[preset_key]
        trace, expected = load_fixture_with_expected(path)
        expected = expected or {}

        t0 = time.perf_counter()
        try:
            result = diagnose_trace(classifier, trace, enabled)
            error = None
        except Exception as exc:
            result = None
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        exp_ft = expected.get("failure_type")
        exp_st = expected.get("subtype")
        act_ft = result.failure_type.value if result else None
        act_st = result.subtype if result else None

        if result:
            passed, ft_ok, st_ok = case_passes(expected, act_ft, act_st)
        else:
            passed, ft_ok, st_ok = False, False, False

        entry = {
            "file": str(path.relative_to(_ROOT)).replace("\\", "/"),
            "category": category_label(path),
            "run_id": trace.run_id,
            "expected": {
                "failure_type": exp_ft,
                "subtype": exp_st,
                "detection_stage": expected.get("detection_stage"),
            },
            "actual": {
                "failure_type": act_ft,
                "subtype": act_st,
                "confidence_score": round(result.confidence_score, 4) if result else None,
                "confidence_band": result.confidence_band.value if result else None,
                "detection_stage": result.detection_stage if result else None,
                "reason": (result.reason[:200] if result and result.reason else None),
            },
            "pass": passed,
            "failure_type_match": ft_ok,
            "subtype_match": st_ok,
            "elapsed_ms": elapsed_ms,
            "error": error,
        }
        results.append(entry)
        status = "PASS" if passed else "FAIL"
        print(f"[{index}/{len(fixtures)}] {status} {entry['file']}")

    total_elapsed_sec = round(time.perf_counter() - start_all, 1)
    passed = sum(1 for r in results if r["pass"])
    failed = len(results) - passed
    accuracy = round(passed / len(results) * 100, 1) if results else 0.0

    by_category: dict[str, dict] = {}
    for row in results:
        cat = row["category"]
        bucket = by_category.setdefault(cat, {"total": 0, "passed": 0, "failed": 0})
        bucket["total"] += 1
        if row["pass"]:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1
    for bucket in by_category.values():
        bucket["accuracy_pct"] = round(bucket["passed"] / bucket["total"] * 100, 1)

    failures = [r for r in results if not r["pass"]]

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixture_count": len(results),
        "passed": passed,
        "failed": failed,
        "accuracy_pct": accuracy,
        "total_elapsed_sec": total_elapsed_sec,
        "avg_elapsed_ms": round(sum(r["elapsed_ms"] for r in results) / len(results), 1) if results else 0,
        "llm_provider": llm_config.provider,
        "llm_model": llm_config.resolved_model(),
        "by_category": by_category,
        "failures": failures,
        "results": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fixture_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_summary_markdown(summary, output_dir / "SUMMARY.md")
    print()
    print(f"Done: {passed}/{len(results)} passed ({accuracy}%) in {total_elapsed_sec}s")
    print(f"Results: {json_path}")
    return summary


def write_summary_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# Fixture Evaluation Summary",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Overall",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Fixtures | {summary['fixture_count']} |",
        f"| Passed | {summary['passed']} |",
        f"| Failed | {summary['failed']} |",
        f"| **Accuracy** | **{summary['accuracy_pct']}%** |",
        f"| Total time | {summary['total_elapsed_sec']}s |",
        f"| Avg per fixture | {summary['avg_elapsed_ms']}ms |",
        f"| LLM provider | {summary['llm_provider']} |",
        f"| LLM model | {summary['llm_model']} |",
        "",
        "## By category",
        "",
        "| Category | Total | Passed | Failed | Accuracy |",
        "|----------|-------|--------|--------|----------|",
    ]
    for cat, bucket in sorted(summary["by_category"].items()):
        lines.append(
            f"| {cat} | {bucket['total']} | {bucket['passed']} | {bucket['failed']} | {bucket['accuracy_pct']}% |"
        )

    lines.extend(["", "## Failures", ""])
    if not summary["failures"]:
        lines.append("No failures — all fixtures passed.")
    else:
        for row in summary["failures"]:
            lines.append(f"- `{row['file']}`")
            lines.append(
                f"  - expected: {row['expected']['failure_type']} / {row['expected']['subtype']}"
            )
            lines.append(
                f"  - actual: {row['actual']['failure_type']} / {row['actual']['subtype']}"
            )
            if row.get("error"):
                lines.append(f"  - error: {row['error']}")

    lines.extend([
        "",
        "## README-ready copy",
        "",
        f"> Validated on {summary['fixture_count']} local JSON trace fixtures with "
        f"{summary['llm_provider']} ({summary['llm_model']}): "
        f"**{summary['passed']}/{summary['fixture_count']} passed ({summary['accuracy_pct']}% accuracy)**.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate all local JSON fixtures")
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "docs" / "evaluation",
        help="Output directory for JSON + markdown summary",
    )
    args = parser.parse_args()

    if not FIXTURE_ROOT.exists():
        print(f"Fixture directory not found: {FIXTURE_ROOT}", file=sys.stderr)
        return 1

    run_evaluation(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
