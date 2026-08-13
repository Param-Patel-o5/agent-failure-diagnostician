#!/usr/bin/env python3
"""Phased fixture evaluation with API rate limiting.

1. Run fixtures that never call the LLM (infinite loop, token-only e2e).
2. Queue fixtures that always need a live LLM (hallucination, classifier e2e, llm_fallback, …).
3. Probe the rest with MockLLMJudge — only skip API when the detector resolves without LLM.
4. Run the LLM queue in rate-limited batches (default: 3, then 15, 15, ...).

Usage:
    python scripts/run_phased_fixture_evaluation.py
    python scripts/run_phased_fixture_evaluation.py --initial-used 11 --first-batch 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_diagnostician.analysis.embeddings import EmbeddingMatcher
from agent_diagnostician.analysis.llm import LLMJudge, MockLLMJudge, create_llm_judge_from_env
from agent_diagnostician.analysis.llm.config import config_from_env
from agent_diagnostician.classifier import Classifier
from agent_diagnostician.models.enums import FailureType

# Reuse evaluation helpers from the full runner.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from run_fixture_evaluation import (
    FIXTURE_ROOT,
    category_label,
    case_passes,
    diagnose_trace,
    is_trace_fixture,
    load_fixture_with_expected,
    resolve_enabled_detectors,
    write_summary_markdown,
)


def never_calls_llm(path: Path) -> bool:
    """Detectors that do not invoke the LLM judge during detect()."""
    rel = path.relative_to(FIXTURE_ROOT)
    parts = rel.parts
    if "infinite_loop" in parts and "classifier_e2e" not in parts:
        return True
    if path.name == "e2e__selective_detectors__token_exhaustion_only.json":
        return True
    return False


def requires_live_llm(path: Path) -> bool:
    """Fixtures that must use a real LLM — never mock-resolve."""
    if never_calls_llm(path):
        return False
    rel = path.relative_to(FIXTURE_ROOT)
    parts_lower = [p.lower() for p in rel.parts]
    name_lower = path.name.lower()

    # Hallucination always calls evaluate_hallucination(); blend requires live LLM.
    if parts_lower[0] == "hallucination":
        return True
    if "classifier_e2e" in parts_lower:
        return True
    if "task misinterpretation" in parts_lower or "aggregator edge cases" in parts_lower:
        return True
    if any(tag in name_lower for tag in ("llm_fallback", "llm_primary", "llm_uncertain")):
        return True
    return False


class RateLimitedJudge(LLMJudge):
    """Forward to a real judge while enforcing requests-per-minute."""

    def __init__(
        self,
        inner: LLMJudge,
        max_per_minute: int = 15,
        initial_used: int = 0,
    ):
        self._inner = inner
        self.max_per_minute = max_per_minute
        self.calls_this_minute = initial_used
        self.total_calls = 0
        self._window_start = time.time()

    def _throttle(self) -> None:
        now = time.time()
        elapsed = now - self._window_start
        if elapsed >= 60:
            self.calls_this_minute = 0
            self._window_start = now
        if self.calls_this_minute >= self.max_per_minute:
            sleep_sec = max(1.0, 61.0 - (now - self._window_start))
            print(f"  [rate limit] {self.calls_this_minute}/{self.max_per_minute} used — sleeping {sleep_sec:.0f}s")
            time.sleep(sleep_sec)
            self.calls_this_minute = 0
            self._window_start = time.time()
        self.calls_this_minute += 1
        self.total_calls += 1

    def evaluate_wrong_tool(self, task, selected_tool, available_tools, thought=None):
        self._throttle()
        return self._inner.evaluate_wrong_tool(task, selected_tool, available_tools, thought)

    def evaluate_parameter_structure(self, tool_name, tool_input, task):
        self._throttle()
        return self._inner.evaluate_parameter_structure(tool_name, tool_input, task)

    def evaluate_parameter_values(self, task, tool_input, prior_outputs, thought=None):
        self._throttle()
        return self._inner.evaluate_parameter_values(task, tool_input, prior_outputs, thought)

    def evaluate_goal_alignment(self, task, final_output, steps, thought=None, embedding_score=None):
        self._throttle()
        return self._inner.evaluate_goal_alignment(
            task, final_output, steps, thought, embedding_score
        )

    def evaluate_hallucination(
        self, task, tool_input, prior_outputs, thought=None, available_tools=None
    ):
        self._throttle()
        return self._inner.evaluate_hallucination(
            task, tool_input, prior_outputs, thought, available_tools
        )

    def evaluate_context_loss(
        self,
        task,
        step_index,
        tool_name,
        tool_input,
        tool_output,
        prior_outputs,
        steps,
        thought=None,
    ):
        self._throttle()
        return self._inner.evaluate_context_loss(
            task, step_index, tool_name, tool_input, tool_output, prior_outputs, steps, thought
        )

    def evaluate_premature_termination(
        self, task, final_output, steps, thought=None, embedding_score=None
    ):
        self._throttle()
        return self._inner.evaluate_premature_termination(
            task, final_output, steps, thought, embedding_score
        )


def run_one_fixture(
    path: Path,
    classifier: Classifier,
    enabled: list[FailureType] | None,
) -> dict:
    trace, expected = load_fixture_with_expected(path)
    expected = expected or {}
    t0 = time.perf_counter()
    error = None
    result = None
    try:
        result = diagnose_trace(classifier, trace, enabled)
    except Exception as exc:
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

    return {
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


def build_classifier(
    llm_judge: LLMJudge,
    enabled: list[FailureType] | None,
    embedding_matcher: EmbeddingMatcher,
) -> Classifier:
    return Classifier(
        llm_judge=llm_judge,
        enabled_detectors=enabled,
        embedding_matcher=embedding_matcher,
    )


def make_summary(results: list[dict], llm_config, total_elapsed_sec: float) -> dict:
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

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixture_count": len(results),
        "passed": passed,
        "failed": failed,
        "accuracy_pct": accuracy,
        "total_elapsed_sec": round(total_elapsed_sec, 1),
        "avg_elapsed_ms": round(sum(r["elapsed_ms"] for r in results) / len(results), 1) if results else 0,
        "llm_provider": llm_config.provider,
        "llm_model": llm_config.resolved_model(),
        "by_category": by_category,
        "failures": failures,
        "results": results,
    }


def run_phased(
    output_dir: Path,
    initial_used: int,
    first_batch: int,
    batch_size: int,
    max_per_minute: int,
) -> dict:
    fixtures = sorted(p for p in FIXTURE_ROOT.rglob("*.json") if is_trace_fixture(p))
    llm_config = config_from_env()
    embedding_matcher = EmbeddingMatcher()
    mock_judge = MockLLMJudge()

    results: list[dict] = []
    llm_queue: list[Path] = []
    classifiers_mock: dict[str, Classifier] = {}

    start_all = time.perf_counter()

    print("=== Phase 0: No-LLM + mock probe ===")
    for path in fixtures:
        enabled = resolve_enabled_detectors(path)
        preset_key = "all" if enabled is None else ",".join(d.value for d in enabled)

        if never_calls_llm(path):
            if preset_key not in classifiers_mock:
                classifiers_mock[preset_key] = build_classifier(mock_judge, enabled, embedding_matcher)
            row = run_one_fixture(path, classifiers_mock[preset_key], enabled)
            row["run_mode"] = "no_llm"
            results.append(row)
            status = "PASS" if row["pass"] else "FAIL"
            print(f"  [no-llm] {status} {row['file']}")
            continue

        if requires_live_llm(path):
            llm_queue.append(path)
            print(f"  [live LLM required] queued -> {path.name}")
            continue

        if preset_key not in classifiers_mock:
            classifiers_mock[preset_key] = build_classifier(mock_judge, enabled, embedding_matcher)
        row = run_one_fixture(path, classifiers_mock[preset_key], enabled)

        if row["pass"]:
            row["run_mode"] = "mock_pass"
            results.append(row)
            print(f"  [mock OK] PASS {row['file']}")
        else:
            llm_queue.append(path)
            print(f"  [mock FAIL] queued for LLM -> {path.name}")

    print(f"\nMock phase done: {len(results)} resolved, {len(llm_queue)} need LLM API")

    if not llm_queue:
        print("No LLM queue — writing results.")
    else:
        real_judge = create_llm_judge_from_env()
        rate_judge = RateLimitedJudge(
            real_judge,
            max_per_minute=max_per_minute,
            initial_used=initial_used,
        )
        classifiers_llm: dict[str, Classifier] = {}

        batch_sizes = [first_batch]
        remaining = len(llm_queue) - first_batch
        while remaining > 0:
            batch_sizes.append(min(batch_size, remaining))
            remaining -= batch_size

        idx = 0
        batch_num = 0
        for size in batch_sizes:
            if idx >= len(llm_queue):
                break
            batch_num += 1
            batch = llm_queue[idx:idx + size]
            idx += size
            print(f"\n=== LLM batch {batch_num}: {len(batch)} fixture(s) ===")
            calls_before = rate_judge.total_calls

            for path in batch:
                enabled = resolve_enabled_detectors(path)
                preset_key = "all" if enabled is None else ",".join(d.value for d in enabled)
                if preset_key not in classifiers_llm:
                    classifiers_llm[preset_key] = build_classifier(
                        rate_judge, enabled, embedding_matcher
                    )
                row = run_one_fixture(path, classifiers_llm[preset_key], enabled)
                row["run_mode"] = "llm"
                row["llm_api_calls"] = rate_judge.total_calls
                results.append(row)
                status = "PASS" if row["pass"] else "FAIL"
                print(f"  [llm] {status} {row['file']} (API calls so far: {rate_judge.total_calls})")

            calls_in_batch = rate_judge.total_calls - calls_before
            print(f"  Batch {batch_num} used {calls_in_batch} API call(s)")

    total_elapsed = time.perf_counter() - start_all
    total_api_calls = rate_judge.total_calls if llm_queue else 0
    summary = make_summary(results, llm_config, total_elapsed)
    summary["phased"] = {
        "initial_used": initial_used,
        "first_batch": first_batch,
        "batch_size": batch_size,
        "max_per_minute": max_per_minute,
        "mock_resolved": sum(1 for r in results if r.get("run_mode") in ("mock_pass", "no_llm")),
        "llm_ran": sum(1 for r in results if r.get("run_mode") == "llm"),
        "total_api_calls": total_api_calls,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fixture_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_summary_markdown(summary, output_dir / "SUMMARY.md")

    passed = summary["passed"]
    total = summary["fixture_count"]
    print()
    print(f"Done: {passed}/{total} passed ({summary['accuracy_pct']}%) in {summary['total_elapsed_sec']}s")
    if llm_queue:
        print(f"Total API calls: {total_api_calls}")
    print(f"Results: {json_path}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Phased fixture evaluation with rate limiting")
    parser.add_argument("--output", type=Path, default=_ROOT / "docs" / "evaluation")
    parser.add_argument(
        "--initial-used",
        type=int,
        default=11,
        help="API calls already consumed in the current minute window",
    )
    parser.add_argument(
        "--first-batch",
        type=int,
        default=3,
        help="Number of LLM fixtures in the first batch",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=15,
        help="Number of LLM fixtures per subsequent batch",
    )
    parser.add_argument(
        "--max-per-minute",
        type=int,
        default=15,
        help="Max API requests per minute",
    )
    args = parser.parse_args()

    if not FIXTURE_ROOT.exists():
        print(f"Fixture directory not found: {FIXTURE_ROOT}", file=sys.stderr)
        return 1

    run_phased(
        args.output,
        initial_used=args.initial_used,
        first_batch=args.first_batch,
        batch_size=args.batch_size,
        max_per_minute=args.max_per_minute,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
