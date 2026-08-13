# Token exhaustion failure detector
# detectors/execution/token_exhaustion.py
# Detects whether the agent hit a token / context-window limit.
#
# Pipeline — two parts, stop-on-first-hit if Part A fires confidently:
#   Part A — Explicit error signal  (error_message fuzzy match, confidence 0.90)
#   Part B — Ratio / heuristic      (total_tokens vs known limit, or floor check)
#
# Fuzzy matching reuses utils.text.fuzzy_match — no local SequenceMatcher copy.
# Subtypes use TokenExhaustionSubtype enum, matching the pattern of every other
# detector in this project.

from agent_diagnostician.detectors.base import BaseDetector
from agent_diagnostician.models.trace import AgentTrace
from agent_diagnostician.models.result import DetectionResult, Evidence
from agent_diagnostician.models.enums import FailureType, TokenExhaustionSubtype, TRACE_FAILURE_STATUSES
from agent_diagnostician.utils.text import fuzzy_match

# ── Model token-limit lookup ──────────────────────────────────────────────────
# Maps lowercase model-name substrings to their published context window (tokens).
# Extend freely — just add a new entry. The resolver does a substring search so
# "gpt-4o" matches "gpt-4o-mini" descriptions too.
MODEL_TOKEN_LIMITS: dict[str, int] = {
    "gpt-4o":            128_000,
    "gpt-4-turbo":       128_000,
    "gpt-4":               8_192,  # Original GPT-4
    "gpt-4-32k":          32_768,  # GPT-4 with larger context
    "gpt-3.5-turbo":      16_385,
    "claude-3-opus":     200_000,
    "claude-3-sonnet":   200_000,
    "claude-3-haiku":    200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-sonnet-4":   200_000,
    "gemini-pro":        128_000,  # Gemini 1.0 Pro
    "gemini-1.0":        128_000,  # Gemini 1.0 family
    "gemini-2.0":      1_000_000,
    "gemini-1.5":      1_000_000,
}

# Reference phrases Part A matches against
_TOKEN_EXHAUSTION_PHRASES = [
    "token limit exceeded",
    "context length exceeded",
    "maximum context",
    "insufficient tokens",
    "token exhausted",
    "context window exceeded",
]

# Similarity threshold (mirrors GroundingAnalyzer.FUZZY_THRESHOLD)
_FUZZY_THRESHOLD = 0.8

# Status values that indicate the run did not complete successfully
_FAILURE_STATUSES = TRACE_FAILURE_STATUSES


class TokenExhaustionDetector(BaseDetector):
    """Detects token / context-window exhaustion in agent traces.

    Part A runs first and returns immediately on a confident match.
    Part B only executes when Part A finds nothing.
    """

    def detect(self, trace: AgentTrace) -> DetectionResult:
        # ── Part A ────────────────────────────────────────────────────────
        result = self._part_a_explicit_error(trace)
        if result is not None:
            return result

        # ── Part B ────────────────────────────────────────────────────────
        return self._part_b_ratio_heuristic(trace)

    # ── Part A — Explicit error signal ────────────────────────────────────────

    def _part_a_explicit_error(self, trace: AgentTrace) -> DetectionResult | None:
        """Check run-level and last-step error_message for token exhaustion text.

        Returns a DetectionResult (confidence 0.90) if any phrase matches,
        None if no match so Part B can run.
        """
        candidates: list[tuple[str, str]] = []  # (message, source_label)

        # Run-level error_message (Tier 2, Optional — added to AgentTrace)
        if trace.error_message:
            candidates.append((trace.error_message, "run-level error_message"))

        # Last step's error_message (Tier 2 on Step)
        if trace.steps:
            last = trace.steps[-1]
            if last.error_message:
                candidates.append((last.error_message, f"step {last.step_index} error_message"))

        for msg, source in candidates:
            for phrase in _TOKEN_EXHAUSTION_PHRASES:
                score = fuzzy_match(phrase, msg)
                if score >= _FUZZY_THRESHOLD:
                    return self.build_result(
                        failure_type=FailureType.TOKEN_EXHAUSTION,
                        subtype=TokenExhaustionSubtype.TOKEN_EXHAUSTION.value,
                        confidence_score=0.90,
                        evidence=[
                            Evidence(
                                detection_stage="Part A - Explicit Error Signal",
                                signal="error_message_match",
                                confidence_contribution=0.90,
                                explanation=(
                                    f"Matched phrase '{phrase}' in {source} "
                                    f"(similarity={score:.2f}): \"{msg[:200]}\""
                                ),
                            )
                        ],
                        reason=f"Explicit token exhaustion error in {source}: \"{msg[:200]}\"",
                        detection_stage="Part A",
                        fix_direction=(
                            "Reduce prompt size or split task into smaller subtasks; "
                            "alternatively use a model with a larger context window"
                        ),
                    )

        return None  # fall through to Part B

    # ── Part B — Ratio / heuristic signal ─────────────────────────────────────

    def _part_b_ratio_heuristic(self, trace: AgentTrace) -> DetectionResult:
        """Evaluate token usage against a known or inferred limit.

        Returns one of TOKEN_EXHAUSTION / NO_TOKEN_EXHAUSTION / INSUFFICIENT_EVIDENCE.
        """
        # 1. No token data at all → can't say anything meaningful
        if trace.total_tokens is None:
            return self.build_result(
                failure_type=FailureType.TOKEN_EXHAUSTION,
                subtype=TokenExhaustionSubtype.INSUFFICIENT_EVIDENCE.value,
                confidence_score=0.0,
                evidence=[
                    Evidence(
                        detection_stage="Part B - Ratio/Heuristic",
                        signal="no_token_data",
                        confidence_contribution=0.0,
                        explanation="trace.total_tokens is not present — cannot evaluate token exhaustion",
                    )
                ],
                reason="Token usage data unavailable; cannot evaluate token exhaustion",
                detection_stage="Part B",
                fix_direction=(
                    "Instrument the agent framework to populate total_tokens on AgentTrace"
                ),
            )

        # 2. Try to resolve a known model limit from available_tools
        limit, model_name = self._resolve_token_limit(trace)

        # 3a. Known model → ratio check
        if limit is not None:
            ratio = trace.total_tokens / limit

            if ratio >= 0.90:
                return self.build_result(
                    failure_type=FailureType.TOKEN_EXHAUSTION,
                    subtype=TokenExhaustionSubtype.TOKEN_EXHAUSTION.value,
                    confidence_score=0.80,
                    evidence=[
                        Evidence(
                            detection_stage="Part B - Ratio Check",
                            signal="high_token_ratio",
                            confidence_contribution=0.80,
                            explanation=(
                                f"Token usage {trace.total_tokens:,} is {ratio*100:.1f}% of "
                                f"{model_name} limit ({limit:,})"
                            ),
                        )
                    ],
                    reason=(
                        f"Token usage ({trace.total_tokens:,}) reached "
                        f"{ratio*100:.1f}% of the {model_name} context window ({limit:,})"
                    ),
                    detection_stage="Part B",
                    fix_direction=(
                        "Reduce prompt size or split task into smaller subtasks; "
                        "alternatively use a model with a larger context window"
                    ),
                )

            if ratio >= 0.75:
                return self.build_result(
                    failure_type=FailureType.TOKEN_EXHAUSTION,
                    subtype=TokenExhaustionSubtype.TOKEN_EXHAUSTION.value,
                    confidence_score=0.55,
                    evidence=[
                        Evidence(
                            detection_stage="Part B - Ratio Check",
                            signal="moderate_token_ratio",
                            confidence_contribution=0.55,
                            explanation=(
                                f"Token usage {trace.total_tokens:,} is {ratio*100:.1f}% of "
                                f"{model_name} limit ({limit:,}) — approaching limit"
                            ),
                        )
                    ],
                    reason=(
                        f"Token usage ({trace.total_tokens:,}) is {ratio*100:.1f}% of "
                        f"the {model_name} context window ({limit:,})"
                    ),
                    detection_stage="Part B",
                    fix_direction=(
                        "Monitor token usage; consider reducing prompt size or splitting task "
                        "before the limit is reached"
                    ),
                )

            # Ratio low — clearly not exhausted
            return self.build_result(
                failure_type=FailureType.TOKEN_EXHAUSTION,
                subtype=TokenExhaustionSubtype.NO_TOKEN_EXHAUSTION.value,
                confidence_score=0.85,
                evidence=[],
                reason=(
                    f"Token usage ({trace.total_tokens:,}) is only {ratio*100:.1f}% of "
                    f"the {model_name} limit ({limit:,})"
                ),
                detection_stage="Part B",
                fix_direction="No fix required — token usage is well within the model limit",
            )

        # 3b. Model unknown — floor heuristic only
        # NOTE: confidence is hard-capped at 0.35 here.
        # total_tokens is scoped to *this run only*. It cannot see whether a prior
        # task in the same session already consumed most of the shared budget.
        # A high single-run token count is therefore a weak proxy for exhaustion,
        # not a real remaining-budget check. We intentionally keep this low so the
        # signal does not fire as LIKELY or CONFIRMED without a known limit.
        if trace.total_tokens > 100_000 and trace.status in _FAILURE_STATUSES:
            return self.build_result(
                failure_type=FailureType.TOKEN_EXHAUSTION,
                subtype=TokenExhaustionSubtype.TOKEN_EXHAUSTION.value,
                confidence_score=0.35,
                evidence=[
                    Evidence(
                        detection_stage="Part B - Floor Heuristic",
                        signal="high_token_floor_with_failure_status",
                        confidence_contribution=0.35,
                        explanation=(
                            f"Token count ({trace.total_tokens:,}) exceeds 100k floor and "
                            f"run status is '{trace.status}' — model limit unknown, weak signal"
                        ),
                    )
                ],
                reason=(
                    f"High token count ({trace.total_tokens:,}) combined with failure "
                    f"status '{trace.status}', but model is unknown so confidence is capped low"
                ),
                detection_stage="Part B",
                fix_direction=(
                    "Identify the model being used and compare against its context limit; "
                    "consider reducing prompt size or splitting task"
                ),
            )

        # No signal
        return self.build_result(
            failure_type=FailureType.TOKEN_EXHAUSTION,
            subtype=TokenExhaustionSubtype.NO_TOKEN_EXHAUSTION.value,
            confidence_score=0.70,
            evidence=[],
            reason="No token exhaustion signals detected",
            detection_stage="Part B",
            fix_direction="No fix required",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_token_limit(
        self, trace: AgentTrace
    ) -> tuple[int | None, str | None]:
        """Search available_tools text for a recognised model identifier.

        Returns:
            (limit, model_name) if a match is found, (None, None) otherwise.
        """
        if not trace.available_tools:
            return None, None

        for tool in trace.available_tools:
            combined = f"{tool.name} {tool.description or ''}".lower()
            for model_key, limit in MODEL_TOKEN_LIMITS.items():
                if model_key in combined:
                    return limit, model_key

        return None, None
