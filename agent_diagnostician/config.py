# Agent diagnostician configuration
# config.py
# Global configuration for Agent Failure Diagnostician.
# Change settings here without touching detector or analysis code.

# ── Embedding Model ────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # local, free, no API needed

# ── LLM Judge ─────────────────────────────────────────────────────────────
LLM_PROVIDER = "gemini"               # "gemini" | "openai" | "mock"
LLM_MODEL = "gemini-2.0-flash"

# ── Confidence Thresholds ──────────────────────────────────────────────────
# These map 0-1 scores to ConfidenceBand labels.
# Tune these after testing on real traces if bands feel off.
CONFIDENCE_THRESHOLDS = {
    "confirmed":             0.85,
    "likely":                0.60,
    "maybe":                 0.30,
    # below 0.30 = insufficient_evidence
}

# ── Detection Settings ─────────────────────────────────────────────────────
# Minimum similarity gap between rank-1 and rank-2 tool to flag wrong tool.
# If gap <= this, tools are considered equivalent and wrong tool is NOT flagged.
TOOL_RANKING_GAP_THRESHOLD = 0.15

# Minimum prior successful calls needed to infer a runtime schema.
MIN_PRIOR_CALLS_FOR_SCHEMA = 2

# Fuzzy match threshold for grounding analysis (0-1).
GROUNDING_FUZZY_THRESHOLD = 0.80

# Minimum confidence to report secondary evidence in GoalFailureDetector.
SECONDARY_EVIDENCE_THRESHOLD = 0.30

# ── Output Settings ────────────────────────────────────────────────────────
DEFAULT_OUTPUT_FORMAT = "cli"         # "cli" | "json" | "markdown"