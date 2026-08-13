# API Reference

## Core Classes

### Classifier

The main entry point for agent failure diagnosis.

```python
class Classifier:
    def __init__(
        self, 
        llm_judge: LLMJudge | None = None,
        enabled_detectors: Optional[List[FailureType]] = None,
        embedding_matcher: EmbeddingMatcher | None = None,
    ):
        """Initialize classifier with optional configuration.
        
        Args:
            llm_judge: LLM implementation for complex reasoning. Defaults to MockLLMJudge.
            enabled_detectors: List of detector types to run. Defaults to all detectors.
            embedding_matcher: Shared EmbeddingMatcher for embedding-based detectors.
        """
    
    def diagnose(self, trace: AgentTrace) -> DetectionResult:
        """Run diagnosis on an agent execution trace.
        
        Args:
            trace: AgentTrace object containing execution data
            
        Returns:
            DetectionResult with failure analysis and confidence
        """
    
    def get_all_failures(self) -> List[DetectionResult]:
        """Get all detected failures, not just the primary one."""
    
    def get_detector_status(self) -> Dict[str, List[str]]:
        """Get status of which detectors ran vs were skipped."""
```

### AgentTrace

Pydantic model representing an agent execution trace.

```python
class AgentTrace(BaseModel):
    # Tier 1 - Universal (always required)
    run_id: str
    task: str  
    status: str
    total_steps: int
    final_output: Optional[Any] = None
    steps: List[Step]
    
    # Tier 2 - Common (optional)
    timestamp: Optional[datetime] = None
    error_message: Optional[str] = None
    total_tokens: Optional[int] = None
    
    # Tier 3 - Advanced (optional)  
    available_tools: Optional[List[ToolSpec]] = None
    constraints: Optional[List[str]] = None
    
    # Tier 4 - Derived (computed by library)
    constraint_list: Optional[List[Dict]] = None
```

### Step

Individual step within an agent execution.

```python
class Step(BaseModel):
    # Required
    step_index: int
    tool_name: str
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Any] = None
    
    # Optional enhancements
    thought: Optional[str] = None
    timestamp: Optional[datetime] = None
    error_message: Optional[str] = None
    step_status: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    retry_count: Optional[int] = None
```

### DetectionResult

Result of failure detection analysis.

```python
class DetectionResult(BaseModel):
    failure_type: FailureType
    subtype: str
    confidence_score: float  # 0.0 to 1.0
    confidence_band: ConfidenceBand
    evidence: List[Evidence]
    reason: str
    fix_direction: Optional[str] = None
    detection_stage: str
    secondary_evidence: Optional['DetectionResult'] = None
```

### Evidence

Supporting evidence for a detection result.

```python
class Evidence(BaseModel):
    detection_stage: str
    signal: str
    confidence_contribution: float
    explanation: str
```

## Enums

### FailureType

```python
class FailureType(str, Enum):
    TOOL_USE_FAILURE = "tool_use_failure"
    GOAL_SATISFACTION_FAILURE = "goal_satisfaction_failure" 
    HALLUCINATION = "hallucination"
    CONTEXT_LOSS = "context_loss"
    TOKEN_EXHAUSTION = "token_exhaustion"
    INFINITE_LOOP = "infinite_loop"
    PREMATURE_TERMINATION = "premature_termination"
    NONE = "none"
```

### ConfidenceBand

Thresholds are defined in `agent_diagnostician/config.py` (`CONFIDENCE_THRESHOLDS`)
and applied by `BaseDetector.confidence_to_band()`:

```python
class ConfidenceBand(str, Enum):
    CONFIRMED = "confirmed"      # score >= 0.85
    LIKELY = "likely"            # score >= 0.60
    MAYBE = "maybe"              # score >= 0.30
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # score < 0.30
```

## LLM Judge Interface

### create_llm_judge / create_llm_judge_from_env

Provider-agnostic factory. Set `LLM_PROVIDER` and `LLM_API_KEY` (or a
provider-specific key such as `GEMINI_API_KEY`, `OPENAI_API_KEY`, or
`ANTHROPIC_API_KEY`).

```python
from agent_diagnostician.analysis.llm import create_llm_judge_from_env

judge = create_llm_judge_from_env()
```

Supported providers: `gemini`, `openai`, `anthropic`, `mock`.

Optional extras: `pip install agent-diagnostician[llm-openai]`,
`pip install agent-diagnostician[llm-anthropic]`, or
`pip install agent-diagnostician[llm-all]`.

Use `python scripts/configure_llm.py --show` to inspect environment settings,
or `--provider gemini --api-key YOUR_KEY` to print setup commands.

### GeminiLLMJudge

Deprecated convenience wrapper — prefer `create_llm_judge()`.

```python
class GeminiLLMJudge:
    def __init__(self, model_name: str | None = None, api_key: str | None = None):
        """Initialize Gemini LLM judge."""
    
    def evaluate_wrong_tool(self, **kwargs) -> Dict[str, Any]:
        """Evaluate if wrong tool was selected."""
    
    def evaluate_parameter_structure(self, **kwargs) -> Dict[str, Any]:
        """Evaluate tool parameter structure validity."""
    
    def evaluate_parameter_values(self, **kwargs) -> Dict[str, Any]:
        """Evaluate tool parameter value correctness."""
    
    def evaluate_goal_alignment(self, **kwargs) -> Dict[str, Any]:
        """Evaluate task completion and goal alignment."""
    
    def evaluate_hallucination(self, **kwargs) -> Dict[str, Any]:
        """Evaluate presence of hallucinated content."""

    def evaluate_context_loss(self, **kwargs) -> Dict[str, Any]:
        """Evaluate whether the agent lost established context at a step."""

    def evaluate_premature_termination(self, **kwargs) -> Dict[str, Any]:
        """Evaluate whether the agent terminated before completing the task."""
```

All `evaluate_*` methods return a dict with `status: ok | error | parse_failed`.
Detectors ignore non-`ok` responses and fall back to rules/embeddings only.

### MockLLMJudge

Testing implementation used when no real LLM judge is configured.

```python
class MockLLMJudge:
    """Mock implementation for development without an API key."""
    
    def evaluate_wrong_tool(self, **kwargs) -> Dict[str, Any]:
        return {"verdict": "uncertain", "confidence": 0.0, "reason": "mock judge"}
    
    # All other evaluate_* methods return uncertain / zero confidence similarly.
```

`Classifier()` defaults to `MockLLMJudge`. LLM fallback stages then defer to
rules and embeddings only.

## Analysis Components

### EmbeddingMatcher

Semantic similarity analysis using sentence transformers.

```python
class EmbeddingMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with specified embedding model."""
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts.
        
        Returns:
            Float between 0.0 and 1.0 (higher = more similar)
        """
    
    def batch_similarity(
        self, 
        query: str, 
        candidates: List[str]
    ) -> List[float]:
        """Compute similarities between query and multiple candidates."""
```

### GroundingAnalyzer

Analyzes whether values are grounded in available context.

```python
class GroundingAnalyzer:
    @staticmethod
    def analyze(
        tool_input: Dict[str, Any],
        task: str, 
        prior_outputs: List[Any]
    ) -> Dict[str, Any]:
        """Analyze grounding of tool input values.
        
        Returns:
            Dictionary with grounding analysis results
        """
    
    @staticmethod
    def summarize(grounding_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize grounding analysis into aggregate metrics."""
```

### ConstraintExtractor

Extracts and validates task constraints.

```python
class ConstraintExtractor:
    @staticmethod
    def extract(task: str) -> List[Dict[str, Any]]:
        """Extract constraints from task description.
        
        Returns:
            List of constraint dictionaries with type, value, etc.
        """
    
    @staticmethod  
    def validate_constraint(
        constraint: Dict[str, Any], 
        actual_value: Any
    ) -> Dict[str, Any]:
        """Validate a constraint against actual output."""
```

## Detector Interfaces

### BaseDetector

Abstract base class for all failure detectors.

```python
class BaseDetector(ABC):
    @abstractmethod
    def detect(self, trace: AgentTrace) -> DetectionResult:
        """Run detection logic on trace.
        
        Args:
            trace: AgentTrace to analyze
            
        Returns:
            DetectionResult with analysis
        """
    
    def build_result(
        self,
        failure_type: FailureType,
        subtype: str, 
        confidence_score: float,
        evidence: List[Evidence],
        reason: str,
        detection_stage: str,
        fix_direction: Optional[str] = None,
        secondary_evidence: Optional[DetectionResult] = None
    ) -> DetectionResult:
        """Helper to build standardized DetectionResult."""
```

## Usage Examples

### Basic Diagnosis

```python
from agent_diagnostician import Classifier
from agent_diagnostician.models.trace import AgentTrace

# Create trace
trace = AgentTrace(**trace_data)

# Run diagnosis  
classifier = Classifier()
result = classifier.diagnose(trace)

print(f"Failure: {result.failure_type.value}")
print(f"Confidence: {result.confidence_band.value}")
print(f"Fix: {result.fix_direction}")
```

### Selective Detection

```python
from agent_diagnostician.models.enums import FailureType

# Only check specific failure types
classifier = Classifier(enabled_detectors=[
    FailureType.TOOL_USE_FAILURE,
    FailureType.GOAL_SATISFACTION_FAILURE
])

result = classifier.diagnose(trace)
```

### Custom LLM Configuration

```python
import os
from agent_diagnostician.analysis.llm import create_llm_judge_from_env

os.environ["LLM_PROVIDER"] = "gemini"  # or openai, anthropic
os.environ["LLM_API_KEY"] = "your-key"

judge = create_llm_judge_from_env()
classifier = Classifier(llm_judge=judge)
result = classifier.diagnose(trace)
```

### Batch Processing

```python
def diagnose_batch(traces: List[AgentTrace]) -> List[DetectionResult]:
    classifier = Classifier()
    results = []
    
    for trace in traces:
        try:
            result = classifier.diagnose(trace)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to diagnose {trace.run_id}: {e}")
            results.append(None)
    
    return results
```

### Error Handling

```python
from agent_diagnostician.models.enums import FailureType

try:
    result = classifier.diagnose(trace)
    
    if result.failure_type != FailureType.NONE:
        logger.warning(f"Detected failure: {result.reason}")
        
    if result.confidence_score < 0.5:
        logger.info("Low confidence - manual review recommended")
        
except ValueError as e:
    logger.error(f"Invalid trace data: {e}")
except Exception as e:
    logger.error(f"Diagnosis failed: {e}")
```