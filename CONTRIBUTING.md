# Contributing to Agent Diagnostician

Thank you for your interest in contributing to Agent Diagnostician! This guide will help you get started with development and understand our contribution process.

## 🚀 Quick Start

### Development Setup

1. **Clone and Install**
```bash
git clone https://github.com/your-org/agent-diagnostician
cd agent-diagnostician
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

2. **Install Pre-commit Hooks**
```bash
pre-commit install
```

3. **Run Tests** 
```bash
pytest tests/ -v
python -m pytest tests/integration/ --slow  # Integration tests
```

4. **Verify Installation**
```bash
python -c "from agent_diagnostician import Classifier; print('Setup successful!')"
```

### Project Structure

```
agent-diagnostician/
├── agent_diagnostician/          # Main library code
│   ├── detectors/               # Failure detection logic
│   │   ├── planning/           # Tool use, goal satisfaction, hallucination
│   │   ├── execution/          # Context loss, token exhaustion
│   │   └── termination/        # Premature termination, infinite loops
│   ├── analysis/               # Reusable analysis components
│   ├── models/                 # Pydantic data models
│   └── prompts/               # LLM prompt templates
├── tests/                      # Test suites
├── docs/                      # Documentation
├── test cases/               # Test case fixtures (JSON traces)
└── examples/                 # Usage examples
```

## 🎯 Contribution Areas

### High Priority

**1. Detector Accuracy Improvements**
- Improve hallucination detection (currently 55.6% accuracy)
- Enhance premature termination threshold tuning
- Fix false positive issues in various detectors

**2. LLM Provider Support**
- Add OpenAI GPT integration
- Implement Anthropic Claude support  
- Add Ollama local model support

**3. Framework Integrations**
- Expand LangChain integration examples
- Add CrewAI support
- Implement Haystack compatibility

### Medium Priority

**4. Performance Optimization**
- Implement embedding caching
- Add async/batch processing capabilities
- Optimize memory usage for large traces

**5. Testing & Validation**  
- Expand test case coverage
- Add property-based testing
- Implement benchmark suite

### Lower Priority

**6. Documentation & Examples**
- Add more real-world examples
- Create tutorial notebooks
- Improve API documentation

**7. Tooling & Infrastructure**
- Add CI/CD improvements
- Implement automated benchmarking
- Create debugging utilities

## 📋 Development Process

### Issue Workflow

1. **Check Existing Issues** - Search for related issues before creating new ones
2. **Create Issue** - Use appropriate template (bug report, feature request, etc.)
3. **Discussion** - Engage with maintainers on approach before major changes
4. **Assignment** - Issues are assigned to prevent duplicate work

### Pull Request Process

1. **Fork & Branch**
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number-description
```

2. **Development**
   - Follow coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. **Commit Standards**
```bash
# Use conventional commits
git commit -m "feat: add OpenAI LLM judge implementation"
git commit -m "fix: resolve premature termination false positives"
git commit -m "docs: update integration examples"
```

4. **Pre-submission Checklist**
   - [ ] Tests pass locally (`pytest`)
   - [ ] Code follows style guidelines (`black`, `flake8`)
   - [ ] Documentation updated if needed
   - [ ] CHANGELOG.md updated for user-facing changes
   - [ ] No breaking changes (or clearly documented)

5. **Submit PR**
   - Use descriptive title and clear description
   - Link related issues (`Fixes #123`)
   - Request review from relevant maintainers

## 🏗️ Architecture Guidelines

### Core Principles

**1. Composition Over Inheritance**
- Favor dependency injection over deep inheritance hierarchies
- Use mixins and protocols for shared behavior
- Keep detector classes focused and single-purpose

**2. Evidence-Based Detection**
```python
# Good: Explicit evidence chain
evidence.append(Evidence(
    detection_stage="1A - Schema Validation",
    signal="missing_required_field",
    confidence_contribution=0.60,
    explanation="Required field 'user_id' not found in tool_input"
))

# Bad: Magic number without explanation
confidence += 0.60  # Why this value?
```

**3. Graceful Degradation**
```python
# Good: Handle missing optional data
if step.thought is not None:
    thought_similarity = self.embeddings.similarity(step.thought, task)
else:
    thought_similarity = None  # Skip thought-based analysis

# Bad: Crash on missing data
thought_similarity = self.embeddings.similarity(step.thought, task)  # KeyError!
```

**4. Deterministic Before Heuristic**
```python
# Good: Check rules first, then embeddings, then LLM
if self._check_schema_violation(step):
    return self._build_schema_violation_result()
elif self._check_embedding_mismatch(step) and confidence > threshold:
    return self._build_semantic_mismatch_result()
else:
    return self._llm_fallback_analysis(step)
```

### Detector Development Pattern

**1. Detector Structure**
```python
class NewFailureDetector(BaseDetector):
    """Detect [specific failure type] in agent execution traces.
    
    [Description of failure type and detection approach]
    """
    
    # Configuration constants
    SIMILARITY_THRESHOLD = 0.45
    MIN_CONFIDENCE = 0.30
    
    def __init__(self, llm_judge: LLMJudge | None = None):
        self.llm_judge = llm_judge or MockLLMJudge()
        # Initialize expensive components once
        self.embeddings = EmbeddingMatcher()
    
    def detect(self, trace: AgentTrace) -> DetectionResult:
        """Main detection pipeline."""
        # Early returns for edge cases
        if not trace.steps:
            return self._no_failure_result("No steps to analyze")
        
        evidence = []
        
        # Stage 1: Deterministic checks
        stage1_confidence = self._deterministic_analysis(trace, evidence)
        
        # Stage 2: Heuristic analysis (if needed)
        stage2_confidence = 0.0
        if stage1_confidence < self.MIN_CONFIDENCE:
            stage2_confidence = self._heuristic_analysis(trace, evidence)
        
        # Stage 3: LLM fallback (if both previous stages inconclusive)
        stage3_confidence = 0.0
        if stage1_confidence + stage2_confidence < self.MIN_CONFIDENCE:
            stage3_confidence = self._llm_analysis(trace, evidence)
        
        # Combine and decide
        total_confidence = stage1_confidence + stage2_confidence + stage3_confidence
        
        if total_confidence >= self.MIN_CONFIDENCE:
            return self._failure_result(total_confidence, evidence)
        else:
            return self._no_failure_result("Insufficient evidence for failure")
```

**2. Evidence Documentation**
```python
def _build_evidence(self, stage: str, signal: str, confidence: float, explanation: str) -> Evidence:
    """Helper to build consistent evidence objects."""
    return Evidence(
        detection_stage=stage,
        signal=signal,
        confidence_contribution=confidence,
        explanation=explanation
    )

# Usage
evidence.append(self._build_evidence(
    stage="1A - Schema Validation",
    signal="required_field_missing", 
    confidence=0.60,
    explanation=f"Tool '{step.tool_name}' missing required parameter '{field_name}'"
))
```

### Testing Standards

**1. Test Organization**
```
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_detectors/     # Individual detector tests
│   ├── test_analysis/      # Analysis component tests  
│   └── test_models/        # Data model tests
├── integration/            # End-to-end workflow tests
├── fixtures/               # Reusable test data
└── conftest.py            # Pytest configuration
```

**2. Test Case Structure**
```python
class TestToolUseDetector:
    """Test suite for ToolUseDetector."""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance for testing."""
        return ToolUseDetector(llm_judge=MockLLMJudge())
    
    @pytest.fixture  
    def valid_trace(self):
        """Create valid trace fixture."""
        return AgentTrace(
            run_id="test_001",
            task="Search for Python tutorials", 
            status="success",
            total_steps=1,
            steps=[Step(
                step_index=0,
                tool_name="web_search",
                tool_input={"query": "Python tutorials"},
                tool_output=["Tutorial 1", "Tutorial 2"]
            )]
        )
    
    def test_detect_no_failure_on_valid_trace(self, detector, valid_trace):
        """Should detect no failure on valid tool usage."""
        result = detector.detect(valid_trace)
        
        assert result.failure_type == FailureType.TOOL_USE_FAILURE
        assert result.subtype == ToolUseSubtype.NO_FAILURE.value
        assert result.confidence_score < 0.3  # Low confidence in failure
    
    def test_detect_wrong_tool_with_clear_mismatch(self, detector):
        """Should detect wrong tool when there's clear mismatch."""
        trace = AgentTrace(
            run_id="test_002", 
            task="Calculate 2+2",
            status="success",
            total_steps=1,
            steps=[Step(
                step_index=0,
                tool_name="web_search",  # Wrong tool for math
                tool_input={"query": "2+2"},
                tool_output="About 1,234,567 results"
            )]
        )
        
        result = detector.detect(trace)
        
        assert result.failure_type == FailureType.TOOL_USE_FAILURE  
        assert result.subtype == ToolUseSubtype.WRONG_TOOL_SELECTED.value
        assert result.confidence_score > 0.5
        assert len(result.evidence) > 0
        assert "wrong tool" in result.reason.lower()
    
    @pytest.mark.parametrize("task,tool_name,expected_failure", [
        ("Search Google", "web_search", False),
        ("Calculate tip", "web_search", True), 
        ("Send email", "calculator", True),
    ])
    def test_tool_task_alignment(self, detector, task, tool_name, expected_failure):
        """Test tool-task alignment across multiple scenarios."""
        # Implementation...
```

**3. Property-Based Testing**
```python
from hypothesis import given, strategies as st

@given(
    task=st.text(min_size=10, max_size=100),
    tool_name=st.sampled_from(["web_search", "calculator", "email", "file_read"]),
    num_steps=st.integers(min_value=1, max_value=5)
)
def test_detector_never_crashes(detector, task, tool_name, num_steps):
    """Detector should never crash regardless of input."""
    trace = generate_random_trace(task, tool_name, num_steps)
    
    result = detector.detect(trace)  # Should not raise
    
    # Basic sanity checks
    assert isinstance(result, DetectionResult)
    assert 0.0 <= result.confidence_score <= 1.0
    assert isinstance(result.evidence, list)
```

## 🎨 Code Style Guidelines

### Python Standards

**1. Use Black + isort**
```bash
black agent_diagnostician/ tests/
isort agent_diagnostician/ tests/
```

**2. Type Hints**
```python
# Good: Full type annotations
def analyze_grounding(
    tool_input: Dict[str, Any], 
    prior_outputs: List[Any]
) -> Dict[str, float]:
    """Analyze value grounding with full type safety."""
    pass

# Bad: Missing types
def analyze_grounding(tool_input, prior_outputs):
    pass
```

**3. Docstrings**
```python
class ToolUseDetector(BaseDetector):
    """Detects tool use failures in agent execution traces.
    
    Identifies three main failure modes:
    1. Wrong tool selected for the task
    2. Correct tool but invalid parameter structure  
    3. Correct tool and structure but incorrect parameter values
    
    Detection uses a multi-stage pipeline with deterministic checks,
    embedding analysis, and LLM fallback reasoning.
    """
    
    def detect(self, trace: AgentTrace) -> DetectionResult:
        """Run tool use failure detection on a trace.
        
        Args:
            trace: AgentTrace containing execution steps to analyze
            
        Returns:
            DetectionResult with failure classification and evidence
            
        Raises:
            ValueError: If trace contains invalid data
        """
```

### Configuration Management

**1. Environment Variables**
```python
# config.py
import os
from typing import List, Optional

class Config:
    """Centralized configuration management."""
    
    # LLM Configuration
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY") 
    DEFAULT_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "models/gemini-2.5-flash")
    
    # Detection Thresholds
    MIN_CONFIDENCE_THRESHOLD: float = float(os.getenv("MIN_CONFIDENCE", "0.3"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.45"))
    
    # Performance
    MAX_EMBEDDING_CACHE_SIZE: int = int(os.getenv("EMBEDDING_CACHE_SIZE", "10000"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
```

**2. Detector Configuration**
```python
# Allow runtime configuration without code changes
detector_config = {
    "tool_use": {
        "similarity_threshold": 0.45,
        "min_confidence": 0.30,
        "enable_llm_fallback": True
    },
    "goal_satisfaction": {
        "constraint_weight": 0.70,
        "misinterpretation_weight": 0.60,
        "enable_semantic_analysis": True
    }
}

classifier = Classifier(detector_config=detector_config)
```

## 🧪 Testing Guidelines

### Test Categories

**1. Unit Tests** (Fast, < 100ms each)
- Individual detector logic
- Analysis component functions
- Data model validation
- Mock LLM responses

**2. Integration Tests** (Medium, < 5s each) 
- End-to-end detection workflows
- Real LLM integration (with API keys)
- Framework integration examples
- Error handling scenarios

**3. Performance Tests** (Slow, benchmarking)
- Large trace processing
- Memory usage profiling  
- Concurrent processing
- Cache effectiveness

### Running Tests

```bash
# Fast unit tests only
pytest tests/unit/ -v

# All tests including integration  
pytest tests/ -v --slow

# Specific test categories
pytest -m "not integration" -v        # Skip integration tests
pytest -m "integration" -v --slow     # Only integration tests
pytest -k "test_tool_use" -v          # Specific test pattern

# With coverage
pytest --cov=agent_diagnostician tests/ --cov-report=html

# Performance profiling
pytest tests/performance/ --benchmark-only
```

### Mock Guidelines

**1. LLM Mocking**
```python
# Good: Deterministic, predictable responses
class MockLLMJudge:
    def evaluate_wrong_tool(self, task: str, tool_name: str, **kwargs) -> Dict:
        """Return deterministic mock response based on inputs."""
        if "calculate" in task.lower() and tool_name == "web_search":
            return {"verdict": "wrong", "confidence": 0.85, "reason": "Math task needs calculator"}
        return {"verdict": "correct", "confidence": 0.90, "reason": "Appropriate tool"}

# Bad: Random or non-deterministic responses  
class BadMockLLMJudge:
    def evaluate_wrong_tool(self, **kwargs) -> Dict:
        return {"verdict": random.choice(["wrong", "correct"]), "confidence": random.random()}
```

**2. Trace Fixtures**
```python
# tests/fixtures/traces.py
def create_tool_use_trace(
    task: str = "Default task",
    tool_name: str = "default_tool", 
    success: bool = True,
    include_thought: bool = False
) -> AgentTrace:
    """Factory for creating test traces with common patterns."""
    
    step_data = {
        "step_index": 0,
        "tool_name": tool_name,
        "tool_input": {"query": task},
        "tool_output": "Success" if success else {"error": "Failed"}
    }
    
    if include_thought:
        step_data["thought"] = f"I need to use {tool_name} for this task"
    
    return AgentTrace(
        run_id="test_trace",
        task=task,
        status="success" if success else "error",
        total_steps=1,
        steps=[Step(**step_data)]
    )
```

## 📊 Performance & Benchmarking

### Performance Targets

**Latency Goals:**
- Simple detection (no LLM): < 100ms
- With embedding analysis: < 500ms  
- With LLM fallback: < 5s
- Batch processing: > 10 traces/second

**Memory Goals:**
- < 100MB baseline memory
- < 50MB per additional trace in batch
- Embedding cache: configurable limit

**Accuracy Goals:**
- > 90% accuracy on clear failure cases
- > 70% accuracy on ambiguous cases
- < 5% false positive rate on success cases

### Benchmarking

```python
# tests/performance/test_benchmarks.py
import pytest
from agent_diagnostician import Classifier

class TestPerformanceBenchmarks:
    
    @pytest.mark.benchmark
    def test_single_trace_latency(self, benchmark, sample_trace):
        """Benchmark single trace processing time."""
        classifier = Classifier()
        
        result = benchmark(classifier.diagnose, sample_trace)
        assert result is not None
    
    @pytest.mark.benchmark  
    def test_batch_processing_throughput(self, benchmark, sample_traces_100):
        """Benchmark batch processing throughput."""
        classifier = Classifier()
        
        def process_batch():
            return [classifier.diagnose(trace) for trace in sample_traces_100]
        
        results = benchmark(process_batch)
        assert len(results) == 100
    
    @pytest.mark.benchmark
    def test_memory_usage_scaling(self, memory_profiler, large_traces):
        """Test memory usage with increasing trace sizes."""
        classifier = Classifier()
        
        with memory_profiler:
            for trace in large_traces:
                classifier.diagnose(trace)
        
        # Assert memory doesn't grow unbounded
        assert memory_profiler.peak_usage < 500 * 1024 * 1024  # 500MB
```

## 📝 Documentation Standards

### API Documentation

**1. Function Documentation**
```python
def similarity(self, text1: str, text2: str) -> float:
    """Compute semantic similarity between two text strings.
    
    Uses sentence transformers to encode texts into embeddings and 
    computes cosine similarity. Results are cached for performance.
    
    Args:
        text1: First text string to compare
        text2: Second text string to compare
        
    Returns:
        Float between 0.0 (completely different) and 1.0 (identical).
        Values > 0.7 typically indicate high semantic similarity.
        
    Raises:
        ValueError: If either text is empty or None
        
    Example:
        >>> matcher = EmbeddingMatcher()
        >>> score = matcher.similarity("hello world", "hi earth") 
        >>> print(f"Similarity: {score:.2f}")
        Similarity: 0.73
        
    Note:
        First call may be slower due to model loading. Subsequent
        calls are cached and much faster.
    """
```

**2. Class Documentation**
```python
class ToolUseDetector(BaseDetector):
    """Detects tool selection and usage failures in agent traces.
    
    This detector identifies three main categories of tool-related failures:
    
    1. **Wrong Tool Selected**: Agent chose inappropriate tool for the task
       - Example: Using web_search for mathematical calculations
       - Detection: Task-tool semantic similarity, available tools ranking
       
    2. **Invalid Parameters**: Tool parameters don't match expected schema  
       - Example: Missing required fields, wrong data types
       - Detection: Schema validation, runtime inference from prior calls
       
    3. **Incorrect Parameter Values**: Valid structure but wrong values
       - Example: Searching for "cat" when task asks about "dog"
       - Detection: Value grounding analysis, LLM reasoning
    
    The detector uses a multi-stage pipeline with early stopping:
    Stage 1 (deterministic) → Stage 2 (heuristic) → Stage 3 (LLM fallback)
    
    Attributes:
        SIMILARITY_THRESHOLD: Minimum similarity for task-tool alignment (0.45)
        MIN_CONFIDENCE: Threshold for reporting failures (0.30)
        
    Example:
        >>> detector = ToolUseDetector(llm_judge=GeminiLLMJudge())
        >>> result = detector.detect(trace)
        >>> if result.subtype == "wrong_tool_selected":
        >>>     print(f"Wrong tool used: {result.reason}")
    """
```

### Example Documentation

**1. Cookbook Examples**
```python
# docs/examples/cookbook/basic_diagnosis.py
"""
Basic Agent Diagnosis Example

This example shows how to diagnose a simple agent execution
with tool use and goal satisfaction checking.
"""

from agent_diagnostician import Classifier
from agent_diagnostician.models.trace import AgentTrace

def main():
    # Sample agent execution data
    trace_data = {
        "run_id": "example_001",
        "task": "Find the weather in San Francisco",
        "status": "success",
        "total_steps": 2,
        "final_output": "The weather in San Francisco is sunny, 72°F",
        "steps": [
            {
                "step_index": 0,
                "tool_name": "web_search", 
                "tool_input": {"query": "San Francisco weather"},
                "tool_output": "Weather results...",
                "thought": "I need to search for current weather information"
            },
            {
                "step_index": 1,
                "tool_name": "format_response",
                "tool_input": {"data": "Weather results..."},
                "tool_output": "The weather in San Francisco is sunny, 72°F"
            }
        ]
    }
    
    # Create trace and run diagnosis
    trace = AgentTrace(**trace_data)
    classifier = Classifier()
    result = classifier.diagnose(trace)
    
    # Print results
    print(f"🔍 Diagnosis Results for {trace.run_id}")
    print(f"   Failure Type: {result.failure_type.value}")
    print(f"   Subtype: {result.subtype}")
    print(f"   Confidence: {result.confidence_score:.2f} ({result.confidence_band.value})")
    print(f"   Status: {'❌ Issue Found' if result.failure_type.value != 'none' else '✅ No Issues'}")
    
    if result.reason:
        print(f"   Reason: {result.reason}")
        
    if result.fix_direction:
        print(f"   💡 Suggestion: {result.fix_direction}")
    
    # Show evidence chain
    if result.evidence:
        print(f"\n📋 Evidence Chain:")
        for i, evidence in enumerate(result.evidence, 1):
            print(f"   {i}. {evidence.detection_stage}")
            print(f"      Signal: {evidence.signal} (+{evidence.confidence_contribution:.2f})")
            print(f"      Details: {evidence.explanation}")

if __name__ == "__main__":
    main()
```

## 🚢 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH` (e.g., `1.2.3`)
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backwards compatible  
- **PATCH**: Bug fixes, backwards compatible

### Release Checklist

**Pre-release:**
- [ ] All tests passing on main branch
- [ ] Documentation updated
- [ ] CHANGELOG.md updated with new features/fixes
- [ ] Version bumped in `pyproject.toml` and `__init__.py`
- [ ] Performance benchmarks run
- [ ] Integration tests with real LLM APIs

**Release:**
- [ ] Create release branch: `release/v1.2.3`
- [ ] Final testing and validation
- [ ] Create GitHub release with changelog
- [ ] Build and publish to PyPI
- [ ] Update conda-forge recipe (if applicable)

**Post-release:**
- [ ] Merge release branch to main
- [ ] Update development dependencies
- [ ] Plan next milestone features

### Hotfix Process

For critical bugs requiring immediate fixes:

1. Create hotfix branch from latest release tag
2. Apply minimal fix with tests
3. Fast-track review process
4. Release patch version immediately
5. Backport fix to main branch

## 🤝 Community & Communication

### Getting Help

**For Contributors:**
- 💬 [GitHub Discussions](https://github.com/your-org/agent-diagnostician/discussions) - Questions, ideas, showcase
- 🐛 [GitHub Issues](https://github.com/your-org/agent-diagnostician/issues) - Bug reports, feature requests
- 📧 [Email](mailto:contributors@agent-diagnostician.dev) - Private concerns, security issues

**For Users:**
- 📚 [Documentation](https://agent-diagnostician.readthedocs.io) - Complete guides and API reference
- 💡 [Examples](https://github.com/your-org/agent-diagnostician/tree/main/examples) - Real-world usage patterns
- 🆘 [Support](mailto:support@agent-diagnostician.dev) - Implementation help

### Code Review Guidelines

**For Authors:**
- Keep PRs focused and reasonably sized (< 400 lines when possible)
- Write clear commit messages and PR descriptions
- Respond to feedback promptly and professionally
- Add tests for all new functionality

**For Reviewers:**
- Focus on correctness, maintainability, and performance
- Provide constructive, specific feedback
- Approve when code meets standards, not when perfect
- Consider the contributor's experience level

### Recognition

We recognize contributors through:
- **Contributors** section in README
- **Release notes** acknowledging significant contributions  
- **Maintainer** status for sustained high-quality contributions
- **Advisory** roles for domain expertise and guidance

---

Thank you for contributing to Agent Diagnostician! Your efforts help make agent debugging more accessible and effective for the entire community. 🙏

Questions about contributing? Feel free to reach out via GitHub Discussions or email us at contributors@agent-diagnostician.dev.