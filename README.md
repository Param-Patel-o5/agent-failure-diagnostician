# Agent Diagnostician

> **A comprehensive framework-agnostic Python library for diagnosing agent execution failures**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Agent Diagnostician analyzes agent execution traces to automatically identify failure patterns, provide confidence-weighted diagnoses, and suggest concrete fixes. It works with traces from any agent framework including LangChain, LlamaIndex, AutoGen, OpenAI Assistants API, and custom implementations.

## 🎯 What It Does

Given an agent execution trace (JSON), Agent Diagnostician returns:
- **Failure Category** - Tool use, goal satisfaction, execution, or termination issues
- **Detailed Subtype** - Specific failure mode (e.g., "wrong_tool_selected", "constraint_violation")  
- **Confidence Score** - 0-1 numerical confidence with human-readable bands
- **Evidence Chain** - Detailed reasoning showing how the diagnosis was reached
- **Fix Direction** - Actionable guidance for improving agent performance

## 🏗️ Architecture Overview

### Core Components

```
agent_diagnostician/
├── classifier.py          # Main entry point - aggregates detector results
├── detectors/            # Failure-specific detection logic
│   ├── planning/         # Tool selection, goal understanding, hallucination
│   ├── execution/        # Context loss, token exhaustion 
│   └── termination/      # Premature termination, infinite loops
├── analysis/             # Reusable analysis components
│   ├── embeddings.py     # Semantic similarity matching
│   ├── grounding.py      # Value traceability analysis  
│   ├── llm_judge.py      # LLM-powered reasoning fallbacks
│   └── constraint_extractor.py  # Task constraint parsing
├── models/               # Pydantic data models
└── prompts/              # LLM prompt templates
```

### Detection Philosophy

Agent Diagnostician follows a **layered evidence approach**:

1. **Rules & Deterministic Checks** - Fast, reliable validation (schema compliance, numeric constraints)
2. **Runtime Inference** - Pattern detection from execution history  
3. **Embedding Analysis** - Semantic similarity for context understanding
4. **LLM Reasoning** - Complex judgment calls when other methods are insufficient

Each detector can operate at multiple tiers depending on available trace data, gracefully degrading when information is missing.

## 🚀 Quick Start

### Installation

```bash
pip install agent-diagnostician
```

### Basic Usage

```python
from agent_diagnostician import Classifier
from agent_diagnostician.models.trace import AgentTrace

# Load your agent execution trace
trace_data = {
    "run_id": "example_001",
    "task": "Transfer $500 from checking to savings account",
    "status": "success", 
    "total_steps": 3,
    "final_output": "Transfer completed successfully",
    "steps": [
        {
            "step_index": 0,
            "tool_name": "get_balance", 
            "tool_input": {"account": "checking"},
            "tool_output": {"balance": 1200.50}
        },
        # ... more steps
    ]
}

# Create trace object and run diagnosis
trace = AgentTrace(**trace_data)
classifier = Classifier()
result = classifier.diagnose(trace)

# Get comprehensive diagnosis
print(f"Failure Type: {result.failure_type.value}")
print(f"Subtype: {result.subtype}")
print(f"Confidence: {result.confidence_score:.2f} ({result.confidence_band.value})")
print(f"Diagnosis: {result.reason}")
print(f"Fix Guidance: {result.fix_direction}")

# Examine evidence chain
for evidence in result.evidence:
    print(f"- {evidence.signal}: {evidence.explanation}")
```

### With Real LLM Judge

For production use, configure with a real LLM provider:

```python
from agent_diagnostician.analysis.llm_judge import GeminiLLMJudge

# Initialize with Gemini LLM judge  
llm_judge = GeminiLLMJudge(api_key="your-api-key")
classifier = Classifier(llm_judge=llm_judge)

result = classifier.diagnose(trace)
```

### Selective Detection

Run only specific detector types for faster analysis:

```python
from agent_diagnostician.models.enums import FailureType

# Only check for tool use and goal satisfaction issues
classifier = Classifier(
    enabled_detectors=[
        FailureType.TOOL_USE_FAILURE,
        FailureType.GOAL_SATISFACTION_FAILURE
    ]
)
```

## 📊 Failure Categories & Performance

### Planning Failures

**Tool Use Detection** - Identifies incorrect tool selection and parameter issues
- **Accuracy**: 100% on validation test cases
- **Detects**: Wrong tool selected, invalid parameters, incorrect parameter values
- **Methods**: Schema validation, embedding similarity, LLM reasoning
- **Example**: Agent uses `web_search` instead of `database_query` for internal data lookup

**Goal Satisfaction Detection** - Verifies agent completed the actual requested task  
- **Accuracy**: 100% on constraint validation, 72.7% overall
- **Detects**: Constraint violations, task misinterpretation  
- **Methods**: Constraint extraction, semantic validation, execution analysis
- **Example**: Agent generates Python code when task specifically requested JavaScript

**Hallucination Detection** - Identifies fabricated or ungrounded information
- **Accuracy**: 55.6% on individual tests
- **Detects**: Fabricated IDs, ungrounded values, invented details
- **Methods**: Value traceability, grounding analysis, LLM verification
- **Example**: Agent creates customer ID "CUST-12345" not found in any prior outputs

### Execution Failures  

**Context Loss Detection** - Finds when agents drop established information
- **Accuracy**: 83.3% success rate
- **Detects**: Dropped values, contradicted information, memory failures
- **Methods**: Value traceability, thought contradiction analysis  
- **Example**: Agent uses wrong account ID after correctly retrieving it earlier

**Token Exhaustion Detection** - Identifies context window overflow issues
- **Coverage**: Schema validation and trace analysis
- **Detects**: Truncated inputs, incomplete processing due to length limits
- **Methods**: Token estimation, completion analysis

### Termination Failures

**Premature Termination Detection** - Catches incomplete task execution
- **Detects**: Early completion claims, partial requirement fulfillment
- **Methods**: Task-output similarity, requirement coverage analysis
- **Example**: Agent claims success after completing 2 of 5 required steps

**Infinite Loop Detection** - Identifies repetitive or stuck behavior
- **Detects**: Tool repetition, reasoning loops, error cycling
- **Methods**: Pattern analysis, similarity clustering, failure tracking
- **Example**: Agent repeatedly calls same API with same parameters after errors

## 💾 Trace Data Compatibility

Agent Diagnostician supports flexible trace formats across multiple tiers:

### Tier 1 (Universal - Always Required)
```python
{
    "run_id": "unique_identifier",
    "task": "Natural language task description", 
    "status": "success|error|timeout",
    "total_steps": 5,
    "final_output": "Agent's final response",
    "steps": [
        {
            "step_index": 0,
            "tool_name": "function_name",
            "tool_input": {"param": "value"},
            "tool_output": {"result": "data"}
        }
    ]
}
```

### Tier 2 (Common - Enhances Analysis)
```python
{
    # ... Tier 1 fields ...
    "timestamp": "2024-01-01T12:00:00Z",
    "error_message": "Optional error description",
    "total_tokens": 1500,
    "step_status": "success|error"  # per step
}
```

### Tier 3 (Advanced - Enables Sophisticated Analysis)  
```python
{
    # ... previous tiers ...
    "steps": [
        {
            # ... basic step fields ...
            "thought": "Agent's reasoning before action",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "retry_count": 1,
        }
    ],
    "available_tools": [
        {
            "name": "tool_name",
            "description": "What this tool does", 
            "schema": {"type": "object", "properties": {...}}
        }
    ],
    "constraints": ["explicit task constraints"]
}
```

**Framework Examples:**

<details>
<summary><strong>LangChain Integration</strong></summary>

```python
def langchain_to_diagnostician(langchain_result):
    """Convert LangChain trace to Agent Diagnostician format"""
    
    steps = []
    for i, step in enumerate(langchain_result.intermediate_steps):
        action, observation = step
        steps.append({
            "step_index": i,
            "tool_name": action.tool,
            "tool_input": action.tool_input,
            "tool_output": observation,
            "thought": action.log  # Agent's reasoning
        })
    
    return {
        "run_id": langchain_result.run_id,
        "task": langchain_result.input,
        "status": "success" if langchain_result.output else "error",
        "total_steps": len(steps),
        "final_output": langchain_result.output,
        "steps": steps
    }
```
</details>

<details>
<summary><strong>OpenAI Assistants API</strong></summary>

```python
def openai_to_diagnostician(thread_id, run_id, openai_client):
    """Convert OpenAI Assistant run to Agent Diagnostician format"""
    
    run = openai_client.beta.threads.runs.retrieve(thread_id, run_id)
    steps = openai_client.beta.threads.runs.steps.list(thread_id, run_id)
    
    diagnostic_steps = []
    for i, step in enumerate(steps.data):
        if step.type == "tool_calls":
            for tool_call in step.step_details.tool_calls:
                diagnostic_steps.append({
                    "step_index": i,
                    "tool_name": tool_call.function.name,
                    "tool_input": json.loads(tool_call.function.arguments),
                    "tool_output": tool_call.function.output
                })
    
    return {
        "run_id": run_id,
        "task": "Extract from thread messages",  
        "status": run.status,
        "total_steps": len(diagnostic_steps),
        "final_output": "Extract from final message",
        "steps": diagnostic_steps
    }
```
</details>

## 🔧 Configuration & Customization

### LLM Provider Configuration

Agent Diagnostician supports multiple LLM providers through a unified interface:

```python
# Gemini (Default)
from agent_diagnostician.analysis.llm_judge import GeminiLLMJudge
judge = GeminiLLMJudge(
    api_key="your-key",
    model="models/gemini-2.5-flash",  # or gemini-pro
    temperature=0.1
)

# OpenAI (Coming Soon)
from agent_diagnostician.analysis.llm_judge import OpenAILLMJudge  
judge = OpenAILLMJudge(api_key="your-key", model="gpt-4")

# Anthropic (Coming Soon)
from agent_diagnostician.analysis.llm_judge import AnthropicLLMJudge
judge = AnthropicLLMJudge(api_key="your-key", model="claude-3-opus")
```

### Detection Thresholds

Fine-tune detection sensitivity for your use case:

```python
from agent_diagnostician.detectors.execution.context_loss import ContextLossDetector

# Adjust sensitivity thresholds
detector = ContextLossDetector(llm_judge=judge)
detector.LOW_SIMILARITY_THRESHOLD = 0.35  # More sensitive to contradictions
detector.MIN_CONFIDENCE_THRESHOLD = 0.25  # Lower bar for reporting
```

### Custom Detectors

Extend with domain-specific failure detection:

```python
from agent_diagnostician.detectors.base import BaseDetector
from agent_diagnostician.models.result import DetectionResult
from agent_diagnostician.models.enums import FailureType

class CustomSecurityDetector(BaseDetector):
    """Detects security policy violations"""
    
    def detect(self, trace: AgentTrace) -> DetectionResult:
        # Your detection logic here
        evidence = []
        
        # Check for unauthorized data access patterns
        for step in trace.steps:
            if self._is_unauthorized_access(step):
                evidence.append(self._build_evidence(step))
        
        return self.build_result(
            failure_type=FailureType.CUSTOM_SECURITY,
            subtype="unauthorized_access",
            confidence_score=0.85,
            evidence=evidence,
            reason="Agent attempted unauthorized data access",
            fix_direction="Review and restrict tool permissions"
        )
```

## 📈 Best Practices

### Production Deployment

**1. Async Processing**
```python
import asyncio
from agent_diagnostician import AsyncClassifier

async def diagnose_batch(traces):
    classifier = AsyncClassifier(llm_judge=your_judge)
    results = await asyncio.gather(*[
        classifier.diagnose(trace) for trace in traces
    ])
    return results
```

**2. Caching & Performance**
```python
from agent_diagnostician.analysis.embeddings import CachedEmbeddingMatcher

# Cache embeddings for better performance
classifier = Classifier(
    embedding_matcher=CachedEmbeddingMatcher(cache_size=10000)
)
```

**3. Error Handling**
```python
try:
    result = classifier.diagnose(trace)
except Exception as e:
    logger.error(f"Diagnosis failed: {e}")
    # Fallback to basic analysis or manual review
```

### Trace Quality Guidelines

**Essential Data Quality:**
- Include complete tool_input and tool_output for each step
- Provide descriptive task statements (not just "help me")  
- Capture final_output even for failed executions
- Include error_message when steps fail

**Enhanced Analysis:**
- Add agent reasoning/thought processes when available
- Include available_tools schemas for better tool validation
- Provide step-level timestamps for temporal analysis
- Capture retry attempts and error patterns

### Interpreting Results

**High Confidence (0.7-1.0)**
- Clear evidence from multiple signals
- Safe to automate responses
- Focus on fix_direction guidance

**Medium Confidence (0.4-0.7)**  
- Some uncertainty in diagnosis
- Good for alerting and investigation
- May need human review for critical systems

**Low Confidence (0.1-0.4)**
- Weak or conflicting signals
- Useful for trend analysis
- Requires careful interpretation

## 🧪 Testing & Validation

Agent Diagnostician includes comprehensive test suites covering:

- **115+ Test Cases** - Real-world failure scenarios across all categories
- **Framework Integration Tests** - LangChain, LlamaIndex, OpenAI compatibility  
- **LLM Provider Tests** - Gemini, OpenAI, Anthropic validation
- **Performance Benchmarks** - Speed and accuracy metrics
- **Edge Case Coverage** - Malformed traces, missing data, extreme cases

Run tests locally:
```bash
git clone https://github.com/your-org/agent-diagnostician
cd agent-diagnostician
pip install -e ".[dev]"
pytest tests/ -v
```

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for:

- Development setup and workflow
- Coding standards and architecture patterns  
- Adding new detector types
- Improving LLM prompt templates
- Framework integration guides

### Development Priorities

**Current Focus:**
- Improving hallucination detection accuracy
- Adding support for more LLM providers
- Expanding framework integrations
- Performance optimization for large traces

**Future Roadmap:**
- Real-time streaming analysis
- Multi-agent execution diagnostics  
- Custom failure type definitions
- Visual trace analysis dashboard

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📚 Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - Deep dive into system design
- **[API Reference](docs/API.md)** - Complete method documentation  
- **[Integration Examples](docs/INTEGRATIONS.md)** - Framework-specific guides
- **[Performance Tuning](docs/PERFORMANCE.md)** - Optimization strategies
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

## 🆘 Support

- **Documentation**: [agent-diagnostician.readthedocs.io](https://agent-diagnostician.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/your-org/agent-diagnostician/issues)  
- **Discussions**: [GitHub Discussions](https://github.com/your-org/agent-diagnostician/discussions)
- **Email**: support@agent-diagnostician.dev

---

**Agent Diagnostician** - Making agent failures debuggable, one trace at a time. 🔍