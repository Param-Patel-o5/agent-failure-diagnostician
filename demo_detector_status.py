#!/usr/bin/env python3
"""
Demo script showing the new detector status reporting feature.
Shows which detectors ran and which were skipped.
"""

from agent_diagnostician.classifier import Classifier
from agent_diagnostician.reporter import Reporter
from agent_diagnostician.models.enums import FailureType
from agent_diagnostician.models.trace import AgentTrace, Step

# Create a simple test trace
trace = AgentTrace(
    run_id="demo_001",
    task="Write a Python function to calculate fibonacci numbers",
    status="completed", 
    total_steps=2,
    final_output="def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
    steps=[
        Step(
            step_index=0,
            tool_name="code_writer", 
            tool_input={"language": "python", "task": "fibonacci function"},
            tool_output="def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"
        ),
        Step(
            step_index=1,
            tool_name="code_tester",
            tool_input={"code": "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)", "test_cases": [0, 1, 5]},
            tool_output={"results": [0, 1, 5], "all_passed": True}
        )
    ]
)

print("=== Demo: Detector Status Reporting ===\n")

# Example 1: Run all detectors
print("1. Running ALL detectors:")
classifier_all = Classifier()
result_all = classifier_all.diagnose(trace)
detector_status_all = classifier_all.get_detector_status()

Reporter.print(result_all, format="cli", detector_status=detector_status_all)

print("\n" + "="*60 + "\n")

# Example 2: Run only some detectors
print("2. Running ONLY token exhaustion and tool use detectors:")
classifier_selective = Classifier(enabled_detectors=[
    FailureType.TOKEN_EXHAUSTION,
    FailureType.TOOL_USE_FAILURE
])
result_selective = classifier_selective.diagnose(trace)  
detector_status_selective = classifier_selective.get_detector_status()

Reporter.print(result_selective, format="cli", detector_status=detector_status_selective)

print("\nDemo complete!")