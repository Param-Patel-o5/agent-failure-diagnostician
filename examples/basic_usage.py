#!/usr/bin/env python3
"""
Agent Diagnostician - Basic Usage Examples

This script demonstrates the fundamental usage patterns of Agent Diagnostician
for diagnosing agent execution failures across different scenarios.
"""

from agent_diagnostician import Classifier
from agent_diagnostician.models.trace import AgentTrace, Step
from agent_diagnostician.models.enums import FailureType
from agent_diagnostician.analysis.llm import create_llm_judge_from_env
from agent_diagnostician.analysis.llm.config import PROVIDER_ENV_KEYS


def example_1_successful_execution():
    """Example 1: Analyzing a successful agent execution."""
    
    print("🔍 Example 1: Successful Agent Execution")
    print("="*50)
    
    # Create a trace for a successful weather query
    trace = AgentTrace(
        run_id="weather_001",
        task="What's the weather like in San Francisco today?",
        status="success",
        total_steps=2,
        final_output="The weather in San Francisco today is sunny with a high of 72°F and a low of 58°F. Perfect day for outdoor activities!",
        steps=[
            Step(
                step_index=0,
                tool_name="web_search",
                tool_input={"query": "San Francisco weather today"},
                tool_output={
                    "results": [
                        "Current weather: Sunny, 72°F",
                        "Today's forecast: High 72°F, Low 58°F, Sunny"
                    ]
                },
                thought="I need to search for current weather information in San Francisco"
            ),
            Step(
                step_index=1,
                tool_name="format_response",
                tool_input={"weather_data": "Sunny, 72°F, High 72°F, Low 58°F"},
                tool_output="The weather in San Francisco today is sunny with a high of 72°F and a low of 58°F. Perfect day for outdoor activities!"
            )
        ]
    )
    
    # Diagnose the trace
    classifier = Classifier()
    result = classifier.diagnose(trace)
    
    # Print results
    print(f"Status: {'✅ No Issues' if result.failure_type.value == 'none' else '❌ Issues Found'}")
    print(f"Failure Type: {result.failure_type.value}")
    print(f"Confidence: {result.confidence_score:.3f} ({result.confidence_band.value})")
    print(f"Reason: {result.reason}")
    print()


def example_2_wrong_tool_selection():
    """Example 2: Agent selects wrong tool for the task."""
    
    print("🔍 Example 2: Wrong Tool Selection")
    print("="*50)
    
    # Create a trace where agent uses wrong tool
    trace = AgentTrace(
        run_id="calculator_001", 
        task="Calculate the tip for a $85 dinner bill at 18%",
        status="success",
        total_steps=1,
        final_output="I found many search results about tip calculations, but I'm not sure of the exact answer.",
        steps=[
            Step(
                step_index=0,
                tool_name="web_search",  # Wrong! Should use calculator
                tool_input={"query": "calculate 18% tip on $85 bill"},
                tool_output=[
                    "Tip calculator websites",
                    "Articles about tipping etiquette", 
                    "Restaurant tipping guidelines"
                ],
                thought="I should search for information about calculating tips"
            )
        ]
    )
    
    classifier = Classifier()
    result = classifier.diagnose(trace)
    
    print(f"Status: {'✅ No Issues' if result.failure_type.value == 'none' else '❌ Issues Found'}")
    print(f"Failure Type: {result.failure_type.value}")
    print(f"Subtype: {result.subtype}")
    print(f"Confidence: {result.confidence_score:.3f} ({result.confidence_band.value})")
    print(f"Reason: {result.reason}")
    
    if result.fix_direction:
        print(f"💡 Fix Suggestion: {result.fix_direction}")
    
    print("\nEvidence:")
    for i, evidence in enumerate(result.evidence, 1):
        print(f"  {i}. {evidence.signal} (+{evidence.confidence_contribution:.3f})")
        print(f"     {evidence.explanation}")
    print()


def example_3_constraint_violation():
    """Example 3: Agent violates explicit task constraints."""
    
    print("🔍 Example 3: Constraint Violation")
    print("="*50)
    
    # Task with specific constraints
    trace = AgentTrace(
        run_id="code_001",
        task="Write a Python function to calculate factorial. The function must be recursive and include proper docstring documentation.",
        status="success", 
        total_steps=1,
        final_output="""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
""",
        steps=[
            Step(
                step_index=0,
                tool_name="code_generator",
                tool_input={
                    "language": "python",
                    "task": "factorial function",
                    "style": "iterative"  # Violates "recursive" constraint
                },
                tool_output="def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
            )
        ]
    )
    
    classifier = Classifier()
    result = classifier.diagnose(trace)
    
    print(f"Status: {'✅ No Issues' if result.failure_type.value == 'none' else '❌ Issues Found'}")
    print(f"Failure Type: {result.failure_type.value}")
    print(f"Subtype: {result.subtype}")
    print(f"Confidence: {result.confidence_score:.3f} ({result.confidence_band.value})")
    print(f"Reason: {result.reason}")
    
    if result.fix_direction:
        print(f"💡 Fix Suggestion: {result.fix_direction}")
    print()


def example_4_context_loss():
    """Example 4: Agent loses context from previous steps."""
    
    print("🔍 Example 4: Context Loss")
    print("="*50)
    
    # Multi-step trace where agent drops important context
    trace = AgentTrace(
        run_id="transfer_001",
        task="Transfer $500 from account ACC-12345 to account ACC-67890",
        status="success",
        total_steps=3,
        final_output="Transfer completed successfully",
        steps=[
            Step(
                step_index=0,
                tool_name="get_account_info",
                tool_input={"account_id": "ACC-12345"},
                tool_output={
                    "account_id": "ACC-12345",
                    "balance": 1200.50,
                    "account_type": "checking"
                }
            ),
            Step(
                step_index=1,
                tool_name="get_account_info", 
                tool_input={"account_id": "ACC-67890"},
                tool_output={
                    "account_id": "ACC-67890", 
                    "balance": 350.00,
                    "account_type": "savings"
                }
            ),
            Step(
                step_index=2,
                tool_name="execute_transfer",
                tool_input={
                    "from_account": "ACC-11111",  # Wrong! Should be ACC-12345
                    "to_account": "ACC-67890",
                    "amount": 500
                },
                tool_output="Transfer completed successfully",
                thought="I need to transfer money between the accounts"
            )
        ]
    )
    
    # Use only context loss detector for focused analysis
    classifier = Classifier(enabled_detectors=[FailureType.CONTEXT_LOSS])
    result = classifier.diagnose(trace)
    
    print(f"Status: {'✅ No Issues' if result.failure_type.value == 'none' else '❌ Issues Found'}")
    print(f"Failure Type: {result.failure_type.value}")
    print(f"Subtype: {result.subtype}")
    print(f"Confidence: {result.confidence_score:.3f} ({result.confidence_band.value})")
    print(f"Reason: {result.reason}")
    
    if result.evidence:
        print("\nEvidence:")
        for i, evidence in enumerate(result.evidence, 1):
            print(f"  {i}. Stage: {evidence.detection_stage}")
            print(f"     Signal: {evidence.signal}")
            print(f"     Confidence: +{evidence.confidence_contribution:.3f}")
            print(f"     Details: {evidence.explanation}")
    print()


def example_5_with_real_llm():
    """Example 5: Using real LLM judge for enhanced analysis."""
    
    print("🔍 Example 5: Enhanced Analysis with Real LLM")
    print("="*50)
    
    import os
    from agent_diagnostician.models.enums import LLMProvider

    provider = os.getenv("LLM_PROVIDER", LLMProvider.GEMINI.value).lower()
    api_key = os.getenv("LLM_API_KEY") or os.getenv(PROVIDER_ENV_KEYS.get(provider, ""), "")
    
    if not api_key:
        print("⚠️  Skipping LLM example - no API key found")
        print("   Set LLM_PROVIDER and LLM_API_KEY (or a provider-specific key):")
        print("   PowerShell: $env:LLM_PROVIDER='gemini'; $env:LLM_API_KEY='your-key'")
        print("   Or run: python scripts/configure_llm.py --provider gemini --api-key YOUR_KEY")
        print()
        return
    
    # Complex trace that benefits from LLM analysis
    trace = AgentTrace(
        run_id="complex_001",
        task="Find restaurants in Tokyo that serve vegetarian ramen and are open late night",
        status="success",
        total_steps=2,
        final_output="Here are some great sushi restaurants in Tokyo that are popular with tourists.",
        steps=[
            Step(
                step_index=0,
                tool_name="search_restaurants",
                tool_input={
                    "location": "Tokyo",
                    "cuisine": "Japanese",
                    "hours": "late_night"
                },
                tool_output=[
                    "Sushi Jiro - Famous sushi, open until 11pm",
                    "Ramen Yashichi - Traditional tonkotsu ramen, open 24h",
                    "Tempura Yamanoue - High-end tempura, closes at 10pm"
                ],
                thought="I should search for restaurants in Tokyo that match the criteria"
            ),
            Step(
                step_index=1,
                tool_name="filter_restaurants",
                tool_input={
                    "restaurants": ["Sushi Jiro", "Ramen Yashichi", "Tempura Yamanoue"],
                    "dietary_requirements": "popular_tourist"  # Wrong filter!
                },
                tool_output="Here are some great sushi restaurants in Tokyo that are popular with tourists."
            )
        ]
    )
    
    # Initialize classifier with real LLM judge (reads LLM_PROVIDER / LLM_API_KEY from environment)
    llm_judge = create_llm_judge_from_env()
    classifier = Classifier(llm_judge=llm_judge)
    
    print(f"🤖 Running analysis with {provider} LLM judge...")
    
    result = classifier.diagnose(trace)
    
    print(f"Status: {'✅ No Issues' if result.failure_type.value == 'none' else '❌ Issues Found'}")
    print(f"Failure Type: {result.failure_type.value}")
    print(f"Subtype: {result.subtype}")
    print(f"Confidence: {result.confidence_score:.3f} ({result.confidence_band.value})")
    print(f"Reason: {result.reason}")
    
    if result.fix_direction:
        print(f"💡 Fix Suggestion: {result.fix_direction}")
    
    print("\nLLM-Enhanced Evidence:")
    for i, evidence in enumerate(result.evidence, 1):
        print(f"  {i}. {evidence.detection_stage}")
        print(f"     {evidence.explanation}")
    print()


def example_6_batch_analysis():
    """Example 6: Batch processing multiple traces efficiently."""
    
    print("🔍 Example 6: Batch Analysis")
    print("="*50)
    
    # Create multiple traces for batch processing
    traces = [
        AgentTrace(
            run_id=f"batch_{i}",
            task=f"Process item {i}",
            status="success",
            total_steps=1,
            final_output=f"Completed processing item {i}",
            steps=[
                Step(
                    step_index=0,
                    tool_name="process_item",
                    tool_input={"item_id": i},
                    tool_output=f"Processed item {i}"
                )
            ]
        )
        for i in range(1, 6)
    ]
    
    # Add one trace with an issue
    problematic_trace = AgentTrace(
        run_id="batch_problem",
        task="Calculate the square root of 16",
        status="success", 
        total_steps=1,
        final_output="I searched for information about square roots but couldn't find a specific answer.",
        steps=[
            Step(
                step_index=0,
                tool_name="web_search",  # Wrong tool for calculation
                tool_input={"query": "square root of 16"},
                tool_output="Search results about square root calculations"
            )
        ]
    )
    traces.append(problematic_trace)
    
    # Process batch
    classifier = Classifier()
    results = []
    
    print(f"Processing {len(traces)} traces...")
    
    for trace in traces:
        result = classifier.diagnose(trace)
        results.append((trace, result))
    
    # Summarize results
    issues_found = sum(1 for _, result in results if result.failure_type.value != "none")
    
    print(f"\n📊 Batch Analysis Summary:")
    print(f"   Total traces: {len(traces)}")
    print(f"   Issues found: {issues_found}")
    print(f"   Success rate: {((len(traces) - issues_found) / len(traces) * 100):.1f}%")
    
    # Show details for problematic traces
    if issues_found > 0:
        print(f"\n❌ Issues detected:")
        for trace, result in results:
            if result.failure_type.value != "none":
                print(f"   • {trace.run_id}: {result.failure_type.value} - {result.subtype}")
                print(f"     Confidence: {result.confidence_band.value}")
    print()


def example_7_selective_detection():
    """Example 7: Using selective detection for performance optimization."""
    
    print("🔍 Example 7: Selective Detection")
    print("="*50)
    
    # Sample trace
    trace = AgentTrace(
        run_id="selective_001",
        task="Send an email to john@example.com with subject 'Meeting Tomorrow'",
        status="success",
        total_steps=1,
        final_output="Email sent successfully",
        steps=[
            Step(
                step_index=0,
                tool_name="send_email",
                tool_input={
                    "to": "john@example.com",
                    "subject": "Meeting Tomorrow",
                    "body": "Hi John, just a reminder about our meeting tomorrow at 2 PM."
                },
                tool_output="Email sent successfully"
            )
        ]
    )
    
    # Compare different detection modes
    detection_modes = [
        ("Fast (Tool Use Only)", [FailureType.TOOL_USE_FAILURE]),
        ("Balanced", [FailureType.TOOL_USE_FAILURE, FailureType.GOAL_SATISFACTION_FAILURE]),
        ("Comprehensive", None)  # All detectors
    ]
    
    import time
    
    for mode_name, enabled_detectors in detection_modes:
        classifier = Classifier(enabled_detectors=enabled_detectors)
        
        start_time = time.time()
        result = classifier.diagnose(trace)
        elapsed = time.time() - start_time
        
        print(f"{mode_name}:")
        print(f"   Time: {elapsed*1000:.1f}ms")
        print(f"   Result: {result.failure_type.value}")
        print(f"   Detectors run: {len(classifier.get_detector_status()['ran'])}")
        print()


if __name__ == "__main__":
    print("🚀 Agent Diagnostician - Basic Usage Examples")
    print("=" * 60)
    print()
    
    # Run all examples
    example_1_successful_execution()
    example_2_wrong_tool_selection()
    example_3_constraint_violation()
    example_4_context_loss()
    example_5_with_real_llm()
    example_6_batch_analysis()
    example_7_selective_detection()
    
    print("✨ All examples completed!")
    print("\nNext steps:")
    print("• See docs/INTEGRATIONS.md for framework integration patterns")
    print("• See docs/PERFORMANCE.md for embedding and detector selection tips")
    print("• See docs/API.md for the full API reference")