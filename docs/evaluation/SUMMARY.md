# Fixture Evaluation Summary

Generated: 2026-08-13T06:44:19.077889+00:00

Phased honest run: **36 fixtures** resolved without API (mock / no-LLM detectors), **24 fixtures** with live `gemini-3.5-flash-lite` (**63 API calls**, 15 req/min limit). All hallucination fixtures validated with real LLM.

## Overall

| Metric | Value |
|--------|-------|
| Fixtures | 60 |
| Passed | 53 |
| Failed | 7 |
| **Accuracy** | **88.3%** |
| Total time | 251.8s |
| Avg per fixture | 4189.6ms |
| LLM provider | gemini |
| LLM model | gemini-3.5-flash-lite |

## By category

| Category | Total | Passed | Failed | Accuracy |
|----------|-------|--------|--------|----------|
| Goal Statisfiction | 11 | 9 | 2 | 81.8% |
| HALLUCINATION | 9 | 9 | 0 | 100.0% |
| TOOL_USE FAILURE | 14 | 10 | 4 | 71.4% |
| classifier_e2e | 5 | 4 | 1 | 80.0% |
| context_loss | 6 | 6 | 0 | 100.0% |
| infinite_loop | 9 | 9 | 0 | 100.0% |
| premature_termination | 6 | 6 | 0 | 100.0% |

## Failures

- `test cases/Goal Statisfiction/Aggregator Edge Cases/goal__aggregator__constraint_primary__misinterpretation_secondary.json`
  - expected: Goal Satisfaction Failure / constraint_violation
  - actual: goal_satisfaction_failure / task_misinterpretation
- `test cases/Goal Statisfiction/Aggregator Edge Cases/goal__aggregator__misinterpretation_primary__constraint_secondary.json`
  - expected: Goal Satisfaction Failure / task_misinterpretation
  - actual: goal_satisfaction_failure / constraint_violation
- `test cases/infinite_loop_and_e2e_traces/classifier_e2e/e2e__equal_confidence__priority_tiebreaker.json`
  - expected: goal_satisfaction_failure / task_misinterpretation
  - actual: hallucination / hallucination_detected
- `test cases/TOOL_USE FAILURE/Invalid Parameters/invalid_parameters__llm_fallback__first_call_edge_case.json`
  - expected: Tool Use Failure / Valid Tool, Invalid Parameters
  - actual: tool_use_failure / wrong_tool_selected
- `test cases/TOOL_USE FAILURE/Invalid Parameters/invalid_parameters__runtime_inference__multi_step.json`
  - expected: Tool Use Failure / Valid Tool, Invalid Parameters
  - actual: tool_use_failure / wrong_tool_selected
- `test cases/TOOL_USE FAILURE/Valid Paremeters , wrong values/incorrect_values__llm_fallback__ungrounded_ambiguous.json`
  - expected: Tool Use Failure / Valid Tool, Incorrect Parameter Values
  - actual: tool_use_failure / no_tool_use_failure
- `test cases/TOOL_USE FAILURE/Wrong tool selected/wrong_tool_selected__embedding_ranking__negative_control_close_match.json`
  - expected: None / No Tool Use Failure
  - actual: tool_use_failure / incorrect_parameter_values

## README-ready copy

> Validated on **60** local JSON trace fixtures with **gemini** (`gemini-3.5-flash-lite`), phased evaluation with live LLM on all required suites: **53/60 passed (88.3% accuracy)**.
>
> **100%** on context loss (6), infinite loop (9), hallucination (9), premature termination (6). **71.4%** tool use (10/14). **81.8%** goal satisfaction (9/11). **80%** classifier e2e (4/5).
