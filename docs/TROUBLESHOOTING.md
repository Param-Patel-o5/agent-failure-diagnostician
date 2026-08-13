# Troubleshooting

## LLM API setup

Set provider and key before running detectors that use the judge:

```powershell
$env:LLM_PROVIDER = "gemini"
$env:LLM_API_KEY = "your-key"
$env:LLM_MODEL = "gemini-3.5-flash-lite"
```

Smoke test:

```powershell
python scripts/configure_llm.py --test
```

Supported providers: `gemini`, `openai`, `anthropic`, `mock` (no API).

## Rate limits (Gemini free tier)

Free-tier Gemini models are often capped around **15 requests per minute**. Hallucination detection calls the judge **once per step**, so a single trace can consume several requests.

Use the phased evaluator (respects limits and skips API for fixtures that resolve without LLM):

```powershell
python scripts/run_phased_fixture_evaluation.py --initial-used 0 --first-batch 3 --batch-size 15
```

`--initial-used` — calls already consumed in the current minute window.

## Mock vs live LLM in tests

`MockLLMJudge` returns uncertain verdicts. It is fine for local dev, but **not** a substitute for live API validation on:

- All `HALLUCINATION/` fixtures (blend requires real judge)
- `classifier_e2e/` traces
- Goal misinterpretation / aggregator cases
- Fixtures with `llm_fallback`, `llm_primary`, or `llm_uncertain` in the name

Hallucination **grounding-only escalation** runs only when the LLM API fails — not when the judge returns uncertain/low confidence.

## Common errors

| Error | Fix |
|-------|-----|
| `GEMINI_API_KEY` / auth errors | Set `LLM_API_KEY` or `GEMINI_API_KEY` |
| Model not found | Set `LLM_MODEL` to a model your key supports (e.g. `gemini-3.5-flash-lite`) |
| Quota / rate limit | Wait 60s, reduce batch size, or use phased runner |
| `FileNotFoundError` for prompts | Run from repo root; install package or use `scripts/` with path bootstrap |
| Empty trace / no steps | Trace JSON must include `steps` array |

## Evaluation results

Latest benchmark numbers live in [`docs/evaluation/SUMMARY.md`](evaluation/SUMMARY.md) and [`docs/evaluation/fixture_results.json`](evaluation/fixture_results.json).

Re-run after code changes:

```powershell
python scripts/run_phased_fixture_evaluation.py --output docs/evaluation
```
