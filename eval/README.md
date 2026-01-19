# Model Evaluation

DeepEval-based evaluation framework for VLM fine-tuning.

## Status

Not yet implemented. See `specs/training_eval_plan.md` for the implementation plan.

## Planned Contents

```
eval/
├── run_eval.py           # Main eval runner
├── test_vlm.py           # DeepEval test cases (pytest compatible)
├── metrics.py            # Custom metrics (NumericExactMatch, KeywordPresence)
├── conftest.py           # DeepEval/pytest fixtures
├── benchmarks/
│   └── manual_probes.json  # Hand-crafted critical questions
├── reports/              # Generated eval reports
└── requirements.txt
```

## Quick Start (Future)

```bash
# Install DeepEval
pip install deepeval

# Run baseline eval on Qwen2-VL-7B (before fine-tuning)
deepeval test run eval/test_vlm.py --model Qwen/Qwen2-VL-7B-Instruct

# Run eval on fine-tuned model
deepeval test run eval/test_vlm.py --model your-username/vlm3-finetuned

# Run just manual probes
deepeval test run eval/test_vlm.py -k "manual_probe"
```

## Metrics

| Metric | What it measures | Threshold |
|--------|------------------|-----------|
| AnswerRelevancy | Does answer address the question? | > 0.7 |
| Faithfulness | Is answer grounded in context? | > 0.7 |
| GEval Correctness | Domain-specific quality | > 0.7 |
| NumericExactMatch | Torque specs, measurements | > 0.85 |
| KeywordPresence | Required technical terms | > 0.80 |
