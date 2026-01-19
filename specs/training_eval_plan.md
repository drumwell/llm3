# VLM Fine-tuning & Evaluation Plan

> Implementation plan for Phase 2: Training and evaluation infrastructure.
> **Phase 1 (reorganization) is complete.** The project has been restructured as documented below.

## Overview

**Goal:** Fine-tune Qwen2-VL-7B on the BMW E30 M3 service manual Q&A dataset and evaluate its ability to answer technical questions from manual images.

**Stack:**
- Model: Qwen2-VL-7B-Instruct
- Training: Modal (GPU cloud) + HuggingFace Transformers + LoRA
- Eval: DeepEval framework + Claude-as-judge

**Resolved Configuration:**
- Dataset: 12,410 Q&A pairs (90/10 train/val split)
- HuggingFace: Personal account (user's choice)
- Model: Qwen2-VL-7B-Instruct
- DeepEval: Free tier (Confident AI dashboard optional)

---

## Phase 1: Project Reorganization ✅ COMPLETE

### Current Directory Structure

```
vlm3/
├── pipeline/                    # Data pipeline
│   ├── scripts/                 # 01-09 scripts
│   ├── tests/                   # Pipeline tests
│   └── config.yaml              # Pipeline config
│
├── training/                    # Fine-tuning infrastructure
│   ├── configs/
│   │   └── lora_qwen2vl.yaml    # LoRA training config
│   └── README.md
│
├── eval/                        # Evaluation framework (DeepEval)
│   ├── benchmarks/
│   │   └── manual_probes.json   # Hand-crafted critical questions
│   └── README.md
│
├── scraping/                    # Data collection (placeholder)
│   └── README.md
│
├── specs/                       # Project-wide specifications
│   └── training_eval_plan.md    # This file
│
├── training_data/               # Pipeline outputs
│   ├── vlm_train.jsonl          # ~11,169 examples
│   ├── vlm_val.jsonl            # ~1,241 examples
│   └── images/
│
├── data_src/                    # Source materials (unchanged)
├── work/                        # Pipeline intermediates (unchanged)
├── Makefile                     # Pipeline orchestration
└── README.md                    # Project overview
```

---

## Phase 2: Training Infrastructure

### 2.1 Dataset Preparation

The pipeline outputs `training_data/vlm_train.jsonl` in this format:
```json
{
  "image": "images/21-03.jpg",
  "conversations": [
    {"role": "user", "content": "What is the torque specification for..."},
    {"role": "assistant", "content": "The torque specification is..."}
  ],
  "metadata": {...}
}
```

Qwen2-VL expects a slightly different format. `prepare_dataset.py` will:
1. Load the JSONL files
2. Convert to HuggingFace Dataset
3. Format conversations for Qwen2-VL chat template
4. Push to HuggingFace Hub (private dataset)

### 2.2 Modal Training App

```python
# training/modal_train.py (pseudocode structure)

import modal

app = modal.App("vlm3-training")

# Docker image with all dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch", "transformers", "accelerate", "peft",
        "bitsandbytes", "datasets", "qwen-vl-utils"
    ])
)

# Shared volume for checkpoints
volume = modal.Volume.from_name("vlm3-checkpoints", create_if_missing=True)

@app.function(
    image=training_image,
    gpu="A100-80GB",  # or H100 if available
    timeout=3600 * 8,  # 8 hours max
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("huggingface")]
)
def train(
    dataset_id: str,           # HuggingFace dataset ID
    base_model: str = "Qwen/Qwen2-VL-7B-Instruct",
    lora_r: int = 64,
    lora_alpha: int = 128,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    output_name: str = "vlm3-e30m3"
):
    """Fine-tune Qwen2-VL with LoRA on the E30 M3 dataset."""
    # Load model with 4-bit quantization
    # Configure LoRA adapters
    # Load dataset from HuggingFace
    # Train with HF Trainer
    # Save adapter weights to volume
    # Push to HuggingFace Hub
    pass

@app.local_entrypoint()
def main(dataset_id: str, epochs: int = 3):
    train.remote(dataset_id=dataset_id, epochs=epochs)
```

### 2.3 Training Configuration

**LoRA Config (recommended starting point):**
```yaml
# training/configs/lora_qwen2vl.yaml
base_model: Qwen/Qwen2-VL-7B-Instruct
method: lora

lora:
  r: 64
  alpha: 128
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  lr_scheduler: cosine
  warmup_ratio: 0.03
  max_grad_norm: 1.0

quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_quant_type: nf4

eval:
  eval_steps: 100
  save_steps: 100
  logging_steps: 10
```

### 2.4 Estimated Training Costs

| Config | GPU | Time (est.) | Cost (Modal) |
|--------|-----|-------------|--------------|
| LoRA 7B, 3 epochs, ~5k examples | A100-80GB | 2-4 hours | $8-16 |
| LoRA 7B, 3 epochs, ~5k examples | H100 | 1-2 hours | $10-20 |
| Full fine-tune (not recommended) | 4xA100 | 8-12 hours | $100+ |

**Recommendation:** Start with LoRA on A100-80GB. Only consider full fine-tune if LoRA results are insufficient.

---

## Phase 3: Evaluation Framework (DeepEval)

### 3.0 Baseline Eval (IMPORTANT: Run First!)

**Before fine-tuning**, establish baseline performance on Qwen2-VL-7B-Instruct:

```bash
# Run baseline eval on unmodified model
deepeval test run eval/test_vlm.py --model Qwen/Qwen2-VL-7B-Instruct

# Save baseline results
deepeval test run eval/test_vlm.py \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --output eval/reports/baseline_qwen2vl.json
```

This provides a reference point to measure fine-tuning improvement.

### 3.1 Why DeepEval

- Built-in LLM-as-judge metrics (no custom prompt engineering)
- Pytest integration — run evals like tests
- Supports Claude as evaluator model
- Dashboard for tracking runs over time (Confident AI - optional)
- Simple to start, extensible when needed

### 3.2 DeepEval Metrics We'll Use

**Core metrics (implement immediately):**

| Metric | What it measures | Threshold | Use case |
|--------|------------------|-----------|----------|
| `AnswerRelevancyMetric` | Does answer address the question? | > 0.7 | All QA pairs |
| `FaithfulnessMetric` | Is answer grounded in context? | > 0.7 | Verify against ground truth |
| `GEval` | Custom criteria via LLM | > 0.7 | Domain-specific quality |
| `NumericExactMatch` | Torque specs, measurements match exactly | > 0.85 | Specifications |
| `KeywordPresence` | Required technical terms present | > 0.80 | Procedures, components |

**Custom metrics to implement:**

```python
# eval/metrics.py
from deepeval.metrics import BaseMetric

class NumericExactMatch(BaseMetric):
    """Check if numeric values (torque, pressure, etc.) match exactly."""
    def measure(self, test_case):
        # Extract numbers from expected and actual
        # Compare with tolerance (e.g., ±1 Nm for torque)
        pass

class KeywordPresence(BaseMetric):
    """Check if required technical keywords are present."""
    def measure(self, test_case):
        # Extract expected keywords from metadata
        # Check presence in actual output
        pass
```

### 3.3 Basic Test Structure

```python
# eval/test_vlm.py
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase

# Configure Claude as the evaluator
import deepeval
deepeval.set_evaluator_model("claude-3-5-sonnet-20241022")

@pytest.fixture
def answer_relevancy():
    return AnswerRelevancyMetric(threshold=0.7)

@pytest.fixture
def correctness():
    return GEval(
        name="Correctness",
        criteria="Is the answer factually correct for BMW E30 M3 service information?",
        evaluation_params=["input", "actual_output", "expected_output"],
        threshold=0.7
    )

def test_vlm_response(answer_relevancy, correctness, vlm_model, test_case):
    """Test a single VLM response."""
    # Generate model output
    model_output = vlm_model.predict(test_case["image"], test_case["question"])

    # Create DeepEval test case
    eval_case = LLMTestCase(
        input=test_case["question"],
        actual_output=model_output,
        expected_output=test_case["expected_answer"],
        context=[test_case.get("context", "")]
    )

    # Assert passes thresholds
    assert_test(eval_case, [answer_relevancy, correctness])
```

### 3.4 Running Evals

```bash
# Run all eval tests
deepeval test run eval/test_vlm.py

# Run with specific model
deepeval test run eval/test_vlm.py --model hf://your-model

# Quick smoke test (subset)
deepeval test run eval/test_vlm.py -k "torque or procedure" --max-samples 20

# Generate report
deepeval test run eval/test_vlm.py --output eval/reports/
```

### 3.5 Eval Pipeline Flow

```
training_data/vlm_val.jsonl
       │
       ▼
┌──────────────────┐
│  run_eval.py     │
│  - Load model    │
│  - Load val set  │
│  - Generate      │
│    predictions   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  DeepEval        │
│  - AnswerRelevancy│
│  - Faithfulness  │
│  - GEval         │
│  (Claude judge)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  DeepEval Report │
│  - Pass/fail     │
│  - Score breakdown│
│  - Failure details│
└──────────────────┘
```

### 3.6 Manual Benchmark Probes

Still hand-craft critical test cases for known-answer validation:

```python
# eval/benchmarks/manual_probes.py
PROBES = [
    {
        "id": "torque_001",
        "category": "specifications",
        "image": "images/27-05.jpg",
        "question": "What is the torque specification for the cylinder head bolts?",
        "expected_answer": "Stage 1: 40 Nm, Stage 2: 90 degrees, Stage 3: 90 degrees",
    },
    {
        "id": "procedure_001",
        "category": "procedures",
        "image": "images/21-12.jpg",
        "question": "What are the steps to remove the clutch assembly?",
        "expected_keywords": ["transmission", "pressure plate", "alignment tool"],
    },
    # ... 20-30 total probes
]
```

### 3.7 Success Criteria

| Metric | Threshold | Notes |
|--------|-----------|-------|
| AnswerRelevancy | > 0.7 | DeepEval default |
| Faithfulness | > 0.7 | Grounded in context |
| GEval Correctness | > 0.7 | Domain-specific |
| Manual probe pass rate | > 85% | Critical questions |

### 3.8 Iteration Based on Results

**If thresholds not met:**
1. Check DeepEval failure breakdown — which metric fails most?
2. Analyze by content_type — diagrams worse than text?
3. Review training data for failing categories
4. Regenerate QA for weak sections → retrain → re-eval

**Adding complexity later:**
- Custom numeric match metric for specs
- Hallucination detection for safety-critical info
- A/B comparison between model versions

---

## Phase 4: Inference & Deployment

### 4.1 Modal Serving

```python
# training/modal_serve.py (pseudocode)

@app.cls(
    image=inference_image,
    gpu="A10G",  # Cheaper GPU for inference
    container_idle_timeout=300,
)
class VLM3Model:
    @modal.enter()
    def load_model(self):
        # Load base model + LoRA adapter
        pass

    @modal.method()
    def predict(self, image_path: str, question: str) -> str:
        # Run inference
        pass

    @modal.web_endpoint(method="POST")
    def api(self, request):
        # HTTP endpoint for external access
        pass
```

### 4.2 Local Testing

```python
# Simple local test script
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

base_model = Qwen2VLForConditionalGeneration.from_pretrained(...)
model = PeftModel.from_pretrained(base_model, "path/to/adapter")

# Test inference
```

---

## Phase 5: Iteration Loop

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   Pipeline Output ──► Train ──► Eval ──► Analyze        │
│        ▲                                    │           │
│        │                                    │           │
│        └────── Regenerate weak sections ◄───┘           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Iteration triggers:**
- Eval scores below threshold
- Specific content types underperforming
- New source materials added to data_src/

---

## Implementation Checklist

### Phase 1: Reorganization ✅ COMPLETE
- [x] Create new directory structure
- [x] Move pipeline files to `pipeline/`
- [x] Update Makefile paths
- [x] Remove legacy LLM config from `pipeline/config.yaml`
- [x] Verify pipeline still works (`make status`, `pytest pipeline/tests/`)
- [x] Update root README and CLAUDE.md
- [x] Add placeholder READMEs to `training/`, `eval/`, `scraping/`

### Phase 2: Training
- [ ] Write `training/prepare_dataset.py`
- [ ] Write `training/modal_train.py`
- [ ] Training configs already created: `training/configs/lora_qwen2vl.yaml`
- [ ] Set up Modal secrets (HF token only — W&B skipped)
- [ ] Test training on small subset (~100 examples)
- [ ] Run full training
- [ ] Push model to HuggingFace

### Phase 3: Evaluation (DeepEval)
- [ ] **Run baseline eval on Qwen2-VL-7B-Instruct FIRST**
- [ ] Install DeepEval (`pip install deepeval`)
- [ ] Configure Claude as evaluator model
- [ ] Write `eval/run_eval.py` (load model, generate predictions)
- [ ] Write `eval/test_vlm.py` (DeepEval test cases)
- [ ] Implement `eval/metrics.py` (NumericExactMatch, KeywordPresence)
- [ ] Create `eval/conftest.py` (fixtures for model, test data)
- [ ] Create manual benchmark probes (20-30 questions) in `eval/benchmarks/manual_probes.json`
- [ ] Run baseline eval: `deepeval test run eval/test_vlm.py --model Qwen/Qwen2-VL-7B-Instruct`
- [ ] Run fine-tuned eval: `deepeval test run eval/test_vlm.py --model your-model`
- [ ] Compare baseline vs fine-tuned results
- [ ] Analyze failures by content type

### Phase 4: Deployment
- [ ] Write `training/modal_serve.py`
- [ ] Test inference endpoint
- [ ] Document API usage

### Phase 5: Iteration
- [ ] Review eval failures
- [ ] Identify weak areas
- [ ] Regenerate/improve data
- [ ] Retrain and re-eval

---

## Makefile Additions

```makefile
# Training targets
.PHONY: train train-dev prepare-dataset

prepare-dataset:
	python training/prepare_dataset.py \
		--train training_data/vlm_train.jsonl \
		--val training_training_data/vlm_val.jsonl \
		--output-repo $(HF_DATASET_REPO)

train:
	cd training && modal run modal_train.py \
		--dataset-id $(HF_DATASET_REPO) \
		--epochs 3

train-dev:  # Quick test run
	cd training && modal run modal_train.py \
		--dataset-id $(HF_DATASET_REPO) \
		--epochs 1 \
		--max-samples 100

# Eval targets (DeepEval)
.PHONY: eval eval-quick eval-probes

eval:
	deepeval test run eval/test_vlm.py \
		--verbose \
		--output eval/reports/

eval-quick:
	deepeval test run eval/test_vlm.py \
		-k "not slow" \
		--max-samples 50

eval-probes:
	deepeval test run eval/test_vlm.py \
		-k "manual_probe" \
		--verbose

eval-compare:  # Compare two model versions
	python eval/run_eval.py \
		--models $(MODEL_A) $(MODEL_B) \
		--val training_training_data/vlm_val.jsonl \
		--output eval/reports/comparison.json

# Serve targets
.PHONY: serve serve-local

serve:
	cd training && modal serve modal_serve.py

serve-local:
	python training/serve_local.py --model $(HF_MODEL_REPO)
```

---

## Timeline Estimate

| Phase | Tasks | Duration |
|-------|-------|----------|
| 1. Reorganization | Move files, update paths | 1-2 hours |
| 2. Training infra | Scripts, configs, test run | 4-6 hours |
| 3. Eval framework | DeepEval setup, test cases, probes | 2-3 hours |
| 4. First training run | Full LoRA training | 2-4 hours (compute) |
| 5. First eval cycle | Run eval, analyze results | 1-2 hours |
| 6. Iteration (if needed) | Data fixes, retrain | Variable |

**Total estimated effort:** 10-18 hours of work + compute time

*Note: DeepEval reduces eval setup time vs. custom framework.*

---

## Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| Dataset size | 12,410 QA pairs (90/10 split → ~11,169 train, ~1,241 val) |
| Validation split | 90/10 (configured in pipeline/config.yaml) |
| HuggingFace org | Personal account |
| Experiment tracking | Skip W&B for simplicity (can add later) |
| Multi-image support | No — each QA pair references single image |
| DeepEval account | Free tier sufficient; Confident AI dashboard optional |

---

## Notes

- Phase 1 (reorganization) complete
- Phase 2 ready for implementation by separate agent
- Modal account required (free tier has GPU credits)
- HuggingFace account required for dataset/model hosting
- DeepEval: `pip install deepeval` — uses Claude as evaluator via existing Anthropic API key
- **Baseline eval on Qwen2-VL-7B-Instruct should run BEFORE fine-tuning**
