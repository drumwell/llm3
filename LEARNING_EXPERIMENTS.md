# QLoRA Finetuning Learning Experiments

## Experiment Log Template

For each model you train, record:
1. **Hypothesis**: What are you testing?
2. **Config**: Base model, LoRA rank, LR, epochs, batch size
3. **Training observations**: Loss curves, time taken
4. **Eval results**: Validation loss, sample outputs
5. **Learnings**: What worked? What didn't?

---

## Experiment 1: Baseline with Enhanced Dataset

### Hypothesis
Establish baseline performance with Llama-3.2-3B on enhanced dataset (1,877 examples including HTML tech specs).

### Config
- Base model: `meta-llama/Llama-3.2-3B-Instruct`
- LoRA rank: 16
- Learning rate: 2e-4
- Epochs: 3
- Batch size: 4
- Dataset: 1,877 train, 235 val (includes HTML tech specs)

### Results (from initial run)
- Training time: 3 minutes on A100
- Final eval_loss: 1.4571
- Mean token accuracy: 76.65%

### Test Queries
| Query | Expected | Got | Status |
|-------|----------|-----|--------|
| `[SPEC] What is the engine displacement?` | `2.3 L` | `4.0 V` | ❌ Hallucination |
| `[SPEC] What is the tightening torque for engine section 11?` | `45 Nm` | ? | ⏳ Test needed |
| `[PROCEDURE] How do you adjust valve clearance?` | Numbered steps | ? | ⏳ Test needed |

### Learnings
- ✅ Model trains successfully, metrics look healthy
- ✅ Architecture (QLoRA + task prefixes) working
- ❌ **Critical gap**: Missing general specs from index pages
- ❌ Model hallucinates plausible-sounding answers for untrained data
- 📊 **Data coverage is more important than model size**

### Next Steps
- [ ] Test queries model WAS trained on (validate it's not broken)
- [ ] Add index page specs to training data
- [ ] Retrain and compare

---

## Experiment 2: Add Index Page Specs (Planned)

### Hypothesis
Adding general engine specs (displacement, bore, stroke, compression ratio) will eliminate hallucinations for basic queries.

### Config
- Base model: Same (`Llama-3.2-3B-Instruct`)
- LoRA rank: 16
- Learning rate: 2e-4
- Epochs: 3
- **Dataset**: 1,877 train (includes 692 HTML specs)

### Data Added (✅ Complete)
Extracted from M3-techspec.html and 320is-techspec.html using `scripts/07_extract_html_specs.py`:

```json
{"messages": [{"role": "user", "content": "What is the engine displacement?"}, {"role": "assistant", "content": "2.3 L"}]}
{"messages": [{"role": "user", "content": "How many liters is the S14 engine?"}, {"role": "assistant", "content": "2.3 L"}]}
{"messages": [{"role": "user", "content": "What is the cubic capacity?"}, {"role": "assistant", "content": "2302 cm³"}]}
{"messages": [{"role": "user", "content": "What is the bore and stroke?"}, {"role": "assistant", "content": "95.0 × 84.0 mm"}]}
{"messages": [{"role": "user", "content": "What is the compression ratio?"}, {"role": "assistant", "content": "10.5:1"}]}
{"messages": [{"role": "user", "content": "What is the power output?"}, {"role": "assistant", "content": "147 kW / 195 hp @ 6750 rpm"}]}
```

**Key insight**: Add 3-5 phrasings per spec to help generalization.

### Expected Results
- Similar eval_loss (~1.4-1.5)
- ✅ Correct answers for displacement, bore, stroke
- ✅ No more "4.0 V" hallucinations

### Learnings to Capture
- How many examples per spec are needed?
- Does adding data hurt performance on existing queries?
- Is 3 epochs still optimal with more data?

---

## Experiment 3: Bigger Model (Planned)

### Hypothesis
Mistral-7B-Instruct will better handle technical jargon and complex procedures than Llama-3.2-3B.

### Config
- Base model: `mistralai/Mistral-7B-Instruct-v0.3`
- LoRA rank: 16 (keep same for fair comparison)
- Learning rate: 2e-4
- Epochs: 3
- Dataset: Same as Experiment 2 (~1,235 examples)

### What to Compare
| Metric | Llama-3.2-3B | Mistral-7B | Winner |
|--------|--------------|------------|--------|
| Eval loss | 1.4571 | ? | ? |
| Spec accuracy | ? | ? | ? |
| Procedure quality | ? | ? | ? |
| Training time | 3 min | ~5 min (est) | Llama |
| Inference speed | Fast | Slower | Llama |

### Learnings to Capture
- Is the quality improvement worth the size increase?
- Does Mistral handle technical terms (Nm, mm, kW) better?
- Would Mistral allow fewer training examples for same quality?

---

## Experiment 4: Hyperparameter Tuning (Planned)

### Hypothesis
Increasing LoRA rank to 32 will improve model's ability to learn technical patterns.

### Config
- Base model: Winner from Exp 2 vs 3
- **LoRA rank: 32** (was 16)
- Learning rate: 2e-4
- Epochs: 3
- Dataset: Same as Experiment 2

### What to Compare
| Config | Trainable Params | Eval Loss | Training Time | Quality |
|--------|------------------|-----------|---------------|---------|
| r=8 | ~4M | ? | Fastest | ? |
| r=16 (baseline) | ~8M | 1.4571 | 3 min | Known |
| r=32 | ~16M | ? | ~4 min | ? |
| r=64 | ~32M | ? | ~6 min | ? |

### Learnings to Capture
- Is there a point of diminishing returns?
- Does higher rank help with rare specs (e.g., compression ratio)?
- Does training become unstable at high ranks?

---

## Experiment 5: Data Quality vs Quantity (Planned)

### Hypothesis
Cleaning OCR errors will improve model quality more than adding more pages.

### Approach
1. Manually review 50 training examples for OCR errors
2. Fix common mistakes:
   - "0" vs "O" (zero vs letter O)
   - "1" vs "l" vs "I" (one vs L vs i)
   - Missing decimal points (45Nm → 45 Nm)
   - Garbled special tool numbers

### Config
- Same as best model so far
- **Dataset**: Cleaned version of 1,235 examples

### Learnings to Capture
- How many errors are in the current data?
- Does cleaning help more than adding 200 more examples?
- What types of errors hurt model most?

---

## Experiment 6: Task-Specific Models (Advanced)

### Hypothesis
Training separate models for SPEC vs PROCEDURE vs EXPLANATION might improve quality.

### Approach
- Train 3 separate models:
  - Model A: SPEC only (~300 examples)
  - Model B: PROCEDURE only (~400 examples)
  - Model C: EXPLANATION only (~300 examples)

### Expected Trade-offs
- ✅ Each model deeply specialized
- ❌ Need to route queries to correct model
- ❌ 3x training cost

### Learnings to Capture
- Is multi-task learning helping or hurting?
- Do SPEC tasks benefit from seeing PROCEDURE examples?

---

## Metrics to Track Across All Experiments

### Quantitative
- [ ] Final training loss
- [ ] Validation loss
- [ ] Training time (wall-clock)
- [ ] Token accuracy (if available)
- [ ] Inference time per query

### Qualitative
- [ ] Sample outputs for 10 held-out queries
- [ ] Hallucination rate (answers not in manual)
- [ ] Format compliance (e.g., torque values with units)
- [ ] Procedure numbering (1., 2., 3. vs unformatted)

### User Experience
- [ ] Response quality for BMW community users
- [ ] Confidence in answers (does it sound certain when wrong?)
- [ ] Usefulness for real mechanic tasks

---

## AutoTrain Workflow

### 1. Prepare Data
```bash
# Your data is already in HF format! Just need to upload.
# Each line: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### 2. Upload to HuggingFace Hub
```python
from datasets import load_dataset
from huggingface_hub import login

login()  # Paste your HF token

dataset = load_dataset('json', data_files={
    'train': 'data/hf_train.jsonl',
    'validation': 'data/hf_val.jsonl'
})

dataset.push_to_hub("your-username/bmw-e30-service-manual")
```

### 3. Launch AutoTrain
- Go to https://huggingface.co/autotrain
- Click "New Project" → "LLM Fine-tuning"
- Select your dataset: `your-username/bmw-e30-service-manual`
- Choose base model: `meta-llama/Llama-3.2-3B-Instruct`
- AutoTrain will suggest LoRA config
- Click "Train" → Costs ~$5-10

### 4. Monitor Training
- Watch loss curves in real-time
- Training completes in ~10-30 minutes
- Model auto-uploaded to your HF account

### 5. Test Inference
```python
from transformers import pipeline

pipe = pipeline("text-generation", model="your-username/bmw-e30-service-manual-v1")
result = pipe("[SPEC] What is the engine displacement?")
print(result[0]['generated_text'])
```

### 6. Compare Models
- Keep a spreadsheet of all experiments
- Track which config produced best results
- Share best model with BMW community

---

## Quick Reference: What Each Parameter Does

| Parameter | What It Controls | Typical Range | When to Increase | When to Decrease |
|-----------|------------------|---------------|------------------|------------------|
| **LoRA Rank (r)** | # trainable params | 8-64 | Model underfitting | Overfitting, slow training |
| **Learning Rate** | Step size in training | 1e-5 to 5e-4 | Training too slow | Loss is jumpy/unstable |
| **Epochs** | Passes through data | 1-10 | Eval loss still decreasing | Eval loss increasing (overfit) |
| **Batch Size** | Examples per update | 4-32 | Stable training, faster | More generalization needed |
| **Alpha** | LoRA scaling factor | r to 2×r | Increase with rank | Rarely changed |

---

## Resources for Learning

### Understanding QLoRA
- Paper: "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- Key idea: Train 4-bit quantized model to save memory, keeps quality

### HuggingFace Docs
- AutoTrain: https://huggingface.co/docs/autotrain
- PEFT (LoRA library): https://huggingface.co/docs/peft
- TRL (training library): https://huggingface.co/docs/trl

### Loss Curves Interpretation
- Decreasing train + val loss: ✅ Learning
- Decreasing train, flat val: 🟡 Plateau (might need more data)
- Decreasing train, increasing val: ❌ Overfitting (reduce epochs)

---

## Next Steps

**Today (30 minutes):**
1. Upload your dataset to HuggingFace Hub
2. Start Experiment 2 (Baseline on AutoTrain)
3. While training, prepare index page specs

**This Week:**
1. Run Experiments 2-4 (different models, add data)
2. Build evaluation script for consistent testing
3. Document learnings in this file

**Next Week:**
1. Share best model with BMW community
2. Gather feedback on real-world queries
3. Iterate based on user needs

Would you like me to help you start with uploading your data to HuggingFace Hub? That's the first step to getting hands-on with AutoTrain.
