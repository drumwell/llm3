# Model Configuration - Llama-3.2-3B-Instruct

## Configuration Update

Updated `config.yaml` to use **Llama-3.2-3B-Instruct** instead of Qwen2.5-7B.

## Why Llama-3.2-3B?

### Advantages for this project:

1. **Lower memory requirements** (~8-10 GB vs ~16 GB)
   - ✅ Runs on Colab free tier (T4 GPU with 15 GB)
   - ✅ Fits on RTX 3060 (12 GB)
   - ✅ Faster training (less timeout risk)

2. **Faster training speed**
   - 3B parameters vs 7B = ~2.3x faster per epoch
   - Better for iterative experimentation

3. **Still capable for task complexity**
   - Llama-3.2 architecture is strong for instruction-following
   - Excellent for technical/factual tasks
   - Your tasks are relatively simple (short outputs, clear patterns)

4. **Better for small datasets**
   - 1,185 training examples is small
   - Smaller model = less risk of overfitting
   - Easier to saturate model capacity

## Current Configuration

```yaml
huggingface:
  model_name: "meta-llama/Llama-3.2-3B-Instruct"
  task_prefix_format: "[{TASK}] {instruction}"
  max_seq_length: 512

qlora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

training:
  num_epochs: 3
  per_device_train_batch_size: 8   # Increased from 4 (smaller model)
  gradient_accumulation_steps: 2   # Decreased from 4 (keep effective=16)
  learning_rate: 2e-4
  optim: "paged_adamw_8bit"
  gradient_checkpointing: true
  fp16: true
```

## Memory Requirements

### With QLoRA (4-bit quantization):

| Configuration | VRAM Required | Hardware Examples |
|---------------|---------------|-------------------|
| **Minimum** | ~6 GB | RTX 3060, Colab T4 |
| **Recommended** | ~8 GB | RTX 3070, A4000 |
| **Optimal** | ~10 GB | RTX 3080, RTX 4070 |

### Training Speed Estimates:

**Dataset**: 1,185 training examples, batch size 16 (effective)

- **Steps per epoch**: 1185 / 16 = ~74 steps
- **Total steps (3 epochs)**: 222 steps
- **Time per step** (Colab T4): ~2-3 seconds
- **Total training time**: ~15-20 minutes per epoch
- **Full training**: **45-60 minutes** ✅

Compare to Qwen2.5-7B: 90-120 minutes (2x slower)

## Training Adjustments for 3B Model

### Increased batch size:
- **Before** (7B): batch=4, grad_accum=4 → effective=16
- **After** (3B): batch=8, grad_accum=2 → effective=16
- **Why**: Smaller model uses less memory, can fit larger batches
- **Benefit**: Faster training (fewer gradient accumulation steps)

### Same effective batch size (16):
- Maintains training stability
- Same gradient noise characteristics
- Comparable convergence behavior

## When to Use Llama-3.2-3B vs Larger Models

### Use Llama-3.2-3B when:
- ✅ Training on Colab free tier
- ✅ Limited GPU memory (<12 GB)
- ✅ Small dataset (< 5K examples)
- ✅ Simple tasks (factual extraction, short outputs)
- ✅ Need fast iteration cycles

### Use larger models (7B+) when:
- ❌ Complex reasoning required
- ❌ Long-form generation (>500 tokens)
- ❌ Large dataset (>10K examples)
- ❌ Multiple complex tasks simultaneously
- ❌ Have high-end GPU (24GB+)

## Expected Performance

### For your BMW manual tasks:

**Spec extraction** (428 examples):
- Expected accuracy: 90-95%
- Llama-3.2-3B is sufficient (simple pattern matching)

**Procedure generation** (231 examples):
- Expected quality: Good (numbered steps, clear instructions)
- 3B model handles structured output well

**Explanation generation** (306 examples):
- Expected quality: Good to Very Good
- May be slightly less fluent than 7B, but still coherent

**Wiring/Troubleshooting** (221 examples):
- Expected quality: Good
- Sufficient for technical descriptions

## Comparison: Llama-3.2-3B vs Alternatives

| Model | Params | VRAM | Speed | Quality | Best For |
|-------|--------|------|-------|---------|----------|
| **Llama-3.2-3B** | 3B | 8 GB | Fast | Good | Your use case ✅ |
| Llama-3.1-8B | 8B | 16 GB | Medium | Excellent | Complex reasoning |
| Qwen2.5-7B | 7B | 16 GB | Medium | Excellent | Multilingual |
| Gemma-2-9B | 9B | 18 GB | Slow | Excellent | Math/reasoning |
| Phi-3-Mini | 3.8B | 8 GB | Fast | Good | Similar to 3.2-3B |

## Upgrading to Larger Model Later

If you want to upgrade after initial results:

1. **Same dataset works**: No changes needed to JSONL files
2. **Update config.yaml**:
   ```yaml
   huggingface:
     model_name: "meta-llama/Llama-3.1-8B-Instruct"  # Upgrade

   training:
     per_device_train_batch_size: 4  # Reduce for memory
     gradient_accumulation_steps: 4  # Increase to keep effective=16
   ```
3. **Re-run training**: Same pipeline, just slower

## Training Setup

**Recommended: Use HuggingFace AutoTrain** for hassle-free training:

1. Upload dataset: `python scripts/09_upload_to_hf.py --repo your-username/bmw-e30-manual`
2. Go to https://huggingface.co/autotrain
3. Select your dataset and this model configuration
4. Train in ~15-30 minutes on A100 (~$5-10)

See `LEARNING_EXPERIMENTS.md` for detailed learning guide and experimentation tips.

## Summary

✅ **Updated to Llama-3.2-3B-Instruct**
- Lower memory: ~8 GB (vs 16 GB for 7B models)
- Faster training: ~15-30 min on A100
- Better for small datasets (1,877 examples)
- Sufficient for technical/factual tasks

✅ **Optimized training params**
- Batch size: 8 (increased from 4)
- Grad accumulation: 2 (decreased from 4)
- Effective batch: 16 (maintains stability)
- LoRA rank: 16 (good starting point)

✅ **Ready for HuggingFace AutoTrain**
- Upload dataset with script 09
- Auto-configured QLoRA
- Cost: ~$5-10 per training run
