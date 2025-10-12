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

## Colab Setup for Llama-3.2-3B

### Dependencies:

```bash
!pip install torch transformers datasets peft trl bitsandbytes accelerate tensorboard pyyaml
```

### Login to Hugging Face (for Llama access):

```python
from huggingface_hub import login
login()  # Enter your HF token
```

### Training script:

```python
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load dataset
dataset = load_dataset('json', data_files={
    'train': 'data/hf_train.jsonl',
    'validation': 'data/hf_val.jsonl'
})

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    config['huggingface']['model_name']
)
tokenizer.pad_token = tokenizer.eos_token

# Format chat
def format_chat(example):
    return tokenizer.apply_chat_template(
        example['messages'],
        tokenize=False,
        add_generation_prompt=False
    )

train_data = dataset['train'].map(
    lambda x: {'text': format_chat(x)},
    remove_columns=dataset['train'].column_names
)
val_data = dataset['validation'].map(
    lambda x: {'text': format_chat(x)},
    remove_columns=dataset['validation'].column_names
)

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    config['huggingface']['model_name'],
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# LoRA config
peft_config = LoraConfig(
    r=config['qlora']['r'],
    lora_alpha=config['qlora']['lora_alpha'],
    lora_dropout=config['qlora']['lora_dropout'],
    target_modules=config['qlora']['target_modules'],
    bias=config['qlora']['bias'],
    task_type=config['qlora']['task_type']
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Training args
training_args = TrainingArguments(
    output_dir="./checkpoints/llama-bmw",
    num_train_epochs=config['training']['num_epochs'],
    per_device_train_batch_size=config['training']['per_device_train_batch_size'],
    per_device_eval_batch_size=config['training']['per_device_train_batch_size'],
    gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
    learning_rate=config['training']['learning_rate'],
    warmup_ratio=config['training']['warmup_ratio'],
    weight_decay=config['training']['weight_decay'],
    optim=config['training']['optim'],
    logging_steps=config['training']['logging_steps'],
    save_strategy=config['training']['save_strategy'],
    evaluation_strategy=config['training']['evaluation_strategy'],
    save_total_limit=config['training']['save_total_limit'],
    load_best_model_at_end=config['training']['load_best_model_at_end'],
    metric_for_best_model=config['training']['metric_for_best_model'],
    gradient_checkpointing=config['training']['gradient_checkpointing'],
    fp16=config['training']['fp16'],
    report_to="none"  # Disable wandb for Colab
)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=config['huggingface']['max_seq_length'],
    peft_config=peft_config
)

# Train
print("Starting training...")
trainer.train()

# Save
trainer.save_model("./models/llama-bmw-final")
tokenizer.save_pretrained("./models/llama-bmw-final")
print("Done!")
```

## Summary

✅ **Updated to Llama-3.2-3B-Instruct**
- Lower memory: ~8 GB (vs 16 GB)
- Faster training: ~45-60 min total (vs 90-120 min)
- Colab-friendly: Fits on T4 GPU
- Still capable: Sufficient for your task complexity

✅ **Optimized training params**
- Batch size: 8 (increased from 4)
- Grad accumulation: 2 (decreased from 4)
- Effective batch: 16 (same)
- Faster training with same stability

✅ **Ready for Colab**
- All dependencies specified
- Login instructions included
- Expected training time: ~1 hour

Next step: Upload dataset to Colab and run training! 🚀
