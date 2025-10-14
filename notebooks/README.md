# Finetuning Notebooks Guide

## Overview

This directory contains two notebooks for the complete BMW service manual finetuning workflow:

1. **`finetune_qlora.ipynb`** - Complete QLoRA finetuning pipeline for Llama-3.1-8B
2. **`test_inference.ipynb`** - Systematic testing and evaluation of the finetuned model

> **Note**: These notebooks provide a free Colab alternative to HuggingFace AutoTrain. For the easiest training experience, use AutoTrain (see main README.md).

## Quick Start

### 1. Upload Files to Google Drive

Create this folder structure in Google Drive:
```
/MyDrive/bmw_finetuning/
└── data/
    ├── hf_train_autotrain.jsonl
    └── hf_val_synthetic.jsonl
```

**Files to upload**:
- `data/hf_train_autotrain.jsonl` (2,510 examples, 438KB)
- `data/hf_val_synthetic.jsonl` (248 examples, 42KB)

### 2. Open in Google Colab

**Option A: Direct upload**
1. Go to [Google Colab](https://colab.research.google.com/)
2. `File` → `Upload notebook`
3. Select `finetune_qlora.ipynb`

**Option B: From GitHub** (if you push to GitHub)
1. Go to [Google Colab](https://colab.research.google.com/)
2. `File` → `Open notebook` → `GitHub` tab
3. Enter your repository URL

### 3. Enable GPU

**Critical**: You must use a GPU runtime!

1. `Runtime` → `Change runtime type`
2. Hardware accelerator: **GPU** (T4 is fine)
3. Click `Save`

### 4. Get HuggingFace Token

You need a token to access Llama-3.1 (gated model):

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a token with `read` permissions
3. Accept Llama-3.1 license at [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

### 5. Run Notebook

Run cells in order (Cell 1 → Cell 12):

1. **Cell 1**: Install packages + login (enter HF token)
2. **Cell 2**: Mount Google Drive
3. **Cell 3**: Load datasets
4. **Cell 4**: Load Llama-3.1-8B with QLoRA
5. **Cell 5**: Configure training
6. **Cell 6**: Train! ⏱️ ~2-3 hours on T4 (8B model)
7. **Cell 7**: Evaluate
8. **Cell 8**: Save model (local + Drive)
9. **Cell 9**: Test inference
10. **Cell 10**: (Optional) Push to HF Hub
11. **Cell 11**: (Optional) Test loading from Hub

## Expected Results

### Training Metrics

**Good training** (should look like this):
```
Epoch 1: train_loss=1.25, eval_loss=1.30
Epoch 2: train_loss=0.85, eval_loss=0.92
Epoch 3: train_loss=0.65, eval_loss=0.71
```

**Signs of overfitting** (stop early if you see this):
```
Epoch 3: train_loss=0.45, eval_loss=1.05  ← val loss increased!
```

### Inference Examples

**Spec task**:
```
Q: [SPEC] What is the torque for cylinder head bolts?
A: 45 Nm
```

**Procedure task**:
```
Q: [PROCEDURE] How do you adjust valve clearance?
A: 1. Remove valve cover
   2. Rotate engine to TDC
   3. Measure clearance with feeler gauge
   4. Adjust shim thickness as needed
   5. Recheck clearance
```

**Explanation task**:
```
Q: [EXPLANATION] Explain the Motronic control unit
A: The Motronic control unit is an integrated engine management
   system that controls fuel injection timing and ignition advance
   based on sensor inputs...
```

## Troubleshooting

### "CUDA out of memory"

**Solution 1**: Reduce batch size in config.yaml
```yaml
training:
  per_device_train_batch_size: 4  # Was 8
  gradient_accumulation_steps: 4  # Was 2 (keep effective=16)
```

**Solution 2**: Use T4 runtime (free tier)
- `Runtime` → `Change runtime type` → `T4 GPU`

**Solution 3**: Restart runtime and try again
- `Runtime` → `Restart runtime`

### "Session timeout" (Colab free tier)

Colab free tier disconnects after ~12 hours idle. To prevent:

1. **Use Colab Pro** ($10/month) - recommended for serious training
2. **Save checkpoints frequently** (already configured in notebook)
3. **Resume from checkpoint** if disconnected:
   ```python
   trainer.train(resume_from_checkpoint=True)
   ```

### "Cannot access Llama-3.1"

You need to:
1. Create HuggingFace account
2. Accept Llama license: [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
3. Create access token: [HF Settings](https://huggingface.co/settings/tokens)
4. Enter token in Cell 1

### "Files not found in Drive"

Check your Google Drive structure:
```bash
/MyDrive/bmw_finetuning/
└── data/
    ├── hf_train_autotrain.jsonl   ← Must exist
    └── hf_val_synthetic.jsonl     ← Must exist
```

Run Cell 2 to verify all files are detected.

## Performance Tips

### Faster Training

1. **Use Colab Pro** (A100 GPU):
   - 3x faster than T4
   - Training: ~45-60 min vs ~2-3 hours (8B model)

2. **Increase batch size** (if GPU allows):
   ```yaml
   per_device_train_batch_size: 16
   gradient_accumulation_steps: 1
   ```

3. **Use bfloat16** (on A100):
   ```yaml
   training:
     fp16: false
     bf16: true  # Better for A100
   ```

### Better Quality

1. **Increase LoRA rank**:
   ```yaml
   qlora:
     r: 32          # Was 16
     lora_alpha: 64  # Was 32
   ```

2. **Train longer** (if not overfitting):
   ```yaml
   training:
     num_epochs: 5  # Was 3
   ```

3. **Lower learning rate**:
   ```yaml
   training:
     learning_rate: 1e-4  # Was 2e-4
   ```

## Output Files

After training, you'll have:

### Local Files (Colab VM)
```
./bmw_e30_qlora_results/        # Training checkpoints
./bmw_e30_m3_service_manual/    # Final model
```

### Google Drive (Persistent)
```
/MyDrive/bmw_finetuning/models/bmw_e30_m3_service_manual/
├── adapter_config.json
├── adapter_model.safetensors   # LoRA weights (~50 MB)
├── tokenizer.json
└── tokenizer_config.json
```

### HuggingFace Hub (if pushed)
```
https://huggingface.co/your-username/bmw-e30-m3-service-manual
```

## Using the Finetuned Model

### Load from local files

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "./bmw_e30_m3_service_manual"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Generate
messages = [{"role": "user", "content": "[SPEC] What is the torque?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Load from HuggingFace Hub

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "your-username/bmw-e30-m3-service-manual"

# Load directly (adapter + base model)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto"
)
model = PeftModel.from_pretrained(model, model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)
```

## Next Steps

1. **Evaluate thoroughly**: Test on validation set
2. **Compare to baseline**: Test base Llama-3.2 without finetuning
3. **Iterate**: Adjust hyperparameters based on results
4. **Deploy**: Create inference API or demo
5. **Share**: Push to HuggingFace Hub for others to use

## Resources

- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - Original QLoRA technique
- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) - Base model info
- [PEFT Docs](https://huggingface.co/docs/peft) - Parameter-Efficient Fine-Tuning
- [TRL Docs](https://huggingface.co/docs/trl) - Transformer Reinforcement Learning
- [Colab Guide](https://colab.research.google.com/notebooks/intro.ipynb) - Getting started with Colab

## Inference Testing Notebook

### test_inference.ipynb

**Purpose**: Systematic testing and evaluation of your finetuned model.

**What it does**:
1. Loads model from HuggingFace Hub
2. Runs predefined test cases across all task types
3. Evaluates on validation set with ground truth comparison
4. Provides accuracy metrics and error analysis
5. Interactive testing mode for custom queries

**Requirements**:
- Model pushed to HuggingFace Hub
- (Optional) Validation data uploaded to Drive

**Expected metrics**:
- SPEC tasks: 90%+ exact match
- PROCEDURE tasks: 80%+ accuracy (numbered steps)
- EXPLANATION tasks: 70%+ quality (coherent, factual)
- Overall: 75-85% combined accuracy

**Usage**:
1. Open `test_inference.ipynb` in Colab
2. Update `model_id` to your HuggingFace model
3. Run cells 1-7 for full evaluation
4. Cell 8 for interactive testing
5. Cell 9 to test different temperatures

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review cell outputs for error messages
3. Verify GPU is enabled (`Runtime` → `Change runtime type`)
4. Check Google Drive file paths are correct
5. Ensure HuggingFace token has correct permissions

## Costs

- **Colab Free Tier**: Free (T4 GPU, session limits)
- **Colab Pro**: $10/month (A100 access, longer sessions)
- **HuggingFace Hub**: Free (public models, up to 100GB)

Training on free tier is sufficient for this project!
