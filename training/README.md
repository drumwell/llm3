# VLM Fine-tuning

Infrastructure for fine-tuning Qwen2-VL-7B on the BMW E30 M3 service manual Q&A dataset.

## Status

Not yet implemented. See `specs/training_eval_plan.md` for the implementation plan.

## Planned Contents

```
training/
├── modal_train.py        # Modal app for training
├── modal_serve.py        # Modal app for inference
├── prepare_dataset.py    # Convert JSONL → HF Dataset format
├── configs/
│   └── lora_qwen2vl.yaml # LoRA training config
├── requirements.txt
└── README.md
```

## Quick Start (Future)

```bash
# Prepare dataset for HuggingFace
python training/prepare_dataset.py \
    --train training_data/vlm_train.jsonl \
    --val training_data/vlm_val.jsonl \
    --output-repo your-username/vlm3-dataset

# Run training on Modal
cd training && modal run modal_train.py \
    --dataset-id your-username/vlm3-dataset \
    --epochs 3
```
