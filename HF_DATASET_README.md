# HuggingFace Dataset - BMW Service Manual

## Overview

This dataset contains **1,185 training examples** and **158 validation examples** in chat format, ready for instruction-tuning with HuggingFace Transformers.

## Files

- `data/hf_train.jsonl` - 1,185 entries (778K) with class balancing
- `data/hf_val.jsonl` - 158 entries (91K) no duplication
- `work/logs/hf_prep.log` - Full preparation report

## Format

Each entry follows the chat format:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "[TASK] instruction text"
    },
    {
      "role": "assistant",
      "content": "expected output"
    }
  ],
  "meta": {
    "task": "spec|procedure|explanation|wiring|troubleshooting",
    "section_id": "section number",
    "page_no": "page identifier",
    "token_count": 42,
    "validation": {
      "valid": true,
      "errors": []
    }
  }
}
```

## Task Prefixes

All instructions are prefixed with task type for better model conditioning:

- `[SPEC]` - Extract specific technical values from specs
- `[PROCEDURE]` - Step-by-step repair/maintenance procedures
- `[EXPLANATION]` - Descriptive text about components/systems
- `[WIRING]` - Wiring diagram descriptions
- `[TROUBLESHOOTING]` - Diagnostic checklists

## Dataset Statistics

### Training Set (1,185 entries)

**Task Distribution (Balanced)**:
- Spec: 428 entries (36.1%)
- Explanation: 306 entries (25.8%)
- Procedure: 231 entries (19.5%)
- Wiring: 220 entries (18.6%)

**Balance Ratio**: 1.95x (max/min) ✅

**Token Statistics**:
- Min: 14 tokens
- Max: 267 tokens
- Mean: 55.2 tokens
- Over 512 tokens: 0 (0.0%) ✅

### Validation Set (158 entries)

**Task Distribution (Original, No Duplication)**:
- Spec: 110 entries (69.6%)
- Explanation: 33 entries (20.9%)
- Procedure: 10 entries (6.3%)
- Wiring: 4 entries (2.5%)
- Troubleshooting: 1 entry (0.6%)

**Token Statistics**:
- Min: 16 tokens
- Max: 286 tokens
- Mean: 40.1 tokens
- Over 512 tokens: 0 (0.0%) ✅

## Class Balancing

Applied duplication weights to training set only:

| Task            | Weight | Original | After | Expansion |
|-----------------|--------|----------|-------|-----------|
| spec            | 1x     | 428      | 428   | 1.0x      |
| explanation     | 2x     | 153      | 306   | 2.0x      |
| procedure       | 7x     | 33       | 231   | 7.0x      |
| wiring          | 10x    | 22       | 220   | 10.0x     |
| troubleshooting | 50x    | 0        | 0     | N/A       |

**Note**: Troubleshooting has no training examples (only 1 in validation set).

## Quality Checks

✅ **All acceptance criteria met**:
- Train set balanced (max/min ratio ≤ 2x): 1.95x
- All examples in proper chat format
- No examples exceed 512 tokens
- 0 validation errors

## Loading with HuggingFace

### Using datasets library

```python
from datasets import load_dataset

# Load from local files
dataset = load_dataset('json', data_files={
    'train': 'data/hf_train.jsonl',
    'validation': 'data/hf_val.jsonl'
})

# Access examples
train_data = dataset['train']
val_data = dataset['validation']

# Print first example
print(train_data[0])
```

### Using with Transformers

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

def format_chat(example):
    """Convert messages to tokenizer chat format."""
    return tokenizer.apply_chat_template(
        example['messages'],
        tokenize=False,
        add_generation_prompt=False
    )

# Apply formatting
train_formatted = train_data.map(
    lambda x: {'text': format_chat(x)},
    remove_columns=train_data.column_names
)
```

## Recommended Training Setup

### Single Multi-Task Model

Train one model on all tasks with task prefixes:

```yaml
model: meta-llama/Llama-3.1-8B-Instruct
method: LoRA

lora_config:
  r: 16-32
  lora_alpha: 32-64
  lora_dropout: 0.05-0.1
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

training_args:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8  # effective batch = 32
  learning_rate: 2e-4
  num_train_epochs: 3-5
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_seq_length: 512

  # Use gradient checkpointing for memory efficiency
  gradient_checkpointing: true

  # Monitor overfitting
  evaluation_strategy: "steps"
  eval_steps: 50
  save_strategy: "steps"
  save_steps: 50
  load_best_model_at_end: true
  metric_for_best_model: "loss"
```

### Task-Specific Adapters

Train separate LoRA adapters per task:

```python
# Group 1: Spec adapter (short, factual)
spec_data = train_data.filter(lambda x: x['meta']['task'] == 'spec')

# Group 2: Procedure adapter (step-by-step)
procedure_data = train_data.filter(lambda x: x['meta']['task'] in ['procedure', 'troubleshooting'])

# Group 3: Explanation adapter (descriptive)
explanation_data = train_data.filter(lambda x: x['meta']['task'] in ['explanation', 'wiring'])
```

## Evaluation Metrics

### Per-Task Metrics

**Spec**:
- Exact match accuracy
- Fuzzy match (allowing unit variations)
- F1 score on value extraction

**Procedure**:
- ROUGE-L (step ordering)
- BERTScore (semantic similarity)
- Human eval (correctness)

**Explanation**:
- BLEU, ROUGE scores
- Perplexity
- Human eval (factual accuracy)

**Wiring/Troubleshooting**:
- ROUGE scores
- Semantic similarity
- Human eval

### Example Evaluation Code

```python
from datasets import load_metric

rouge = load_metric('rouge')
bleu = load_metric('bleu')

def evaluate(model, val_data):
    predictions = []
    references = []

    for example in val_data:
        # Generate prediction
        prompt = example['messages'][0]['content']
        pred = model.generate(prompt)
        predictions.append(pred)

        # Get reference
        ref = example['messages'][1]['content']
        references.append(ref)

    # Compute metrics
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bleu_scores = bleu.compute(predictions=predictions, references=references)

    return rouge_scores, bleu_scores
```

## Regenerating Dataset

To regenerate with different class weights:

```bash
# Edit weights in Makefile or run directly:
python scripts/08_prepare_hf_dataset.py \\
  --train data/train.jsonl \\
  --val data/val.jsonl \\
  --output-dir data \\
  --config config.yaml \\
  --duplicate-weights "spec=1,explanation=2,procedure=7,wiring=10,troubleshooting=50"

# Check results
cat work/logs/hf_prep.log
```

## Data Augmentation Ideas

1. **Paraphrase instructions**: "What is X?" → "Tell me X"
2. **Multi-hop reasoning**: Combine multiple specs
3. **Negative examples**: "not found in this section"
4. **Back-translation**: English → German → English

## Known Limitations

1. **Small dataset**: 1,185 training examples may lead to overfitting
2. **No troubleshooting in train**: Only 1 example exists (in val set)
3. **OCR errors**: Some text may be misread from scanned images
4. **Fragmented procedures**: Multi-page procedures may be incomplete

## Citation

```bibtex
@dataset{bmw_service_manual_2025,
  title={BMW Service Manual Instruction-Tuning Dataset},
  author={Your Name},
  year={2025},
  howpublished={\\url{https://github.com/yourusername/llm3}},
  note={Extracted from scanned BMW service manual pages}
}
```

## License

Check original BMW service manual licensing. This dataset is for research/educational purposes only.
