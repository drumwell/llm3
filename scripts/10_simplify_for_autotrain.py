#!/usr/bin/env python3
"""
Simplify dataset for AutoTrain by removing complex nested metadata.

AutoTrain sometimes has issues with deeply nested structures in Parquet.
This script creates a cleaner version with just messages and text fields.
"""

import json
from pathlib import Path

def simplify_dataset(input_file, output_file):
    """Remove nested meta field, keep only messages."""
    count = 0
    with open(input_file) as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            obj = json.loads(line)
            # Keep only messages field
            simplified = {
                'messages': obj['messages']
            }
            f_out.write(json.dumps(simplified) + '\n')
            count += 1
    return count

def main():
    data_dir = Path('data')

    print("🔄 Simplifying dataset for AutoTrain compatibility...")

    train_count = simplify_dataset(
        data_dir / 'hf_train.jsonl',
        data_dir / 'hf_train_simple.jsonl'
    )

    val_count = simplify_dataset(
        data_dir / 'hf_val.jsonl',
        data_dir / 'hf_val_simple.jsonl'
    )

    print(f"✅ Simplified dataset created:")
    print(f"   Train: {train_count} examples → data/hf_train_simple.jsonl")
    print(f"   Val: {val_count} examples → data/hf_val_simple.jsonl")
    print(f"\n📤 Upload with:")
    print(f"   python scripts/09_upload_to_hf.py --repo drumwell/llm3 --force")

if __name__ == '__main__':
    main()
