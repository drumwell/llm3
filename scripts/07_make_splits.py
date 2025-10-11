#!/usr/bin/env python3
"""
07_make_splits.py - Create deterministic 80/20 train/val split across all JSONL slices.
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create train/val splits from JSONL files")
    parser.add_argument('--data-dir', default='data',
                        help='Directory containing JSONL files')
    parser.add_argument('--pattern', default='*.slice.jsonl',
                        help='Pattern to match input files')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Training set proportion (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print(f"[07_make_splits] Data directory: {data_dir}")
    print(f"[07_make_splits] Pattern: {args.pattern}")
    print(f"[07_make_splits] Train split: {args.train_split}")
    print(f"[07_make_splits] Random seed: {args.seed}")
    
    # Find all matching JSONL files
    jsonl_files = sorted(data_dir.glob(args.pattern))
    
    if not jsonl_files:
        print(f"[07_make_splits] ⚠ No files matching pattern found")
        return
    
    print(f"[07_make_splits] Found {len(jsonl_files)} files:")
    for f in jsonl_files:
        print(f"  - {f.name}")
    
    # Load all rows from all files
    all_rows = []
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                        all_rows.append(row)
                    except json.JSONDecodeError as e:
                        print(f"[07_make_splits] ⚠ Skip invalid JSON in {jsonl_file.name}: {e}")
    
    if not all_rows:
        print(f"[07_make_splits] ⚠ No valid rows found")
        return
    
    print(f"\n[07_make_splits] Total rows loaded: {len(all_rows)}")
    
    # Shuffle with fixed seed
    random.shuffle(all_rows)
    
    # Split
    n = len(all_rows)
    val_size = int(n * (1 - args.train_split))
    val_rows = all_rows[:val_size]
    train_rows = all_rows[val_size:]
    
    # Write splits
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for row in val_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    
    print(f"\n[07_make_splits] Split complete:")
    print(f"  Train: {len(train_rows)} rows → {train_file.name}")
    print(f"  Val:   {len(val_rows)} rows → {val_file.name}")
    print(f"  Total: {len(all_rows)} rows")
    
    # Show task distribution in each split
    train_tasks = {}
    for row in train_rows:
        task = row.get('meta', {}).get('task', 'unknown')
        train_tasks[task] = train_tasks.get(task, 0) + 1
    
    val_tasks = {}
    for row in val_rows:
        task = row.get('meta', {}).get('task', 'unknown')
        val_tasks[task] = val_tasks.get(task, 0) + 1
    
    print(f"\n[07_make_splits] Task distribution:")
    print(f"  Train: {dict(sorted(train_tasks.items()))}")
    print(f"  Val:   {dict(sorted(val_tasks.items()))}")
    
    print(f"\n[07_make_splits] Done!")


if __name__ == '__main__':
    main()
