#!/usr/bin/env python3
"""
08_prepare_hf_dataset.py - Convert JSONL to HuggingFace chat format with class balancing

Transforms instruction-tuning JSONL into chat format with task prefixes and applies
class weighting to balance underrepresented tasks.

Usage:
    python scripts/08_prepare_hf_dataset.py \\
      --train data/train.jsonl \\
      --val data/val.jsonl \\
      --output-dir data \\
      --config config.yaml \\
      --duplicate-weights "procedure=2,wiring=3,troubleshooting=10"
"""

import json
import argparse
import logging
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any
import yaml
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('08_prepare_hf_dataset')


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    entries = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def count_tokens_rough(text: str) -> int:
    """Rough token count estimation (words * 1.3 for subword tokenization)."""
    # Simple whitespace + punctuation tokenization as proxy
    tokens = len(re.findall(r'\w+|[^\w\s]', text))
    return int(tokens * 1.3)  # Account for subword splitting


def validate_spec_output(output: str, regex: str) -> bool:
    """Validate spec output matches value-only regex."""
    return bool(re.match(regex, output))


def validate_procedure_output(output: str, step_regex: str) -> bool:
    """Validate procedure has numbered steps."""
    lines = [line.strip() for line in output.split('\n') if line.strip()]
    if not lines:
        return False
    # At least one line should match step pattern
    return any(re.match(step_regex, line) for line in lines)


def transform_to_chat_format(entry: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform instruction-tuning entry to chat format.

    Format:
    {
      "messages": [
        {"role": "user", "content": "[TASK] instruction"},
        {"role": "assistant", "content": "output"}
      ],
      "meta": {original metadata + validation status}
    }
    """
    task = entry['meta']['task']
    instruction = entry['instruction']
    output = entry['output']

    # Add task prefix to instruction
    user_content = f"[{task.upper()}] {instruction}"

    # Create chat format
    chat_entry = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ],
        "meta": {
            **entry['meta'],
            "original_instruction": instruction,  # Store without prefix
        }
    }

    # Validate output format
    validation = {"valid": True, "errors": []}

    if task == "spec":
        spec_regex = config['validation']['spec_output_regex']
        if not validate_spec_output(output, spec_regex):
            validation["valid"] = False
            validation["errors"].append(f"Spec output doesn't match regex: {output[:50]}")

    elif task in ["procedure", "troubleshooting"]:
        step_regex = config['validation']['step_line_regex']
        if not validate_procedure_output(output, step_regex):
            validation["valid"] = False
            validation["errors"].append(f"Missing numbered steps: {output[:50]}")

    chat_entry["meta"]["validation"] = validation

    # Token counting
    total_text = user_content + " " + output
    token_count = count_tokens_rough(total_text)
    chat_entry["meta"]["token_count"] = token_count

    return chat_entry


def apply_class_weights(
    entries: List[Dict[str, Any]],
    weights: Dict[str, int]
) -> List[Dict[str, Any]]:
    """
    Duplicate entries based on class weights to balance task distribution.

    Args:
        entries: List of chat-formatted entries
        weights: Dict of task -> duplication factor (e.g., {"procedure": 2})

    Returns:
        Expanded list with duplicates
    """
    weighted_entries = []
    duplication_stats = Counter()

    for entry in entries:
        task = entry['meta']['task']
        weight = weights.get(task, 1)

        # Add original
        weighted_entries.append(entry)

        # Add duplicates (weight-1 copies)
        for i in range(weight - 1):
            duplicate = entry.copy()
            # Mark as duplicate in metadata
            duplicate['meta'] = duplicate['meta'].copy()
            duplicate['meta']['duplicate_id'] = i + 1
            weighted_entries.append(duplicate)

        duplication_stats[task] += weight

    return weighted_entries, duplication_stats


def write_jsonl(entries: List[Dict[str, Any]], path: Path):
    """Write list of dicts to JSONL file."""
    with open(path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')


def compute_statistics(
    entries: List[Dict[str, Any]],
    split_name: str
) -> Dict[str, Any]:
    """Compute dataset statistics."""
    stats = {
        "split": split_name,
        "total_entries": len(entries),
        "task_distribution": Counter(),
        "token_stats": {
            "min": float('inf'),
            "max": 0,
            "mean": 0,
            "over_512": 0
        },
        "validation_errors": []
    }

    token_counts = []

    for entry in entries:
        task = entry['meta']['task']
        stats["task_distribution"][task] += 1

        token_count = entry['meta']['token_count']
        token_counts.append(token_count)

        if token_count > 512:
            stats["token_stats"]["over_512"] += 1

        # Collect validation errors
        if not entry['meta']['validation']['valid']:
            stats["validation_errors"].append({
                "task": task,
                "errors": entry['meta']['validation']['errors']
            })

    # Token statistics
    if token_counts:
        stats["token_stats"]["min"] = min(token_counts)
        stats["token_stats"]["max"] = max(token_counts)
        stats["token_stats"]["mean"] = sum(token_counts) / len(token_counts)

    return stats


def format_statistics_report(
    before_stats: Dict[str, Any],
    after_stats: Dict[str, Any],
    val_stats: Dict[str, Any]
) -> str:
    """Format statistics as readable report."""
    report = []
    report.append("=" * 80)
    report.append("HuggingFace Dataset Preparation Report")
    report.append("=" * 80)
    report.append("")

    # Training set transformation
    report.append("TRAINING SET")
    report.append("-" * 80)
    report.append(f"Original entries: {before_stats['total_entries']}")
    report.append(f"After weighting: {after_stats['total_entries']}")
    report.append(f"Expansion ratio: {after_stats['total_entries'] / before_stats['total_entries']:.2f}x")
    report.append("")

    # Task distribution comparison
    report.append("Task Distribution (Train):")
    report.append(f"{'Task':<20} {'Before':<12} {'After':<12} {'Ratio':>8}")
    report.append("-" * 52)

    all_tasks = set(before_stats['task_distribution'].keys()) | set(after_stats['task_distribution'].keys())
    for task in sorted(all_tasks):
        before = before_stats['task_distribution'].get(task, 0)
        after = after_stats['task_distribution'].get(task, 0)
        ratio = f"{after/before:.1f}x" if before > 0 else "N/A"
        pct_before = f"{before/before_stats['total_entries']*100:.1f}%"
        pct_after = f"{after/after_stats['total_entries']*100:.1f}%"
        report.append(f"{task:<20} {before:>4} ({pct_before:>5}) {after:>4} ({pct_after:>5}) {ratio:>8}")
    report.append("")

    # Check balance (max/min ratio should be < 2x)
    counts_after = list(after_stats['task_distribution'].values())
    if len(counts_after) > 1:
        max_count = max(counts_after)
        min_count = min(counts_after)
        balance_ratio = max_count / min_count
        balance_status = "✅ BALANCED" if balance_ratio <= 2.0 else "⚠️ IMBALANCED"
        report.append(f"Balance ratio (max/min): {balance_ratio:.2f}x {balance_status}")
        report.append("")

    # Validation set
    report.append("VALIDATION SET")
    report.append("-" * 80)
    report.append(f"Total entries: {val_stats['total_entries']}")
    report.append("")
    report.append("Task Distribution (Val):")
    for task, count in sorted(val_stats['task_distribution'].items()):
        pct = count / val_stats['total_entries'] * 100
        report.append(f"  {task:<20} {count:>4} ({pct:>5.1f}%)")
    report.append("")

    # Token statistics
    report.append("TOKEN STATISTICS")
    report.append("-" * 80)

    for split, stats in [("Train", after_stats), ("Val", val_stats)]:
        ts = stats['token_stats']
        report.append(f"{split} set:")
        report.append(f"  Min tokens:     {ts['min']}")
        report.append(f"  Max tokens:     {ts['max']}")
        report.append(f"  Mean tokens:    {ts['mean']:.1f}")
        report.append(f"  Over 512 tokens: {ts['over_512']} ({ts['over_512']/stats['total_entries']*100:.1f}%)")

        if ts['over_512'] > 0:
            report.append(f"  ⚠️ WARNING: {ts['over_512']} examples exceed 512 tokens")
        else:
            report.append(f"  ✅ All examples under 512 tokens")
        report.append("")

    # Validation errors
    report.append("VALIDATION ERRORS")
    report.append("-" * 80)

    total_errors = len(after_stats['validation_errors']) + len(val_stats['validation_errors'])
    if total_errors == 0:
        report.append("✅ No validation errors detected")
    else:
        report.append(f"⚠️ Found {total_errors} validation errors:")
        report.append("")
        report.append("Training set errors:")
        for i, err in enumerate(after_stats['validation_errors'][:10], 1):  # Show first 10
            report.append(f"  {i}. [{err['task']}] {err['errors']}")
        if len(after_stats['validation_errors']) > 10:
            report.append(f"  ... and {len(after_stats['validation_errors']) - 10} more")

        report.append("")
        report.append("Validation set errors:")
        for i, err in enumerate(val_stats['validation_errors'][:10], 1):
            report.append(f"  {i}. [{err['task']}] {err['errors']}")
        if len(val_stats['validation_errors']) > 10:
            report.append(f"  ... and {len(val_stats['validation_errors']) - 10} more")

    report.append("")
    report.append("=" * 80)
    report.append("ACCEPTANCE CRITERIA")
    report.append("=" * 80)

    # Check criteria
    criteria = []

    # 1. Balanced task distribution
    counts_after = list(after_stats['task_distribution'].values())
    if len(counts_after) > 1:
        balance_ratio = max(counts_after) / min(counts_after)
        balanced = balance_ratio <= 2.0
        criteria.append(("✅" if balanced else "❌",
                        f"Train set balanced (max/min ≤ 2x): {balance_ratio:.2f}x"))

    # 2. Proper chat format (implicitly validated by transform)
    criteria.append(("✅", "All examples in proper chat format"))

    # 3. Token limit
    train_ok = after_stats['token_stats']['over_512'] == 0
    val_ok = val_stats['token_stats']['over_512'] == 0
    criteria.append(("✅" if train_ok else "⚠️",
                    f"Train: {after_stats['token_stats']['over_512']} examples over 512 tokens"))
    criteria.append(("✅" if val_ok else "⚠️",
                    f"Val: {val_stats['token_stats']['over_512']} examples over 512 tokens"))

    # 4. Validation errors
    no_errors = total_errors == 0
    criteria.append(("✅" if no_errors else "⚠️",
                    f"Validation errors: {total_errors}"))

    for status, msg in criteria:
        report.append(f"{status} {msg}")

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def parse_duplicate_weights(weights_str: str) -> Dict[str, int]:
    """
    Parse duplication weights string.

    Example: "procedure=2,wiring=3,troubleshooting=10" ->
             {"procedure": 2, "wiring": 3, "troubleshooting": 10}
    """
    weights = {}
    if not weights_str:
        return weights

    for pair in weights_str.split(','):
        task, weight = pair.strip().split('=')
        weights[task.strip()] = int(weight.strip())

    return weights


def main():
    parser = argparse.ArgumentParser(
        description='Convert JSONL to HuggingFace chat format with class balancing'
    )
    parser.add_argument('--train', type=Path, required=True,
                       help='Path to train.jsonl')
    parser.add_argument('--val', type=Path, required=True,
                       help='Path to val.jsonl')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for HF datasets')
    parser.add_argument('--config', type=Path, required=True,
                       help='Path to config.yaml')
    parser.add_argument('--duplicate-weights', type=str, default='',
                       help='Duplication weights as "task=weight,task=weight"')

    args = parser.parse_args()

    logger.info(f"Train file: {args.train}")
    logger.info(f"Val file: {args.val}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Duplicate weights: {args.duplicate_weights}")

    # Parse weights
    duplicate_weights = parse_duplicate_weights(args.duplicate_weights)
    if duplicate_weights:
        logger.info(f"Parsed weights: {duplicate_weights}")

    # Load config
    logger.info("Loading configuration...")
    config = load_config(args.config)

    # Load datasets
    logger.info("Loading training set...")
    train_entries = load_jsonl(args.train)
    logger.info(f"Loaded {len(train_entries)} training entries")

    logger.info("Loading validation set...")
    val_entries = load_jsonl(args.val)
    logger.info(f"Loaded {len(val_entries)} validation entries")

    # Compute before statistics (train only)
    logger.info("Computing before-transformation statistics...")
    before_stats = {
        "total_entries": len(train_entries),
        "task_distribution": Counter(e['meta']['task'] for e in train_entries)
    }

    # Transform to chat format
    logger.info("Transforming training set to chat format...")
    train_chat = [transform_to_chat_format(e, config) for e in train_entries]

    logger.info("Transforming validation set to chat format...")
    val_chat = [transform_to_chat_format(e, config) for e in val_entries]

    # Apply class weights to training set only
    if duplicate_weights:
        logger.info("Applying class weights to training set...")
        train_weighted, dup_stats = apply_class_weights(train_chat, duplicate_weights)
        logger.info(f"Expanded training set: {len(train_chat)} → {len(train_weighted)} entries")
        for task, count in dup_stats.items():
            logger.info(f"  {task}: duplicated {count}x")
    else:
        logger.info("No class weights specified, skipping duplication")
        train_weighted = train_chat

    # Compute after statistics
    logger.info("Computing after-transformation statistics...")
    after_stats = compute_statistics(train_weighted, "train")
    val_stats = compute_statistics(val_chat, "val")

    # Write output files
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_output = output_dir / "hf_train.jsonl"
    val_output = output_dir / "hf_val.jsonl"

    logger.info(f"Writing training set to {train_output}...")
    write_jsonl(train_weighted, train_output)
    logger.info(f"✓ Wrote {len(train_weighted)} entries")

    logger.info(f"Writing validation set to {val_output}...")
    write_jsonl(val_chat, val_output)
    logger.info(f"✓ Wrote {len(val_chat)} entries")

    # Generate and write report
    report = format_statistics_report(before_stats, after_stats, val_stats)

    # Write to log file
    log_dir = Path("work/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "hf_prep.log"

    logger.info(f"Writing report to {log_file}...")
    with open(log_file, 'w') as f:
        f.write(report)

    # Print report to console
    print("\n" + report)

    logger.info("Done!")


if __name__ == "__main__":
    main()
