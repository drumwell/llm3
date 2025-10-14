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


def transform_to_autotrain_format(entry: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform instruction-tuning entry to flat text format for AutoTrain.

    Format:
    {
      "text": "User: [TASK] instruction\\nAssistant: output"
    }
    """
    task = entry['meta']['task']
    instruction = entry['instruction']
    output = entry['output']

    # Add task prefix to instruction
    user_content = f"[{task.upper()}] {instruction}"

    # Create flat text format
    text_entry = {
        "text": f"User: {user_content}\nAssistant: {output}"
    }

    # Store metadata for statistics (not included in output)
    meta = {
        **entry['meta'],
        "original_instruction": instruction,
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

    meta["validation"] = validation

    # Token counting
    token_count = count_tokens_rough(text_entry["text"])
    meta["token_count"] = token_count

    # Store meta separately for statistics
    text_entry["_meta"] = meta

    return text_entry


def clean_output_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove metadata from entries for final output.
    Keep only the 'text' field.
    """
    clean_entries = []
    for entry in entries:
        clean_entries.append({"text": entry["text"]})
    return clean_entries


def write_jsonl(entries: List[Dict[str, Any]], path: Path):
    """Write list of dicts to JSONL file."""
    with open(path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')


def compute_statistics(
    entries: List[Dict[str, Any]],
    split_name: str
) -> Dict[str, Any]:
    """Compute dataset statistics from entries with _meta field."""
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
        meta = entry.get('_meta', {})
        task = meta.get('task', 'unknown')
        stats["task_distribution"][task] += 1

        token_count = meta.get('token_count', 0)
        token_counts.append(token_count)

        if token_count > 512:
            stats["token_stats"]["over_512"] += 1

        # Collect validation errors
        validation = meta.get('validation', {})
        if not validation.get('valid', True):
            stats["validation_errors"].append({
                "task": task,
                "errors": validation.get('errors', [])
            })

    # Token statistics
    if token_counts:
        stats["token_stats"]["min"] = min(token_counts)
        stats["token_stats"]["max"] = max(token_counts)
        stats["token_stats"]["mean"] = sum(token_counts) / len(token_counts)

    return stats


def format_statistics_report(
    combined_stats: Dict[str, Any],
    original_train_count: int,
    original_val_count: int
) -> str:
    """Format statistics as readable report."""
    report = []
    report.append("=" * 80)
    report.append("AutoTrain Dataset Preparation Report")
    report.append("=" * 80)
    report.append("")

    # Consolidation summary
    report.append("DATASET CONSOLIDATION")
    report.append("-" * 80)
    report.append(f"Original train examples: {original_train_count}")
    report.append(f"Original val examples:   {original_val_count}")
    report.append(f"Combined total:          {combined_stats['total_entries']}")
    report.append("")
    report.append("✅ All service manual data consolidated into single training set")
    report.append("📝 Synthetic validation will be generated separately")
    report.append("")

    # Task distribution
    report.append("TASK DISTRIBUTION")
    report.append("-" * 80)
    report.append(f"{'Task':<20} {'Count':<12} {'Percentage':>12}")
    report.append("-" * 52)

    for task in sorted(combined_stats['task_distribution'].keys()):
        count = combined_stats['task_distribution'][task]
        pct = count / combined_stats['total_entries'] * 100
        report.append(f"{task:<20} {count:>4} {pct:>15.1f}%")
    report.append("")

    # Token statistics
    report.append("TOKEN STATISTICS")
    report.append("-" * 80)
    ts = combined_stats['token_stats']
    report.append(f"  Min tokens:      {ts['min']}")
    report.append(f"  Max tokens:      {ts['max']}")
    report.append(f"  Mean tokens:     {ts['mean']:.1f}")
    report.append(f"  Over 512 tokens: {ts['over_512']} ({ts['over_512']/combined_stats['total_entries']*100:.1f}%)")

    if ts['over_512'] > 0:
        report.append(f"  ⚠️ WARNING: {ts['over_512']} examples exceed 512 tokens")
    else:
        report.append(f"  ✅ All examples under 512 tokens")
    report.append("")

    # Validation errors
    report.append("VALIDATION ERRORS")
    report.append("-" * 80)

    total_errors = len(combined_stats['validation_errors'])
    if total_errors == 0:
        report.append("✅ No validation errors detected")
    else:
        report.append(f"⚠️ Found {total_errors} validation errors:")
        report.append("")
        for i, err in enumerate(combined_stats['validation_errors'][:10], 1):  # Show first 10
            report.append(f"  {i}. [{err['task']}] {err['errors']}")
        if len(combined_stats['validation_errors']) > 10:
            report.append(f"  ... and {len(combined_stats['validation_errors']) - 10} more")

    report.append("")
    report.append("=" * 80)
    report.append("OUTPUT FORMAT")
    report.append("=" * 80)
    report.append('Flat text format: {"text": "User: [TASK] question\\nAssistant: answer"}')
    report.append("✅ Compatible with AutoTrain LLM fine-tuning")
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
        description='Convert JSONL to AutoTrain flat text format'
    )
    parser.add_argument('--train', type=Path, required=True,
                       help='Path to train.jsonl')
    parser.add_argument('--val', type=Path, required=True,
                       help='Path to val.jsonl')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory for AutoTrain dataset')
    parser.add_argument('--config', type=Path, required=True,
                       help='Path to config.yaml')

    args = parser.parse_args()

    logger.info(f"Train file: {args.train}")
    logger.info(f"Val file: {args.val}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Config file: {args.config}")

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

    # Store original counts
    original_train_count = len(train_entries)
    original_val_count = len(val_entries)

    # Consolidate all entries
    logger.info("Consolidating train + val into single training set...")
    all_entries = train_entries + val_entries
    logger.info(f"Combined total: {len(all_entries)} entries")

    # Transform to AutoTrain flat text format
    logger.info("Transforming to AutoTrain flat text format...")
    transformed_entries = [transform_to_autotrain_format(e, config) for e in all_entries]

    # Compute statistics (before cleaning meta)
    logger.info("Computing statistics...")
    combined_stats = compute_statistics(transformed_entries, "combined")

    # Clean output (remove _meta field)
    logger.info("Cleaning output entries...")
    clean_entries = clean_output_entries(transformed_entries)

    # Write output file
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_output = output_dir / "hf_train_autotrain.jsonl"

    logger.info(f"Writing consolidated training set to {train_output}...")
    write_jsonl(clean_entries, train_output)
    logger.info(f"✓ Wrote {len(clean_entries)} entries")

    # Generate and write report
    report = format_statistics_report(combined_stats, original_train_count, original_val_count)

    # Write to log file
    log_dir = Path("work/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "hf_prep_autotrain.log"

    logger.info(f"Writing report to {log_file}...")
    with open(log_file, 'w') as f:
        f.write(report)

    # Print report to console
    print("\n" + report)

    logger.info("✅ Done! Next step: Generate synthetic validation with script 11")

    # Show sample output
    logger.info("\n📝 Sample output format:")
    if clean_entries:
        print(f"  {json.dumps(clean_entries[0])}")


if __name__ == "__main__":
    main()
