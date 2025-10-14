SECT_FILTER?=

inventory:
	python scripts/01_inventory.py --data-src data_src --output work/inventory.csv --section-filter "$(SECT_FILTER)"

preprocess:
	python scripts/02_preprocess.py --inventory work/inventory.csv --out work/images_clean

ocr:
	python scripts/03_ocr.py --input-dir work/images_clean --output-dir work/ocr_raw
	python scripts/03b_ocr_tables.py --ocr-dir work/ocr_raw --images-dir work/images_clean --output-dir work/ocr_tables

blocks:
	python scripts/04_parse_blocks.py --ocr work/ocr_raw --tables work/ocr_tables --out work/blocks --config config.yaml

emit:
	python scripts/05_emit_jsonl.py --blocks-dir work/blocks --output-dir data

validate:
	python scripts/06_split_validate.py --data-dir data --output work/logs/qa_report.md

split:
	python scripts/07_make_splits.py --data-dir data --pattern "*.slice.jsonl" --train-split 0.8

hf_prep:
	python scripts/08_prepare_hf_dataset.py --train data/train.jsonl --val data/val.jsonl --output-dir data --config config.yaml --duplicate-weights "spec=1,explanation=2,procedure=7,wiring=10,troubleshooting=50"

extract_html:
	python scripts/07_extract_html_specs.py

autotrain_prep:
	@echo "🔄 Converting to AutoTrain format..."
	@python3 -c 'import json; from pathlib import Path; \
	train_path = Path("data/hf_train.jsonl"); \
	val_path = Path("data/hf_val.jsonl"); \
	output_path = Path("data/hf_train_autotrain.jsonl"); \
	combined = []; \
	[combined.append({"text": f"User: {json.loads(line)[\"messages\"][0][\"content\"]}\\nAssistant: {json.loads(line)[\"messages\"][1][\"content\"]}"}) for line in open(train_path) if line.strip()]; \
	train_count = len(combined); \
	[combined.append({"text": f"User: {json.loads(line)[\"messages\"][0][\"content\"]}\\nAssistant: {json.loads(line)[\"messages\"][1][\"content\"]}"}) for line in open(val_path) if line.strip()]; \
	output_path.write_text("\\n".join(json.dumps(e) for e in combined)); \
	print(f"✅ Wrote {len(combined)} examples ({train_count} train + {len(combined)-train_count} val)")'

synthetic_val:
	python scripts/11_generate_synthetic_validation.py --train data/hf_train_autotrain.jsonl --output data/hf_val_synthetic.jsonl --count 250

upload_hf:
	@echo "📤 Uploading dataset to HuggingFace Hub..."
	@echo "Usage: python scripts/09_upload_to_hf.py --repo drumwell/llm3"
	@echo ""
	@echo "First time setup:"
	@echo "  1. Install: pip install datasets huggingface_hub"
	@echo "  2. Login: huggingface-cli login"
	@echo "  3. Run: python scripts/09_upload_to_hf.py --repo drumwell/llm3"

all: inventory preprocess ocr blocks emit validate split hf_prep extract_html autotrain_prep synthetic_val
