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

all: inventory preprocess ocr blocks emit validate split
