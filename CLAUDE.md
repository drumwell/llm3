# CLAUDE.md — Project brief for Claude Code

## Goal
Convert scanned service-manual pages (images across many sections) into instruction-tuning JSONL suitable for LoRA/QLoRA finetuning. Output JSONL tasks: `spec`, `procedure`, `troubleshooting`, `explanation`, `wiring`.

## Deliverables (data-prep only)
- `work/` artifact folders: `inventory.csv`, `images_clean/`, `ocr_raw/`, `ocr_tables/`, `blocks/`, `logs/`
- `data/` JSONL files (train/val) per task
- `work/logs/qa_report.md` with counts + validations

## Repo map (create if missing)
/data_src/ # your forked manual (images by section)
/work/ # generated artifacts
/data/ # final JSONL
/scripts/ # pipeline scripts
/config.yaml # rules, units, regex fixes
/Makefile # end-to-end targets
/CLAUDE.md # this file


## Tools & commands Claude should know
> Claude: run `--help` for any tool before first use. Keep commands explicit. Log outputs to `work/logs/`.

- Python: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- OCR: PaddleOCR (text + table)  
- Image ops: OpenCV / Pillow
- Make: orchestrate pipeline (`make all`)
- Git: commit after each milestone (`git add -A && git commit -m "milestone: <name>"`)

## Safety & permissions
- Start **read-only**, then enable writes when ready.
- Never run destructive commands (`rm -rf`, mass renames) without a plan + backup.
- Keep diffs **small**; prefer many small PRs/commits over one huge one.
- On uncertainty: stop, print a plan, and ask for confirmation.

## Working style
1) **Plan → Execute → Verify → Commit** (tight loop).  
2) Prefer **idempotent** scripts (safe re-runs).  
3) Add **acceptance checks** before moving on.  
4) Summarize context when the session is long (Claude can “compress context”). :contentReference[oaicite:1]{index=1}

## Acceptance checks (data-prep)
- `inventory.csv` has every image once; no empties.
- `images_clean/` has 1:1 processed PNGs (deskewed, legible).
- `ocr_raw/*.json` non-empty for non-diagram pages.
- `ocr_tables/*.csv` exists where tables detected.
- `blocks/*.json` exist; fields populated by task.
- `data/*.jsonl` exist; **spec outputs match value-only regex**, procedures/troubleshooting are **numbered**.
- `work/logs/qa_report.md` shows counts, 0 critical failures.

## References
- Anthropic: *Claude Code – Best practices for agentic coding* (tool docs in `CLAUDE.md`, run `--help`, incremental diffs, explicit goals). :contentReference[oaicite:2]{index=2}

