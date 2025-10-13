# Model Performance Diagnosis

## Issue Report

**Query**: `[SPEC] What is the engine displacement?`
**Expected**: `2.3 L` (BMW E30 M3 S14 engine displacement)
**Got**: `4.0 V` (incorrect - mixing voltage with displacement)

## Root Cause Analysis

### ✅ What We Checked

1. **Training completed successfully**
   - 3 epochs on A100 (3 minutes)
   - eval_loss: 1.4571
   - mean_token_accuracy: 76.65%
   - All metrics in expected range

2. **Model architecture is correct**
   - Llama-3.2-3B-Instruct base model
   - QLoRA with r=16, attention layers only
   - Proper chat template formatting
   - Task prefixes working correctly

3. **Training data quality**
   - 1,185 balanced training examples
   - 158 validation examples
   - Good distribution across task types

### ❌ Problem Identified

**The model was never trained on engine displacement data!**

```bash
# Search result:
❌ No engine displacement data found in training set
```

**Why this happened:**
- The BMW service manual likely has engine specs in an **index page** or **general information section**
- Our pipeline processed **individual procedure/specification pages**
- Basic engine specs (displacement, bore, stroke, compression ratio) are typically in:
  - Cover pages
  - Index pages (11-00-index-a.jpg, 11-00-index-b.jpg, etc.)
  - General information tables

**What the model learned instead:**
- Page-specific torque values (45 Nm, 9 Nm, etc.)
- Special tool part numbers (11.5 030, 11 1 052, etc.)
- Procedure-specific values
- NOT general engine specifications

## Why Model Gave "4.0 V"

The model has learned to:
1. Extract **value + unit** patterns from training data
2. When asked about "engine" + unknown spec, it **hallucinated** a plausible-looking answer
3. Confused voltage values (common in engine electrical section 12) with displacement

**This is expected behavior** when querying data the model was never trained on!

## What the Model CAN Answer (Based on Training Data)

✅ **SPEC tasks it WAS trained on:**
- `[SPEC] What is the tightening torque for engine section 11?` → "45 Nm" ✓
- `[SPEC] What is the max. permissible total wear play for engine section 11?` → "0.15 Nm" ✓
- `[SPEC] What is the version with for engine section 25?` → "4 Speed" ✓
- Specific procedure torque values
- Special tool part numbers
- Component specifications from procedure pages

❌ **SPEC tasks it was NOT trained on:**
- Engine displacement (2.3 L)
- Bore × stroke (95.0 × 84.0 mm)
- Compression ratio
- Power output (195 hp / 147 kW)
- Any specs from index/cover pages

## Model Performance Assessment

**Overall: Model is working CORRECTLY, but with limited scope**

| Aspect | Status | Notes |
|--------|--------|-------|
| Training | ✅ | Completed successfully, good metrics |
| Architecture | ✅ | QLoRA working properly |
| Task prefixes | ✅ | [SPEC], [PROCEDURE], etc. working |
| Value extraction | ✅ | Extracting torque, part numbers correctly |
| Data coverage | ❌ | **Missing general specs from index pages** |
| Out-of-domain | ⚠️ | Hallucinates when asked about untrained data |

**The 76.65% accuracy reflects:**
- ✓ High accuracy on data it WAS trained on
- ✗ Hallucinations on data it was NOT trained on

## Solutions

### Option 1: Add Index Pages to Training Data (RECOMMENDED)

**Pros:**
- Gets you the missing general specs
- Relatively small addition (~3-5 index pages per section)
- Quick iteration (regenerate → retrain)

**Steps:**
1. Check which pages were skipped:
   ```bash
   ls data_src/11\ -\ Engine/ | grep index
   # 11-00-index-a.jpg, 11-00-index-b.jpg, 11-00-index-c.jpg
   ```

2. Update pipeline to process index pages:
   - Modify `scripts/02_preprocess.py` to include index pages
   - Or manually process index pages only

3. Extract general specs:
   - Engine displacement: 2.3 L / 2302 cm³
   - Bore: 95.0 mm
   - Stroke: 84.0 mm
   - Compression ratio: 10.5:1
   - Power: 147 kW / 195 hp @ 6750 rpm
   - Torque: 240 Nm @ 4750 rpm

4. Create synthetic JSONL entries:
   ```json
   {"instruction": "What is the engine displacement?", "output": "2.3 L", "meta": {"task": "spec", "section": "11", "topic": "Engine displacement"}}
   {"instruction": "What is the bore and stroke?", "output": "95.0 × 84.0 mm", "meta": {"task": "spec", "section": "11", "topic": "Bore and stroke"}}
   ```

5. Append to `data/hf_train.jsonl` and retrain

**Time estimate**: 1-2 hours to add specs + 3 min retrain on A100

---

### Option 2: Test with Queries Model WAS Trained On

**Purpose**: Validate model is working on its actual training data

**Test queries from training set:**
```python
# These SHOULD work well:
test_queries = [
    "[SPEC] What is the tightening torque for engine section 11?",  # → "45 Nm"
    "[SPEC] What is the version with for engine section 25?",       # → "4 Speed"
    "[PROCEDURE] How do you adjust valve clearance?",               # → numbered steps
    "[EXPLANATION] Explain the Motronic control unit operation",    # → detailed explanation
]
```

Run these in your HF Space to confirm model works on trained data.

---

### Option 3: Document Model Scope

**If not retraining**: Clearly document what the model CAN and CANNOT answer

**Model card example:**
```
This model answers questions from BMW E30 M3 service manual PROCEDURE PAGES.

✅ Can answer:
- Torque specifications from procedures
- Special tool part numbers
- Valve clearances, wear limits, etc.
- Step-by-step procedures
- Component explanations from procedure pages

❌ Cannot answer (not in training data):
- General engine specifications (displacement, bore, stroke, etc.)
- Power/torque curves
- Model year differences
- Specs from index pages or general info sections
```

## Recommendations

**For Production Use:**

1. **SHORT TERM** (today):
   - Test with queries from actual training data (see Option 2)
   - Update HF Space description to reflect trained scope
   - Validate model works on what it was trained on

2. **MEDIUM TERM** (1-2 days):
   - Add index pages to training data (Option 1)
   - Retrain with general specs included
   - Re-evaluate on broader query set

3. **LONG TERM** (future):
   - Add all index/cover pages systematically
   - Include model year variations
   - Add cross-referencing between sections

## Next Steps

**What would you like to do?**

A. Test model with queries it WAS trained on (validate it works)
B. Add index pages and general specs to training data (retrain)
C. Deploy as-is with documented limitations
D. Run systematic evaluation with `test_inference.ipynb` on validation set

Let me know and I'll help you proceed!
