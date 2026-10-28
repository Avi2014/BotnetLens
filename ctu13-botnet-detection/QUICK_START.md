# Quick Start Guide

## Step 1 — Environment Setup

```powershell
# Navigate into the project folder
cd ctu13-botnet-detection

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt

# Verify setup
python scripts\validate_environment.py
```

## Step 2 — Place the Dataset

Your CTU-13 data is already extracted at:
```
data\raw\CTU-13-Dataset.tar\
    1\  capture20110810.binetflow   (Neris)
    2\  capture20110811.binetflow   (Neris)
    3\  ...                         (Rbot)
    ...
    13\ ...                         (Virut)
```

## Step 3 — Run Full Pipeline

```powershell
python src\run_full_pipeline.py --strategy family_holdout
```

This runs everything end-to-end (~30-90 min depending on hardware):
- Extract → Clean → Feature Analysis → Split → Phase 1 → Phase 2 → Reports

## Step 4 — View Reports

After completion:
```
reports/phase1/binary_results_all_algorithms.csv    ← binary metrics
reports/phase1/multiclass_results_all_algorithms.csv ← family metrics
reports/phase2/phase2_summary.md                     ← anomaly results
reports/family_analysis/family_feature_profiles.md   ← family signatures
reports/figures/                                      ← all charts
```

## Step 5 — Run Inference

```powershell
# Single flow
python src\cascade_inference.py --input examples\sample_flow.json

# Batch
python src\cascade_inference.py --input examples\sample_batch.csv --output predictions.csv
```

## Step 6 — Run Tests

```powershell
pytest -q
```

---

## Dev Mode (fast, limited rows)

```powershell
python src\run_full_pipeline.py --strategy family_holdout --max-rows 50000
```

This limits each scenario to 50,000 rows for quick testing.
