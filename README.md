# BotnetLens 🔍

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: CTU-13](https://img.shields.io/badge/Dataset-CTU--13-orange.svg)](https://www.stratosphereips.org/datasets-ctu13)

**BotnetLens** is an open-source machine learning framework for detecting and classifying botnet network traffic using a two-phase cascade pipeline:
1. **Phase 1 (Supervised Multi-Class Classifier)**: Detects known botnet families (*Neris*, *Rbot*, *Virut*, *Menti*, *Sogou*, *Murlo*).
2. **Phase 2 (Unsupervised Anomaly Detector)**: Flags novel behaviors and zero-day variants (evaluated with held-out *NsisAy*).
3. **Cascade Inference Engine**: Intelligently routes network flows between Phase 1 and Phase 2 based on prediction confidence.

---

## 📂 Repository Structure

The core research pipeline and application are located in [`ctu13-botnet-detection/`](ctu13-botnet-detection/):

```
BotnetLens/
├── ctu13-botnet-detection/   # Main project directory
│   ├── app/                  # Web dashboard / inference API
│   ├── config/               # Model & pipeline configurations
│   ├── data/                 # Dataset folder (raw/interim/processed/splits)
│   ├── examples/             # Sample JSON/CSV flows for quick testing
│   ├── models/               # Model checkpoints & preprocessors
│   ├── reports/              # Visualizations, confusion matrices, logs
│   ├── src/                  # End-to-end Python source code
│   ├── tests/                # Automated unit & integration tests
│   ├── QUICK_START.md        # Step-by-step setup guide
│   └── README.md             # Detailed documentation & benchmarks
├── .gitignore                # Global git exclusions (data, models, venvs)
└── README.md                 # Project landing page
```

---

## ⚡ Quick Start

```bash
# 1. Navigate to the project directory
cd ctu13-botnet-detection

# 2. Set up virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify environment
python scripts/validate_environment.py

# 5. Run automated tests (uses built-in synthetic fixtures, no heavy data required)
pytest -v

# 6. Run inference on a sample flow
python src/cascade_inference.py --input examples/sample_flow.json
```

---

## 💾 Handling Datasets and Pre-trained Models

To keep the repository fast to clone and compliant with GitHub file limits (<100 MB):
- **Raw CTU-13 Dataset**: Excluded from Git. Download it directly from the [Stratosphere IPS CTU-13 Repository](https://www.stratosphereips.org/datasets-ctu13) and place into `ctu13-botnet-detection/data/raw/CTU-13-Dataset.tar/`.
- **Large Splits & Trained Model Weights (`*.joblib`)**: Excluded via `.gitignore`. You can train models locally in minutes using the dev mode (`--max-rows 50000`), or download pre-trained weights from the repository's [Releases](../../releases) tab.

For in-depth architecture details, feature engineering definitions, and full reproduction commands, see the **[Detailed Documentation](ctu13-botnet-detection/README.md)**.

---

## 📄 License

This project is licensed under the [MIT License](ctu13-botnet-detection/LICENSE).
