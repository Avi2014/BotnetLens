# BotnetLens — CTU-13 Two-Phase Botnet Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dataset: CTU-13](https://img.shields.io/badge/Dataset-CTU--13-orange.svg)](https://www.stratosphereips.org/datasets-ctu13)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A complete, reproducible, two-phase botnet detection pipeline built on the **CTU-13 benchmark dataset**. It combines supervised machine learning for known botnet family identification with unsupervised anomaly detection for zero-day / novel behavior discovery.

---

## 📌 Table of Contents
- [What This Project Does](#-what-this-project-does)
- [System Architecture](#-system-architecture)
- [CTU-13 Dataset & Families](#-ctu-13-dataset--families)
- [Unified Behavioral Features (UBF)](#-unified-behavioral-features-ubf)
- [Dataset Acquisition & Setup](#-dataset-acquisition--setup)
- [Handling Large Files & Models](#-handling-large-files--models)
- [Quick Start](#-quick-start)
- [Pipeline Execution](#-pipeline-execution)
- [Cascade Decision Logic](#-cascade-decision-logic)
- [Model Evaluation & Metrics](#-model-evaluation--metrics)
- [Running Unit Tests](#-running-unit-tests)
- [Directory Layout](#-directory-layout)
- [Ethical Use & Limitations](#-ethical-use--limitations)

---

## 🎯 What This Project Does

Traditional intrusion detection systems (IDS) struggle with unknown botnet variants, while purely unsupervised systems suffer from high false alarm rates. **BotnetLens** bridges this gap using a **two-phase cascade decision engine**:

1. **Phase 1 — Known Botnet Detection (Supervised)**:
   - Identifies malicious vs. benign flows.
   - Categorizes flows into specific botnet families (*Neris*, *Rbot*, *Virut*, *Menti*, *Sogou*, *Murlo*) using multi-class classifiers.
2. **Phase 2 — Novel Behavior Detection (Unsupervised / Anomaly)**:
   - Trained strictly on **benign-only traffic**.
   - Inspects flows where Phase 1 is uncertain or passes as benign to flag previously unseen or zero-day behavior (benchmarked against held-out *NsisAy* family).
3. **Cascade Decision Engine**:
   - Synthesizes Phase 1 and Phase 2 outputs to produce definitive flow classifications: `BENIGN`, `KNOWN_BOTNET`, `NOVEL_BEHAVIOR`, or `UNCERTAIN`.

---

## 🏗 System Architecture

```
Raw CTU-13 NetFlow / binetflow
              │
              ▼
Phase 0: Extraction, Schema & Label Normalization
              │
              ▼
Phase 0: UBF Extraction (16 Behavioral Features)
              │
              ▼
Phase 0: Scenario-Aware & Family-Holdout Splitting
              │
      ┌───────┴───────┐
      ▼               ▼
Phase 1: Supervised  Phase 2: Anomaly Detection
Binary & Multi-Class (Trained strictly on benign)
      └───────┬───────┘
              ▼
Cascade Inference Engine
  ├── High Confidence Botnet ──► KNOWN_BOTNET (Family identified)
  ├── Uncertain / Borderline ──► Evaluated by Phase 2
  │                                ├── Anomaly  ──► NOVEL_BEHAVIOR
  │                                └── Normal   ──► BENIGN
  └── Low Confidence Ambiguity ──► UNCERTAIN
              │
              ▼
Performance Visualizations & Evaluation Reports
```

---

## 📊 CTU-13 Dataset & Families

The [CTU-13 dataset](https://www.stratosphereips.org/datasets-ctu13) (captured at Czech Technical University in 2011) provides labeled NetFlow (`binetflow`) captures spanning 13 scenarios and 7 botnet families:

| Scenario(s) | Family | Malware Type | Role in Pipeline |
| :--- | :--- | :--- | :--- |
| **1, 2, 9** | **Neris** | IRC botnet / spammer | Supervised Training & Test |
| **3, 4, 10, 11** | **Rbot** | IRC botnet / scanner | Supervised Training & Test |
| **5, 13** | **Virut** | Polymorphic worm | Supervised Training & Test |
| **6** | **Menti** | HTTP botnet | Supervised Training & Test |
| **7** | **Sogou** | Download bot | Supervised Training & Test |
| **8** | **Murlo** | Spambot / reconnaissance | Supervised Training & Test |
| **12** | **NsisAy** | Trojan / Downloader | **HELD OUT** (Zero-day evaluation proxy) |

> 🔒 **Zero-Data-Leakage Guarantee**: Family **NsisAy (scenario 12)** is strictly excluded from Phase 1 training. It is evaluated only as unseen traffic to test zero-day detection capability in Phase 2.

---

## 🔬 Unified Behavioral Features (UBF)

The pipeline derives **16 network behavioral features** from raw flow records. To prevent synthetic shortcuts and ensure model generalization, **IP addresses and port identifiers are never used directly as input features**.

| # | Feature Name | Category | Description |
|---|:---|:---|:---|
| 1 | `duration` | Basic | Flow duration in seconds |
| 2 | `src_bytes` | Basic | Bytes transferred by source host |
| 3 | `dst_bytes` | Basic | Bytes transferred by destination host |
| 4 | `total_bytes` | Basic | Aggregate bidirectional bytes |
| 5 | `total_packets` | Basic | Aggregate bidirectional packets |
| 6 | `bytes_per_second` | Rate | Throughput rate (bytes/s) |
| 7 | `packets_per_second` | Rate | Packet rate (packets/s) |
| 8 | `bytes_ratio` | Ratio | Flow byte ratio (src_bytes / total_bytes) |
| 9 | `mean_packet_size` | Packet Size | Average size per packet across flow |
| 10 | `src_mean_packet_size`| Packet Size | Average packet size from source |
| 11 | `proto_tcp` | Protocol | One-hot indicator: TCP traffic |
| 12 | `proto_udp` | Protocol | One-hot indicator: UDP traffic |
| 13 | `proto_icmp` | Protocol | One-hot indicator: ICMP traffic |
| 14 | `state_SF` | State | Established and cleanly closed connection |
| 15 | `state_S0` | State | Connection attempted, no reply seen (SYN sent) |
| 16 | `state_REJ` | State | Connection rejected |

---

## 💾 Dataset Acquisition & Setup

Because raw CTU-13 files are several gigabytes in size, they are **not committed to the Git repository** (following standard open-source ML practices).

### Download Steps:
1. Download the CTU-13 dataset archive from the official Stratosphere IPS laboratory:
   - [Stratosphere IPS CTU-13 Official Repository](https://www.stratosphereips.org/datasets-ctu13)
2. Extract the scenario folders into `data/raw/CTU-13-Dataset.tar/`:
   ```
   ctu13-botnet-detection/
   └── data/
       └── raw/
           └── CTU-13-Dataset.tar/
               ├── 1/  capture20110810.binetflow
               ├── 2/  capture20110811.binetflow
               ├── ...
               └── 13/ capture20110815.binetflow
   ```

---

## 📦 Handling Large Files & Models

* **Ignored Data Artifacts**: Large dataset splits (`train.csv` ~2.6 GB, `test.csv` ~1.7 GB) and raw archives are git-ignored.
* **Pre-trained Model Binaries**:
  - Pre-trained models (`*.joblib`) exceed GitHub's standard commit limit and are hosted via [GitHub Releases](https://github.com/) under the tagged releases.
  - Alternatively, you can train all models locally in just a few minutes using the sample / dev commands below.
* **Included Fixtures**: The repository includes lightweight, zero-dependency test fixtures in `examples/` (`sample_batch.csv`, `sample_flow.json`) and `tests/fixtures/` so you can verify the entire codebase immediately upon cloning.

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/your-username/BotnetLens.git
cd BotnetLens/ctu13-botnet-detection

# Create and activate a virtual environment
python -m venv .venv
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify environment
python scripts/validate_environment.py
```

### 2. Fast Dev Mode (Train on Subset in < 3 minutes)
```bash
python src/run_full_pipeline.py --strategy family_holdout --max-rows 50000
```

### 3. Run Inference Using the Cascade Engine
```bash
# Single flow inference (JSON input)
python src/cascade_inference.py --input examples/sample_flow.json

# Batch inference (CSV input)
python src/cascade_inference.py --input examples/sample_batch.csv --output predictions.csv
```

---

## 🔄 Pipeline Execution

Run individual pipeline steps modularly as needed:

```bash
# 1. Extraction and cleaning
python src/extract_ctu13.py
python src/clean_ctu13.py

# 2. Family feature profile analysis
python src/family_feature_analysis.py

# 3. Create scenario-aware train/validation/test splits
python src/split_data.py --strategy family_holdout --holdout-family NsisAy

# 4. Supervised Phase 1 training (Binary & Multi-class)
python src/train_phase1_binary.py
python src/train_phase1_multiclass.py

# 5. Unsupervised Phase 2 training (Isolation Forest, LOF, OCSVM)
python src/train_phase2_anomaly.py

# 6. Generate comparative figures and summary reports
python src/generate_visualizations.py
```

---

## 🧠 Cascade Decision Logic

```
Flow Input
    │
    ▼
[Phase 1: Binary Classifier (Random Forest / LightGBM)]
    ├── Botnet Confidence ≥ 0.70 ──► KNOWN_BOTNET
    │                                  └─ Multiclass identifies family
    └── Benign or Confidence < 0.70
            │
            ▼
    [Phase 2: Isolation Forest Anomaly Detector]
            ├── Outlier Score ≥ Threshold ──► NOVEL_BEHAVIOR (Zero-day alert)
            └── Inlier Score  < Threshold ──► BENIGN
```

Output classification tags:
* `BENIGN`: Verified regular traffic.
* `KNOWN_BOTNET`: Matches behavioral signatures of a trained family.
* `NOVEL_BEHAVIOR`: Flagged by anomaly detection (novel pattern or zero-day variant).
* `UNCERTAIN`: Borderline metrics requiring human analyst inspection.

---

## 📈 Model Evaluation & Benchmarks

The framework evaluates multiple baseline and ensemble algorithms:

### Phase 1: Supervised Family Classifiers
- **Random Forest** (High accuracy and robustness against noise)
- **LightGBM** (High-speed inference on large NetFlow batches)
- **XGBoost** (Gradient boosted baseline)
- **MLP / Neural Network**
- **Linear SVM & Logistic Regression** (Linear baselines)

### Phase 2: Unsupervised Anomaly Detectors
- **Isolation Forest** (Primary production detector; robust against high dimensionality)
- **One-Class SVM**
- **Local Outlier Factor (LOF)** (Novelty detection mode)
- **Elliptic Envelope**

Detailed CSV tables, confusion matrices, and ROC curves are automatically generated under `reports/`.

---

## 🧪 Running Unit Tests

Run the automated test suite without downloading any external datasets:

```bash
pytest -v
```

Tests validate:
* Flow schema parsing and column renaming
* Label sanitization and regex extraction
* Feature computation mathematics (rates, ratios, mean sizes)
* Train/test split leakage guards
* Cascade logic routing and edge cases

---

## 📁 Directory Layout

```
ctu13-botnet-detection/
├── app/                     # Web dashboard / inference API service
├── config/                  # Pipeline YAML configs & feature schemas
├── data/                    # Dataset directories (.gitkeep preserved)
│   ├── raw/                 # Put CTU-13 raw files here
│   ├── interim/             # Cleaned intermediate tables
│   ├── processed/           # Feature-engineered parquet/csv
│   └── splits/              # Train/test/holdout splits
├── examples/                # Quick sample flows for inference testing
├── models/                  # Trained model serialization directory
│   ├── metadata/            # Encoders and scaler metadata
│   ├── phase1/              # Phase 1 classifier weights
│   └── phase2/              # Phase 2 anomaly detector weights
├── notebooks/               # Exploratory data analysis notebooks
├── reports/                 # Evaluation figures, logs, and summaries
│   ├── dataset/             # Data quality & label mapping reports
│   ├── figures/             # Comparative performance charts
│   ├── phase1/              # Binary & multiclass metric CSVs
│   └── phase2/              # Anomaly detection summaries
├── scripts/                 # Environment validation and utility tools
├── src/                     # Core pipeline source code
├── tests/                   # Pytest test suite with synthetic fixtures
├── .gitignore               # Strict exclusion rules for datasets/models
├── requirements.txt         # Pinned Python package dependencies
└── QUICK_START.md           # Quick setup reference guide
```

---

## ⚠️ Ethical Use & Limitations

1. **Benchmark Age**: CTU-13 was recorded in 2011 on x86 PC architectures. It is a benchmark for behavioral modeling, not a drop-in defense against modern IoT botnets (e.g., Mirai).
2. **Zero-Day Proxy**: Using held-out families (e.g., *NsisAy*) provides a realistic experimental benchmark, but real-world zero-day threats may exhibit unmodeled evasion techniques.
3. **Research Purpose**: This software is released for academic, research, and defensive evaluation purposes only.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
