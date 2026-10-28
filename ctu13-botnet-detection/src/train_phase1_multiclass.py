"""
src/train_phase1_multiclass.py
Phase 1 — Multiclass family classification.

Trains classifiers over: Benign + all known botnet families (excl. holdout).
Saves models, metrics, confusion matrices, and per-family reports.

Usage:
    python src/train_phase1_multiclass.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from config import PROJECT_ROOT, SPLITS_DIR, RANDOM_SEED, UBF_FEATURES, HOLDOUT_FAMILY
from logging_utils import get_logger
from preprocessing import UBFPreprocessor
from feature_engineering import extract_features
from ubf_features import get_ubf

logger = get_logger("train_phase1_multiclass",
                    log_file=PROJECT_ROOT / "reports" / "phase1" / "multiclass_train.log")

MODELS_DIR  = PROJECT_ROOT / "models" / "phase1"
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase1"


_CSV_DTYPES = {"binary_label": "int8", "scenario_id": "int16",
               "total_packets": "float32", "total_bytes": "float32",
               "src_bytes": "float32", "dst_bytes": "float32",
               "duration": "float32", "src_port": "float32", "dst_port": "float32"}


CHUNKSIZE = 500_000


def _load_split_safe(path: Path, max_benign: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Stream CSV in chunks: keep 100% of all botnet families and a balanced benign subsample to prevent OOM."""
    if not path.exists():
        return pd.DataFrame()
    logger.info("Loading %s (max_benign=%s) …", path.name, max_benign)
    chunks = []
    chunk_benign_limit = (max_benign // 25) if max_benign else None
    for i, chunk in enumerate(pd.read_csv(path, chunksize=CHUNKSIZE, dtype=_CSV_DTYPES, low_memory=False)):
        if "family" in chunk.columns:
            mal = chunk[chunk["family"] != "Benign"]
            ben = chunk[chunk["family"] == "Benign"]
            if chunk_benign_limit and len(ben) > chunk_benign_limit:
                ben = ben.sample(n=chunk_benign_limit, random_state=seed + i)
            chunks.append(pd.concat([mal, ben], ignore_index=True))
        else:
            chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    if "family" in df.columns and max_benign is not None:
        mal = df[df["family"] != "Benign"]
        ben = df[df["family"] == "Benign"]
        if len(ben) > max_benign:
            ben = ben.sample(n=max_benign, random_state=seed)
        df = pd.concat([mal, ben], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    logger.info("Loaded %s: %d rows (families: %s)",
                path.name, len(df),
                df["family"].value_counts().to_dict() if "family" in df.columns else "N/A")
    return df


def _load_data(splits_dir: Path):
    train = _load_split_safe(splits_dir / "train.csv", max_benign=300_000)
    val   = _load_split_safe(splits_dir / "val.csv",   max_benign=100_000)
    test  = _load_split_safe(splits_dir / "test.csv",  max_benign=100_000)
    return train, val, test


def _prepare_xy(df: pd.DataFrame, preprocessor: UBFPreprocessor,
                le: LabelEncoder, fit: bool = False):
    feats = extract_features(df)
    ubf   = get_ubf(feats)
    y_raw = df["family"].fillna("Unknown").astype(str)

    if fit:
        X = preprocessor.fit_transform(ubf)
        y = le.fit_transform(y_raw)
        return X, y
    else:
        X = preprocessor.transform(ubf)
        # Filter rows whose family label was not seen during training
        known_classes = set(le.classes_)
        mask = y_raw.isin(known_classes)
        unseen = y_raw[~mask].unique().tolist()
        if unseen:
            logger.warning("Filtering %d rows with unseen families at test time: %s",
                           (~mask).sum(), unseen)
        X   = X[mask.values]
        y   = le.transform(y_raw[mask])
        return X, y


def _get_algorithms(seed: int) -> list[tuple[str, Any]]:
    algs = [
        ("random_forest", RandomForestClassifier(
            n_estimators=200, max_depth=20, class_weight="balanced",
            random_state=seed, n_jobs=-1)),
        ("logistic_regression", CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, class_weight="balanced",
                               random_state=seed, n_jobs=-1))),
        ("linear_svm", CalibratedClassifierCV(
            LinearSVC(max_iter=3000, class_weight="balanced", random_state=seed))),
        ("mlp", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400,
                              random_state=seed, early_stopping=True)),
    ]
    try:
        from xgboost import XGBClassifier
        algs.insert(1, ("xgboost", XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=seed, n_jobs=-1, eval_metric="mlogloss", verbosity=0)))
    except ImportError:
        logger.warning("xgboost not installed — skipped.")

    try:
        from lightgbm import LGBMClassifier
        algs.insert(2, ("lightgbm", LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1)))
    except ImportError:
        logger.warning("lightgbm not installed — skipped.")

    return algs


def _metrics(y_true, y_pred, classes) -> dict:
    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1":       float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _save_confusion_matrix(cm, name, class_names, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(7, len(class_names) - 1)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel("Predicted", fontweight="bold")
        ax.set_ylabel("Actual",    fontweight="bold")
        ax.set_title(f"Multiclass CM — {name}", fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_multiclass_cm.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        logger.warning("CM plot failed for %s: %s", name, exc)


def train_multiclass(splits_dir: Path, out_dir: Path, seed: int = RANDOM_SEED) -> dict:
    splits_dir = Path(splits_dir)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = _load_data(splits_dir)

    # Exclude holdout family from train/val/test (in case scenario split didn't remove it)
    train_df = train_df[train_df["family"] != HOLDOUT_FAMILY]
    val_df   = val_df  [val_df  ["family"] != HOLDOUT_FAMILY]
    test_df  = test_df [test_df ["family"] != HOLDOUT_FAMILY]

    logger.info("Multiclass: train=%d val=%d test=%d (excl. %s)",
                len(train_df), len(val_df), len(test_df), HOLDOUT_FAMILY)

    pre = UBFPreprocessor(scaler_type="robust", random_seed=seed)
    le  = LabelEncoder()
    X_train, y_train = _prepare_xy(train_df, pre, le, fit=True)
    X_test,  y_test  = _prepare_xy(test_df,  pre, le)
    class_names = list(le.classes_)
    logger.info("Classes: %s", class_names)

    pre.save("multiclass", PROJECT_ROOT / "models" / "metadata")
    joblib.dump(le, MODELS_DIR / "multiclass_label_encoder.joblib")

    algorithms = _get_algorithms(seed)
    all_results = []
    best_name, best_score, best_model = None, -1.0, None

    for name, model in algorithms:
        logger.info("Training multiclass %s …", name)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_test)
        m = _metrics(y_test, y_pred, class_names)
        m["algorithm"]    = name
        m["train_time_s"] = round(train_time, 3)
        all_results.append(m)

        logger.info("  %s  acc=%.4f  macro_f1=%.4f", name, m["accuracy"], m["macro_f1"])

        class_indices = list(range(len(class_names)))
        cm = confusion_matrix(y_test, y_pred, labels=class_indices)
        _save_confusion_matrix(cm, name, class_names, out_dir / "confusion_matrices")

        report = classification_report(y_test, y_pred, labels=class_indices,
                                       target_names=class_names, zero_division=0)
        (out_dir / "multiclass_classification_reports" / f"{name}.txt").write_text(
            report, encoding="utf-8")

        joblib.dump(model, MODELS_DIR / f"multiclass_{name}.joblib")

        score = 0.5 * m["macro_f1"] + 0.5 * m["balanced_accuracy"]
        if score > best_score:
            best_score = score
            best_name  = name

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "multiclass_results_all_algorithms.csv", index=False)

    best_info = {"best_algorithm": best_name, "selection_score": best_score,
                 "classes": class_names, "holdout_family": HOLDOUT_FAMILY}
    with open(out_dir / "best_multiclass_model_selection.json", "w") as f:
        json.dump(best_info, f, indent=2)

    logger.info("Best multiclass model: %s (score=%.4f)", best_name, best_score)
    return best_info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Phase 1 multiclass classifiers.")
    parser.add_argument("--splits-dir", default=str(SPLITS_DIR))
    parser.add_argument("--output-dir", default=str(REPORTS_DIR))
    parser.add_argument("--seed",       type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)
    train_multiclass(Path(args.splits_dir), Path(args.output_dir), args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
