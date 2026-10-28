"""
src/train_phase1_binary.py
Phase 1 — Binary classification: Benign (0) vs Botnet (1).

Trains and compares: RandomForest, XGBoost, LightGBM, LogisticRegression,
                     LinearSVM, MLP.
Saves best model + all metrics.

Usage:
    python src/train_phase1_binary.py
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
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score, average_precision_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from config import PROJECT_ROOT, SPLITS_DIR, RANDOM_SEED, UBF_FEATURES
from logging_utils import get_logger
from preprocessing import UBFPreprocessor
from feature_engineering import extract_features
from ubf_features import get_ubf

logger = get_logger("train_phase1_binary",
                    log_file=PROJECT_ROOT / "reports" / "phase1" / "binary_train.log")

MODELS_DIR  = PROJECT_ROOT / "models" / "phase1"
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase1"


_CSV_DTYPES = {"binary_label": "int8", "scenario_id": "int16",
               "total_packets": "float32", "total_bytes": "float32",
               "src_bytes": "float32", "dst_bytes": "float32",
               "duration": "float32", "src_port": "float32", "dst_port": "float32"}


CHUNKSIZE = 500_000


def _load_split_safe(path: Path, max_benign: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Stream CSV in chunks: keep 100% of botnet flows and a representative benign subsample to prevent OOM."""
    if not path.exists():
        return pd.DataFrame()
    logger.info("Loading %s (max_benign=%s) …", path.name, max_benign)
    chunks = []
    chunk_benign_limit = (max_benign // 25) if max_benign else None
    for i, chunk in enumerate(pd.read_csv(path, chunksize=CHUNKSIZE, dtype=_CSV_DTYPES, low_memory=False)):
        if "binary_label" in chunk.columns:
            mal = chunk[chunk["binary_label"] == 1]
            ben = chunk[chunk["binary_label"] == 0]
            if chunk_benign_limit and len(ben) > chunk_benign_limit:
                ben = ben.sample(n=chunk_benign_limit, random_state=seed + i)
            chunks.append(pd.concat([mal, ben], ignore_index=True))
        else:
            chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    if "binary_label" in df.columns and max_benign is not None:
        mal = df[df["binary_label"] == 1]
        ben = df[df["binary_label"] == 0]
        if len(ben) > max_benign:
            ben = ben.sample(n=max_benign, random_state=seed)
        df = pd.concat([mal, ben], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    logger.info("Loaded %s: %d rows (%d malware, %d benign)",
                path.name, len(df),
                int((df["binary_label"] == 1).sum()) if "binary_label" in df.columns else 0,
                int((df["binary_label"] == 0).sum()) if "binary_label" in df.columns else 0)
    return df


def _load_data(splits_dir: Path):
    train = _load_split_safe(splits_dir / "train.csv", max_benign=500_000)
    val   = _load_split_safe(splits_dir / "val.csv",   max_benign=100_000)
    test  = _load_split_safe(splits_dir / "test.csv",  max_benign=200_000)
    return train, val, test


def _prepare_xy(df: pd.DataFrame, preprocessor: UBFPreprocessor | None,
                fit: bool = False):
    feats = extract_features(df)
    ubf   = get_ubf(feats)
    y     = df["binary_label"].values.astype(int)
    if fit:
        X = preprocessor.fit_transform(ubf)
    else:
        X = preprocessor.transform(ubf)
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
            LinearSVC(max_iter=2000, class_weight="balanced", random_state=seed))),
        ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                              random_state=seed, early_stopping=True)),
    ]
    # Optional: XGBoost
    try:
        from xgboost import XGBClassifier
        algs.insert(1, ("xgboost", XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=1, random_state=seed, n_jobs=-1,
            eval_metric="logloss", verbosity=0)))
    except ImportError:
        logger.warning("xgboost not installed — skipped.")

    # Optional: LightGBM
    try:
        from lightgbm import LGBMClassifier
        algs.insert(2, ("lightgbm", LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1)))
    except ImportError:
        logger.warning("lightgbm not installed — skipped.")

    return algs


def _metrics(y_true, y_pred, y_prob=None) -> dict:
    m: dict[str, Any] = {
        "accuracy":           float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy":  float(balanced_accuracy_score(y_true, y_pred)),
        "precision":          float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":             float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":                 float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1":           float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1":        float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc":                float(matthews_corrcoef(y_true, y_pred)),
    }
    tn = int(confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()[0])
    fp = int(confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()[1])
    fn = int(confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()[2])
    tp = int(confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()[3])
    total_neg = tn + fp
    total_pos = fn + tp
    m["false_positive_rate"] = fp / max(total_neg, 1)
    m["false_negative_rate"] = fn / max(total_pos, 1)
    m["specificity"]         = tn / max(total_neg, 1)
    if y_prob is not None:
        try:
            m["roc_auc"]  = float(roc_auc_score(y_true, y_prob))
            m["pr_auc"]   = float(average_precision_score(y_true, y_prob))
        except Exception:
            pass
    return m


def train_binary(splits_dir: Path, out_dir: Path, seed: int = RANDOM_SEED) -> dict:
    splits_dir = Path(splits_dir)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = _load_data(splits_dir)
    logger.info("train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    pre = UBFPreprocessor(scaler_type="robust", random_seed=seed)
    X_train, y_train = _prepare_xy(train_df, pre, fit=True)
    X_val,   y_val   = _prepare_xy(val_df,   pre)
    X_test,  y_test  = _prepare_xy(test_df,  pre)
    pre.save("binary", MODELS_DIR.parent / "metadata")

    algorithms = _get_algorithms(seed)
    all_results = []
    best_name, best_score, best_model = None, -1.0, None

    for name, model in algorithms:
        logger.info("Training %s …", name)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = model.predict(X_test)
        infer_time = (time.time() - t0) / max(len(X_test), 1) * 1000  # ms/sample

        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]

        m = _metrics(y_test, y_pred, y_prob)
        m["algorithm"]   = name
        m["train_time_s"] = round(train_time, 3)
        m["infer_ms_per_sample"] = round(infer_time, 4)
        all_results.append(m)

        logger.info("  %s  acc=%.4f  macro_f1=%.4f  roc_auc=%.4f",
                    name, m["accuracy"], m["macro_f1"], m.get("roc_auc", -1))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        _save_confusion_matrix(cm, name, ["Benign", "Botnet"],
                               out_dir / "confusion_matrices")

        # Classification report
        report = classification_report(y_test, y_pred,
                                       target_names=["Benign", "Botnet"])
        (out_dir / "binary_classification_reports" / f"{name}.txt").write_text(
            report, encoding="utf-8")

        # Timing
        timing = {"algorithm": name, "train_time_s": train_time,
                  "infer_ms_per_sample": infer_time}
        with open(out_dir / "timing" / f"{name}_timing.json", "w") as f:
            json.dump(timing, f, indent=2)

        # Feature importance
        _save_feature_importance(model, name, out_dir / "feature_importance")

        # Save model
        model_path = MODELS_DIR / f"binary_{name}.joblib"
        joblib.dump(model, model_path)

        # Best model: prioritise macro_f1 + balanced_accuracy
        score = 0.5 * m["macro_f1"] + 0.5 * m["balanced_accuracy"]
        if score > best_score:
            best_score = score
            best_name  = name
            best_model = model

    # Save results CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "binary_results_all_algorithms.csv", index=False)
    logger.info("Binary results → %s", out_dir / "binary_results_all_algorithms.csv")

    # Save best model info
    best_info = {"best_algorithm": best_name, "selection_score": best_score,
                 "criterion": "0.5*macro_f1 + 0.5*balanced_accuracy"}
    with open(out_dir / "best_model_selection.json", "w") as f:
        json.dump(best_info, f, indent=2)
    logger.info("Best binary model: %s (score=%.4f)", best_name, best_score)

    return best_info


def _save_confusion_matrix(cm, name, labels, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_xlabel("Predicted", fontweight="bold")
        ax.set_ylabel("Actual",    fontweight="bold")
        ax.set_title(f"Binary CM — {name}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_cm.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        logger.warning("CM plot failed for %s: %s", name, exc)


def _save_feature_importance(model, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        elif hasattr(model, "estimator") and hasattr(model.estimator, "coef_"):
            coef = model.estimator.coef_
            imp  = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
        else:
            return
        fi_df = pd.DataFrame({"feature": UBF_FEATURES[:len(imp)], "importance": imp})
        fi_df = fi_df.sort_values("importance", ascending=False)
        fi_df.to_csv(out_dir / f"{name}_importance.csv", index=False)
    except Exception as exc:
        logger.debug("Feature importance failed for %s: %s", name, exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Phase 1 binary classifiers.")
    parser.add_argument("--splits-dir", default=str(SPLITS_DIR))
    parser.add_argument("--output-dir", default=str(REPORTS_DIR))
    parser.add_argument("--seed",       type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)

    train_binary(Path(args.splits_dir), Path(args.output_dir), args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
