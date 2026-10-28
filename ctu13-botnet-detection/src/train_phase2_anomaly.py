"""
src/train_phase2_anomaly.py
Phase 2 — Anomaly detection trained on CTU-13 BENIGN traffic only.

Detectors: IsolationForest, OneClassSVM, LocalOutlierFactor (novelty).
Evaluated on: benign test, known botnet, held-out family (NsisAy).

This simulates a zero-day-style evaluation using CTU-13 data only.
The held-out family is treated as an "unseen threat" proxy.

Usage:
    python src/train_phase2_anomaly.py
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
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score, average_precision_score,
    precision_score, recall_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from config import PROJECT_ROOT, SPLITS_DIR, RANDOM_SEED, HOLDOUT_FAMILY
from logging_utils import get_logger
from preprocessing import UBFPreprocessor
from feature_engineering import extract_features
from ubf_features import get_ubf

logger = get_logger("train_phase2_anomaly",
                    log_file=PROJECT_ROOT / "reports" / "phase2" / "anomaly_train.log")

MODELS_DIR  = PROJECT_ROOT / "models" / "phase2"
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase2"
META_DIR    = PROJECT_ROOT / "models" / "metadata"

# Anomaly label: 1 = normal/benign, -1 = anomaly (sklearn convention)
SKLEARN_NORMAL    =  1
SKLEARN_ANOMALY   = -1

# Our evaluation labels: 0 = benign, 1 = botnet/anomaly
EVAL_BENIGN  = 0
EVAL_BOTNET  = 1


_CSV_DTYPES = {"binary_label": "int8", "scenario_id": "int16",
               "total_packets": "float32", "total_bytes": "float32",
               "src_bytes": "float32", "dst_bytes": "float32",
               "duration": "float32", "src_port": "float32", "dst_port": "float32"}


CHUNKSIZE = 500_000


def _load_split_safe(path: Path, max_benign: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Stream CSV in chunks: keep 100% of botnet flows and a representative benign subsample."""
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
    return df


def _load_data(splits_dir: Path):
    train   = _load_split_safe(splits_dir / "train.csv",   max_benign=100_000)
    test    = _load_split_safe(splits_dir / "test.csv",    max_benign=50_000)
    holdout_path = splits_dir / "holdout.csv"
    holdout = pd.read_csv(holdout_path, low_memory=False, dtype=_CSV_DTYPES) if holdout_path.exists() else pd.DataFrame()
    return train, test, holdout


def _extract_ubf(df: pd.DataFrame, pre: UBFPreprocessor, fit: bool = False) -> np.ndarray:
    feats = extract_features(df)
    ubf   = get_ubf(feats)
    if fit:
        return pre.fit_transform(ubf)
    return pre.transform(ubf)


def _get_detectors(seed: int, contamination: float = 0.05):
    return [
        ("isolation_forest", IsolationForest(
            n_estimators=200, contamination=contamination,
            random_state=seed, n_jobs=-1)),
        ("one_class_svm", OneClassSVM(
            kernel="rbf", nu=contamination, gamma="scale")),
        ("local_outlier_factor", LocalOutlierFactor(
            n_neighbors=20, contamination=contamination, novelty=True, n_jobs=-1)),
        ("elliptic_envelope", EllipticEnvelope(
            contamination=contamination, random_state=seed)),
    ]


def _evaluate(detector, X_benign_test, X_botnet, X_holdout,
              threshold: float | None = None) -> dict:
    """Evaluate detector on three groups."""

    def _predict(X):
        if hasattr(detector, "score_samples"):
            scores = detector.score_samples(X)
        elif hasattr(detector, "decision_function"):
            scores = detector.decision_function(X)
        else:
            scores = -detector.predict(X).astype(float)
        return scores

    scores_benign  = _predict(X_benign_test)
    scores_botnet  = _predict(X_botnet) if len(X_botnet) > 0 else np.array([])
    scores_holdout = _predict(X_holdout) if len(X_holdout) > 0 else np.array([])

    # sklearn raw predict: 1=normal, -1=anomaly → convert to 0=benign, 1=botnet
    raw_benign  = detector.predict(X_benign_test)
    pred_benign = (raw_benign == SKLEARN_ANOMALY).astype(int)

    results: dict[str, Any] = {
        "n_benign_test": len(X_benign_test),
        "n_botnet":      len(X_botnet),
        "n_holdout":     len(X_holdout),
        "fpr_on_benign": float(pred_benign.mean()),  # false positives
    }

    if len(X_botnet) > 0:
        raw_bot  = detector.predict(X_botnet)
        pred_bot = (raw_bot == SKLEARN_ANOMALY).astype(int)
        results["tpr_on_known_botnet"] = float(pred_bot.mean())

    if len(X_holdout) > 0:
        raw_hold  = detector.predict(X_holdout)
        pred_hold = (raw_hold == SKLEARN_ANOMALY).astype(int)
        results["tpr_on_holdout"] = float(pred_hold.mean())

        # Combine for ROC if possible
        if len(X_botnet) > 0:
            y_all = np.concatenate([
                np.zeros(len(X_benign_test)),
                np.ones(len(X_botnet)),
            ])
            s_all = np.concatenate([scores_benign, scores_botnet])
            try:
                results["roc_auc_known"] = float(roc_auc_score(y_all, -s_all))
            except Exception:
                pass

    return results


def train_anomaly(splits_dir: Path, out_dir: Path, seed: int = RANDOM_SEED) -> None:
    splits_dir = Path(splits_dir)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df, holdout_df = _load_data(splits_dir)

    # Benign-only training data
    train_benign = train_df[train_df["binary_label"] == 0]
    logger.info("Benign train rows: %d", len(train_benign))

    pre = UBFPreprocessor(scaler_type="robust", random_seed=seed)
    X_train_benign = _extract_ubf(train_benign, pre, fit=True)
    pre.save("anomaly", META_DIR)

    # Test sets
    test_benign = test_df[test_df["binary_label"] == 0]
    test_botnet = test_df[test_df["binary_label"] == 1]
    X_test_benign  = _extract_ubf(test_benign, pre)
    X_test_botnet  = _extract_ubf(test_botnet, pre) if len(test_botnet) > 0 else np.array([]).reshape(0, X_train_benign.shape[1])
    X_holdout      = _extract_ubf(holdout_df, pre) if len(holdout_df) > 0 else np.array([]).reshape(0, X_train_benign.shape[1])

    logger.info("Test benign=%d  botnet=%d  holdout(%s)=%d",
                len(X_test_benign), len(X_test_botnet), HOLDOUT_FAMILY, len(X_holdout))

    detectors   = _get_detectors(seed)
    all_results = []

    for name, detector in detectors:
        logger.info("Training %s …", name)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if name == "one_class_svm" and len(X_train_benign) > 50000:
                logger.info("  Subsampling benign train to 50,000 rows for One-Class SVM efficiency")
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(X_train_benign), 50000, replace=False)
                detector.fit(X_train_benign[idx])
            else:
                detector.fit(X_train_benign)
        train_time = time.time() - t0

        res = _evaluate(detector, X_test_benign, X_test_botnet, X_holdout)
        res["algorithm"]    = name
        res["train_time_s"] = round(train_time, 3)
        all_results.append(res)

        logger.info(
            "  %s  fpr=%.4f  tpr_botnet=%.4f  tpr_holdout=%.4f  roc=%.4f",
            name,
            res.get("fpr_on_benign", -1),
            res.get("tpr_on_known_botnet", -1),
            res.get("tpr_on_holdout", -1),
            res.get("roc_auc_known", -1),
        )

        # Save model
        joblib.dump(detector, MODELS_DIR / f"anomaly_{name}.joblib")

        # Score distribution plots
        _plot_score_distributions(detector, X_test_benign, X_test_botnet, X_holdout,
                                  name, out_dir / "score_distributions")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "anomaly_results_all_algorithms.csv", index=False)
    logger.info("Anomaly results → %s", out_dir / "anomaly_results_all_algorithms.csv")

    _write_phase2_summary(all_results, out_dir)


def _plot_score_distributions(detector, X_ben, X_bot, X_hold, name, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _scores(X):
            if hasattr(detector, "score_samples"):
                return detector.score_samples(X)
            elif hasattr(detector, "decision_function"):
                return detector.decision_function(X)
            return -detector.predict(X).astype(float)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(_scores(X_ben), bins=80, alpha=0.6, label="Benign",  color="#2ecc71", density=True)
        if len(X_bot) > 0:
            ax.hist(_scores(X_bot), bins=80, alpha=0.6, label="Known Botnet", color="#e74c3c", density=True)
        if len(X_hold) > 0:
            ax.hist(_scores(X_hold), bins=80, alpha=0.6, label=f"Holdout ({HOLDOUT_FAMILY})",
                    color="#f39c12", density=True)
        ax.set_xlabel("Anomaly Score", fontweight="bold")
        ax.set_ylabel("Density",       fontweight="bold")
        ax.set_title(f"Score Distribution — {name}", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_score_dist.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        logger.warning("Score dist plot failed for %s: %s", name, exc)


def _write_phase2_summary(results: list[dict], out_dir: Path) -> None:
    lines = [
        "# Phase 2 — Anomaly Detection Summary",
        "",
        f"**Holdout family (zero-day proxy):** {HOLDOUT_FAMILY}",
        "",
        "## Results",
        "",
        "| Algorithm | FPR on Benign | TPR Known Botnet | TPR Holdout | ROC-AUC |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['algorithm']} | "
            f"{r.get('fpr_on_benign', -1):.4f} | "
            f"{r.get('tpr_on_known_botnet', -1):.4f} | "
            f"{r.get('tpr_on_holdout', -1):.4f} | "
            f"{r.get('roc_auc_known', -1):.4f} |"
        )
    lines += [
        "",
        "## Limitations",
        "",
        "- Holdout family evaluation is a **proxy** for zero-day detection.",
        "- CTU-13 families may share behavioral patterns.",
        "- Results are not directly comparable to real-world zero-day detection rates.",
    ]
    (out_dir / "phase2_summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Phase 2 summary → %s", out_dir / "phase2_summary.md")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Phase 2 anomaly detectors.")
    parser.add_argument("--splits-dir", default=str(SPLITS_DIR))
    parser.add_argument("--output-dir", default=str(REPORTS_DIR))
    parser.add_argument("--seed",       type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)
    train_anomaly(Path(args.splits_dir), Path(args.output_dir), args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
