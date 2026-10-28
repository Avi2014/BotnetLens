"""
src/cascade_inference.py
Two-phase cascade inference engine.

Flow:
  Input flow/features
      → Phase 1 binary: is it Botnet?
        → if yes + confidence >= threshold: KNOWN_BOTNET + family
        → if no or low confidence:
            → Phase 2 anomaly detector:
                → if anomalous: NOVEL_BEHAVIOR
                → else:         BENIGN

Output fields:
  timestamp, phase1_binary_prediction, phase1_binary_confidence,
  phase1_family_prediction, phase1_family_confidence, phase1_decision,
  phase2_executed, phase2_prediction, phase2_anomaly_score, phase2_threshold,
  final_label, final_category, explanation

Usage:
    python src/cascade_inference.py --input examples/sample_flow.json
    python src/cascade_inference.py --input examples/sample_batch.csv --output predictions.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import PROJECT_ROOT, SPLITS_DIR, UBF_FEATURES, HOLDOUT_FAMILY, RANDOM_SEED
from logging_utils import get_logger
from preprocessing import UBFPreprocessor
from feature_engineering import extract_features
from ubf_features import get_ubf

logger = get_logger("cascade_inference")

MODELS_PHASE1 = PROJECT_ROOT / "models" / "phase1"
MODELS_PHASE2 = PROJECT_ROOT / "models" / "phase2"
META_DIR      = PROJECT_ROOT / "models" / "metadata"

FINAL_CATEGORIES = {
    "BENIGN":        "BENIGN",
    "KNOWN_BOTNET":  "KNOWN_BOTNET",
    "NOVEL_BEHAVIOR":"NOVEL_BEHAVIOR",
    "UNCERTAIN":     "UNCERTAIN",
    "INVALID_INPUT": "INVALID_INPUT",
}

DEFAULT_CONFIDENCE_THRESHOLD = 0.70
DEFAULT_ANOMALY_THRESHOLD    = 0.0  # score below this = anomaly (IF convention)


class CascadeEngine:
    """Load models and run two-phase inference."""

    def __init__(
        self,
        binary_model_name:     str = "random_forest",
        multiclass_model_name: str = "random_forest",
        anomaly_model_name:    str = "isolation_forest",
        confidence_threshold:  float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self.confidence_threshold = confidence_threshold
        self._binary_model     = None
        self._multiclass_model = None
        self._anomaly_model    = None
        self._le               = None
        self._pre_binary       = None
        self._pre_multiclass   = None
        self._pre_anomaly      = None
        self._binary_name      = binary_model_name
        self._mc_name          = multiclass_model_name
        self._anomaly_name     = anomaly_model_name
        self._loaded           = False

    def load(self) -> "CascadeEngine":
        """Load all models and preprocessors."""
        # Binary
        b_path = MODELS_PHASE1 / f"binary_{self._binary_name}.joblib"
        if not b_path.exists():
            raise FileNotFoundError(f"Binary model not found: {b_path}")
        self._binary_model = joblib.load(b_path)

        # Multiclass
        mc_path = MODELS_PHASE1 / f"multiclass_{self._mc_name}.joblib"
        if mc_path.exists():
            self._multiclass_model = joblib.load(mc_path)
        else:
            logger.warning("Multiclass model not found: %s", mc_path)

        # Label encoder
        le_path = MODELS_PHASE1 / "multiclass_label_encoder.joblib"
        if le_path.exists():
            self._le = joblib.load(le_path)

        # Anomaly
        a_path = MODELS_PHASE2 / f"anomaly_{self._anomaly_name}.joblib"
        if a_path.exists():
            self._anomaly_model = joblib.load(a_path)
        else:
            logger.warning("Anomaly model not found: %s", a_path)

        # Preprocessors
        self._pre_binary     = UBFPreprocessor.load("binary",     META_DIR)
        self._pre_multiclass = UBFPreprocessor.load("multiclass", META_DIR)
        if (META_DIR / "anomaly_preprocessor.joblib").exists():
            self._pre_anomaly = UBFPreprocessor.load("anomaly", META_DIR)

        self._loaded = True
        logger.info("CascadeEngine loaded successfully.")
        return self

    def _to_ubf(self, flow_dict: dict) -> np.ndarray:
        """Convert a flow dict or row to UBF feature array using binary preprocessor."""
        df = pd.DataFrame([flow_dict])
        feats = extract_features(df)
        ubf   = get_ubf(feats)
        return self._pre_binary.transform(ubf)

    def predict_one(self, flow: dict) -> dict:
        """Run cascade on a single flow dict. Returns result dict."""
        if not self._loaded:
            return self._invalid("Model not loaded. Call load() first.")

        try:
            X = self._to_ubf(flow)
        except Exception as exc:
            return self._invalid(f"Feature extraction failed: {exc}")

        result: dict[str, Any] = {
            "timestamp":                 datetime.now(timezone.utc).isoformat(),
            "phase1_binary_prediction":  None,
            "phase1_binary_confidence":  None,
            "phase1_family_prediction":  None,
            "phase1_family_confidence":  None,
            "phase1_decision":           None,
            "phase2_executed":           False,
            "phase2_prediction":         None,
            "phase2_anomaly_score":      None,
            "phase2_threshold":          DEFAULT_ANOMALY_THRESHOLD,
            "final_label":               None,
            "final_category":            None,
            "explanation":               "",
        }

        # ── Phase 1: Binary ───────────────────────────────────────────────────
        bin_pred = int(self._binary_model.predict(X)[0])
        bin_conf = 0.5
        if hasattr(self._binary_model, "predict_proba"):
            bin_conf = float(self._binary_model.predict_proba(X)[0, bin_pred])

        result["phase1_binary_prediction"] = "Botnet" if bin_pred == 1 else "Benign"
        result["phase1_binary_confidence"] = round(bin_conf, 4)

        if bin_pred == 1 and bin_conf >= self.confidence_threshold:
            # Known botnet — run multiclass family identification
            result["phase1_decision"] = "BOTNET_HIGH_CONFIDENCE"
            family_pred, family_conf = "Unknown", 0.0

            if self._multiclass_model is not None and self._le is not None:
                try:
                    X_mc = self._pre_multiclass.transform(
                        get_ubf(extract_features(pd.DataFrame([flow])))
                    )
                    mc_pred = int(self._multiclass_model.predict(X_mc)[0])
                    family_pred = self._le.inverse_transform([mc_pred])[0]
                    if hasattr(self._multiclass_model, "predict_proba"):
                        family_conf = float(self._multiclass_model.predict_proba(X_mc)[0, mc_pred])
                except Exception as exc:
                    logger.warning("Multiclass failed: %s", exc)

            result["phase1_family_prediction"] = family_pred
            result["phase1_family_confidence"] = round(family_conf, 4)
            result["final_label"]    = "Botnet"
            result["final_category"] = FINAL_CATEGORIES["KNOWN_BOTNET"]
            result["explanation"]    = (
                f"Phase 1 detected known botnet (conf={bin_conf:.4f} >= {self.confidence_threshold}). "
                f"Family: {family_pred} (conf={family_conf:.4f})."
            )
            return result

        # Low confidence or benign → Phase 2
        result["phase1_decision"] = "BENIGN_OR_LOW_CONFIDENCE"
        result["phase2_executed"] = True

        if self._anomaly_model is None:
            result["final_label"]    = "Benign"
            result["final_category"] = FINAL_CATEGORIES["UNCERTAIN"]
            result["explanation"]    = "Phase 1 low/benign; Phase 2 model unavailable."
            return result

        X_a = X
        if self._pre_anomaly is not None:
            X_a = self._pre_anomaly.transform(
                get_ubf(extract_features(pd.DataFrame([flow])))
            )

        a_pred = int(self._anomaly_model.predict(X_a)[0])  # 1=normal, -1=anomaly
        a_score = 0.0
        if hasattr(self._anomaly_model, "score_samples"):
            a_score = float(self._anomaly_model.score_samples(X_a)[0])
        elif hasattr(self._anomaly_model, "decision_function"):
            a_score = float(self._anomaly_model.decision_function(X_a)[0])

        result["phase2_prediction"]    = "Anomaly" if a_pred == -1 else "Normal"
        result["phase2_anomaly_score"] = round(a_score, 6)

        if a_pred == -1:
            result["final_label"]    = "Suspicious"
            result["final_category"] = FINAL_CATEGORIES["NOVEL_BEHAVIOR"]
            result["explanation"]    = (
                f"Phase 1 inconclusive (conf={bin_conf:.4f}). "
                f"Phase 2 flagged as anomalous (score={a_score:.4f})."
            )
        else:
            result["final_label"]    = "Benign"
            result["final_category"] = FINAL_CATEGORIES["BENIGN"]
            result["explanation"]    = (
                f"Phase 1: benign/low-confidence. Phase 2: normal (score={a_score:.4f})."
            )
        return result

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorised cascade inference on a full DataFrame.

        Runs feature extraction and model inference as batch numpy operations
        instead of row-by-row, making it ~100-1000x faster than predict_one in a loop.
        """
        if not self._loaded:
            return pd.DataFrame([self._invalid("Model not loaded.")] * len(df))
        if len(df) == 0:
            return pd.DataFrame()

        now_ts = datetime.now(timezone.utc).isoformat()

        # ── Feature extraction (once for all rows) ─────────────────────────
        feats = extract_features(df)
        ubf   = get_ubf(feats)

        # Binary preprocessor
        X_bin = self._pre_binary.transform(ubf)

        # Multiclass preprocessor
        X_mc = self._pre_multiclass.transform(ubf) if self._pre_multiclass else X_bin

        # Anomaly preprocessor
        X_ano = self._pre_anomaly.transform(ubf) if self._pre_anomaly else X_bin

        n = len(df)

        # ── Phase 1: Binary (batch) ────────────────────────────────────────
        bin_preds = self._binary_model.predict(X_bin).astype(int)          # shape (n,)
        bin_confs = np.full(n, 0.5)
        if hasattr(self._binary_model, "predict_proba"):
            proba     = self._binary_model.predict_proba(X_bin)             # (n, 2)
            bin_confs = proba[np.arange(n), bin_preds]

        # High-confidence botnet mask
        hc_mask = (bin_preds == 1) & (bin_confs >= self.confidence_threshold)  # (n,)
        p2_mask = ~hc_mask                                                       # goes to Phase 2

        # ── Phase 1: Multiclass (batch, only HC rows) ──────────────────────
        family_preds = np.full(n, "Unknown", dtype=object)
        family_confs = np.zeros(n)

        if hc_mask.any() and self._multiclass_model is not None and self._le is not None:
            try:
                mc_raw = self._multiclass_model.predict(X_mc[hc_mask])
                family_preds[hc_mask] = self._le.inverse_transform(mc_raw)
                if hasattr(self._multiclass_model, "predict_proba"):
                    mc_proba = self._multiclass_model.predict_proba(X_mc[hc_mask])
                    family_confs[hc_mask] = mc_proba[np.arange(hc_mask.sum()), mc_raw]
            except Exception as exc:
                logger.warning("Batch multiclass failed: %s", exc)

        # ── Phase 2: Anomaly (batch, only P2 rows) ─────────────────────────
        ano_preds  = np.ones(n, dtype=int)     # default: normal (1)
        ano_scores = np.zeros(n)

        if p2_mask.any() and self._anomaly_model is not None:
            try:
                ano_preds[p2_mask]  = self._anomaly_model.predict(X_ano[p2_mask])
                if hasattr(self._anomaly_model, "score_samples"):
                    ano_scores[p2_mask] = self._anomaly_model.score_samples(X_ano[p2_mask])
                elif hasattr(self._anomaly_model, "decision_function"):
                    ano_scores[p2_mask] = self._anomaly_model.decision_function(X_ano[p2_mask])
            except Exception as exc:
                logger.warning("Batch anomaly failed: %s", exc)

        # ── Assemble results ───────────────────────────────────────────────
        final_categories = np.full(n, "BENIGN", dtype=object)
        final_labels     = np.full(n, "Benign", dtype=object)

        # High-confidence botnet rows
        final_categories[hc_mask] = "KNOWN_BOTNET"
        final_labels[hc_mask]     = "Botnet"

        # Phase 2 anomaly rows
        novel_mask = p2_mask & (ano_preds == -1)
        final_categories[novel_mask] = "NOVEL_BEHAVIOR"
        final_labels[novel_mask]     = "Suspicious"

        results = {
            "timestamp":                 now_ts,
            "phase1_binary_prediction":  np.where(bin_preds == 1, "Botnet", "Benign"),
            "phase1_binary_confidence":  np.round(bin_confs, 4),
            "phase1_family_prediction":  family_preds,
            "phase1_family_confidence":  np.round(family_confs, 4),
            "phase1_decision":           np.where(hc_mask, "BOTNET_HIGH_CONFIDENCE",
                                                  "BENIGN_OR_LOW_CONFIDENCE"),
            "phase2_executed":           p2_mask,
            "phase2_prediction":         np.where(ano_preds == -1, "Anomaly", "Normal"),
            "phase2_anomaly_score":      np.round(ano_scores, 6),
            "phase2_threshold":          DEFAULT_ANOMALY_THRESHOLD,
            "final_label":               final_labels,
            "final_category":            final_categories,
        }
        return pd.DataFrame(results)

    @staticmethod
    def _invalid(reason: str) -> dict:
        return {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "final_label": "INVALID",
            "final_category": FINAL_CATEGORIES["INVALID_INPUT"],
            "explanation": reason,
        }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Two-phase cascade inference.")
    parser.add_argument("--input",   required=True,
                        help="JSON file (single flow) or CSV file (batch)")
    parser.add_argument("--output",  default=None,
                        help="Output CSV path (batch mode only)")
    parser.add_argument("--binary-model",     default="random_forest")
    parser.add_argument("--multiclass-model", default="random_forest")
    parser.add_argument("--anomaly-model",    default="isolation_forest")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    args = parser.parse_args(argv)

    engine = CascadeEngine(
        binary_model_name=args.binary_model,
        multiclass_model_name=args.multiclass_model,
        anomaly_model_name=args.anomaly_model,
        confidence_threshold=args.confidence_threshold,
    )
    try:
        engine.load()
    except FileNotFoundError as exc:
        logger.error("Model loading failed: %s\nRun run_full_pipeline.py first.", exc)
        return 1

    input_path = Path(args.input)
    if input_path.suffix.lower() == ".json":
        with open(input_path) as f:
            flow = json.load(f)
        result = engine.predict_one(flow)
        print(json.dumps(result, indent=2))
    elif input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path, low_memory=False)
        results_df = engine.predict_batch(df)
        if args.output:
            results_df.to_csv(args.output, index=False)
            logger.info("Predictions → %s", args.output)
        else:
            print(results_df.to_string())
    else:
        logger.error("Input must be .json or .csv")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
