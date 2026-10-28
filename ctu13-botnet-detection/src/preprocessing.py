"""
src/preprocessing.py
Scikit-learn preprocessing pipeline: scaler + imputer, fitted only on training data.
Persists all preprocessing objects with metadata.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from config import PROJECT_ROOT, RANDOM_SEED, UBF_FEATURES
from logging_utils import get_logger

logger = get_logger("preprocessing")

MODELS_META_DIR = PROJECT_ROOT / "models" / "metadata"
SCALER_MAP = {
    "standard": StandardScaler,
    "robust":   RobustScaler,
    "minmax":   MinMaxScaler,
}


class UBFPreprocessor:
    """Impute → Scale for UBF features.  Fitted on train only."""

    def __init__(self, scaler_type: str = "robust", random_seed: int = RANDOM_SEED):
        self.scaler_type  = scaler_type
        self.random_seed  = random_seed
        self.feature_names: list[str] = UBF_FEATURES
        self.imputer: SimpleImputer | None = None
        self.scaler:  Any               = None
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> "UBFPreprocessor":
        cols = [c for c in self.feature_names if c in X.columns]
        if len(cols) < len(self.feature_names):
            logger.warning("Missing UBF cols at fit: %s", set(self.feature_names) - set(cols))
        X_arr = X[cols].values.astype(np.float64)

        self.imputer = SimpleImputer(strategy="median")
        X_arr = self.imputer.fit_transform(X_arr)

        ScalerClass = SCALER_MAP.get(self.scaler_type, RobustScaler)
        self.scaler = ScalerClass()
        self.scaler.fit(X_arr)

        self._fitted = True
        logger.info("UBFPreprocessor fitted on %d samples, %d features.", *X_arr.shape)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        cols = self.feature_names
        X_out = pd.DataFrame(index=X.index)
        for col in cols:
            if col in X.columns:
                X_out[col] = X[col]
            else:
                X_out[col] = 0.0
        X_arr = X_out[cols].values.astype(np.float64)
        X_arr = self.imputer.transform(X_arr)
        X_arr = self.scaler.transform(X_arr)
        return X_arr

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, name: str, out_dir: Path | None = None) -> Path:
        out_dir = Path(out_dir) if out_dir else MODELS_META_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}_preprocessor.joblib"
        joblib.dump(self, path)
        meta = {
            "name":          name,
            "scaler_type":   self.scaler_type,
            "feature_names": self.feature_names,
            "n_features":    len(self.feature_names),
            "fitted":        self._fitted,
        }
        with open(out_dir / f"{name}_preprocessor_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Preprocessor saved → %s", path)
        return path

    @classmethod
    def load(cls, name: str, in_dir: Path | None = None) -> "UBFPreprocessor":
        in_dir = Path(in_dir) if in_dir else MODELS_META_DIR
        path   = in_dir / f"{name}_preprocessor.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {path}")
        obj = joblib.load(path)
        logger.info("Preprocessor loaded ← %s", path)
        return obj
