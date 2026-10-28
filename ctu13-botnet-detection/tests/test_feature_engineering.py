"""
tests/test_feature_engineering.py
Tests for feature extraction and UBF layer.
"""
import numpy as np
import pandas as pd
import pytest

from feature_engineering import extract_features
from ubf_features import get_ubf
from config import UBF_FEATURES


def test_feature_extraction_output_columns(synthetic_df):
    feats = extract_features(synthetic_df)
    for col in ["duration", "bytes_per_second", "packets_per_second",
                "bytes_ratio", "mean_packet_size",
                "proto_tcp", "proto_udp", "proto_icmp",
                "state_SF", "state_S0", "state_REJ"]:
        assert col in feats.columns, f"Missing: {col}"


def test_no_inf_or_nan_after_extraction(synthetic_df):
    feats = extract_features(synthetic_df)
    assert not feats.isin([np.inf, -np.inf]).any().any()
    assert not feats.isnull().any().any()


def test_ubf_column_order(synthetic_df):
    feats = extract_features(synthetic_df)
    ubf   = get_ubf(feats)
    assert list(ubf.columns) == UBF_FEATURES


def test_ubf_no_label_column(synthetic_df):
    feats = extract_features(synthetic_df)
    ubf   = get_ubf(feats)
    for forbidden in ["label", "family", "binary_label", "scenario_id"]:
        assert forbidden not in ubf.columns


def test_safe_division_zero_duration():
    """Zero-duration flows should not produce inf."""
    df = pd.DataFrame([{
        "duration": 0.0, "src_bytes": 100.0, "total_bytes": 200.0,
        "total_packets": 5, "proto": "tcp", "state": "SF",
        "label": "benign", "binary_label": 0, "family": "Benign",
        "scenario_id": "1",
    }])
    feats = extract_features(df)
    assert not feats.isin([np.inf, -np.inf]).any().any()
    assert not feats.isnull().any().any()
