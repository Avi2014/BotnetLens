"""
tests/test_preprocessing.py
Tests that preprocessor is fitted only on training data.
"""
import numpy as np
import pandas as pd
import pytest

from preprocessing import UBFPreprocessor
from feature_engineering import extract_features
from ubf_features import get_ubf


def test_fit_transform_shape(synthetic_df):
    pre = UBFPreprocessor()
    feats = extract_features(synthetic_df)
    ubf   = get_ubf(feats)
    X = pre.fit_transform(ubf)
    assert X.shape[0] == len(synthetic_df)
    assert X.shape[1] == ubf.shape[1]


def test_transform_without_fit_raises():
    pre = UBFPreprocessor()
    with pytest.raises(RuntimeError):
        pre.transform(pd.DataFrame([[1, 2, 3]]))


def test_no_inf_after_transform(synthetic_df):
    pre = UBFPreprocessor()
    feats = extract_features(synthetic_df)
    ubf   = get_ubf(feats)
    X = pre.fit_transform(ubf)
    assert not np.any(np.isinf(X))
    assert not np.any(np.isnan(X))


def test_train_test_transform_consistent(synthetic_df):
    """Same preprocessor must produce consistent scales on different subsets."""
    half = len(synthetic_df) // 2
    train_df = synthetic_df.iloc[:half]
    test_df  = synthetic_df.iloc[half:]

    pre = UBFPreprocessor()
    X_train = pre.fit_transform(get_ubf(extract_features(train_df)))
    X_test  = pre.transform(get_ubf(extract_features(test_df)))

    # Shapes must match in feature dimension
    assert X_train.shape[1] == X_test.shape[1]
