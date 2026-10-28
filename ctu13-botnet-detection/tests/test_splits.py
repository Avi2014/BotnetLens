"""
tests/test_splits.py
Tests for data splitting: reproducibility, no leakage, scenario separation.
"""
import numpy as np
import pandas as pd
import pytest

from split_data import random_split, scenario_aware_split, family_holdout_split
from config import HOLDOUT_FAMILY


def test_random_split_no_overlap(synthetic_df):
    train, val, test = random_split(synthetic_df)
    total = len(train) + len(val) + len(test)
    assert total == len(synthetic_df), "Rows lost in split"

    train_idx = set(train.index)
    val_idx   = set(val.index)
    test_idx  = set(test.index)
    assert len(train_idx & val_idx)  == 0, "Train/val overlap"
    assert len(train_idx & test_idx) == 0, "Train/test overlap"
    assert len(val_idx   & test_idx) == 0, "Val/test overlap"


def test_random_split_reproducible(synthetic_df):
    t1, v1, te1 = random_split(synthetic_df, seed=42)
    t2, v2, te2 = random_split(synthetic_df, seed=42)
    assert list(t1.index) == list(t2.index)


def test_family_holdout_excludes_holdout(synthetic_df):
    train, val, test, holdout = family_holdout_split(
        synthetic_df, holdout_family=HOLDOUT_FAMILY)
    # Holdout family must NOT appear in train/val/test
    for split_name, split in [("train", train), ("val", val), ("test", test)]:
        families_in_split = set(split["family"].unique())
        assert HOLDOUT_FAMILY not in families_in_split, \
            f"{HOLDOUT_FAMILY} found in {split_name}"
    # Holdout df must only contain the holdout family (botnet rows)
    assert set(holdout["family"].unique()) <= {HOLDOUT_FAMILY}


def test_scenario_aware_no_scenario_overlap(synthetic_df):
    train, val, test = scenario_aware_split(synthetic_df)
    train_scens = set(train["scenario_id"].unique())
    val_scens   = set(val  ["scenario_id"].unique())
    test_scens  = set(test ["scenario_id"].unique())
    assert len(train_scens & test_scens) == 0, "Scenario in both train and test"
