"""
tests/test_labels.py
Tests for label normalisation.
"""
import pytest

from label_normalization import normalise_single, normalise_labels
import pandas as pd


def test_background_is_benign():
    r = normalise_single("flow=Background-Established-cmpgw", "1")
    assert r["binary_label"] == 0
    assert r["family"] == "Benign"


def test_botnet_label_resolved():
    r = normalise_single("flow=From-Botnet-IRC", "1")
    assert r["binary_label"] == 1
    assert r["family"] == "Neris"   # scenario 1 → Neris


def test_unknown_scenario_gives_unknown_family():
    r = normalise_single("flow=From-Botnet-IRC", "99")
    assert r["binary_label"] == 1
    assert r["family"] == "Unknown"


def test_unrecognised_label():
    r = normalise_single("some-weird-label", "1")
    assert r["binary_label"] == -1
    assert r["status"] == "unresolved"


def test_normalise_labels_dataframe():
    df = pd.DataFrame({
        "label":       ["flow=Background-TCP", "flow=From-Botnet", "flow=From-Botnet"],
        "scenario_id": ["1", "3", "5"],
    })
    result = normalise_labels(df)
    assert list(result["binary_label"]) == [0, 1, 1]
    assert result.loc[1, "family"] == "Rbot"
    assert result.loc[2, "family"] == "Virut"
