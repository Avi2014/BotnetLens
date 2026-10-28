"""
tests/test_schema.py
Tests for CTU-13 column normalisation.
"""
import pandas as pd
import pytest

from ctu13_schema import normalise_columns, check_required_columns


def test_normalise_known_variants():
    df = pd.DataFrame(columns=["StartTime", "Dur", "Proto", "SrcAddr",
                                "TotPkts", "TotBytes", "SrcBytes", "Label"])
    result = normalise_columns(df)
    assert "start_time"     in result.columns
    assert "duration"       in result.columns
    assert "proto"          in result.columns
    assert "src_addr"       in result.columns
    assert "total_packets"  in result.columns
    assert "total_bytes"    in result.columns
    assert "src_bytes"      in result.columns
    assert "label"          in result.columns


def test_normalise_lowercase_variants():
    df = pd.DataFrame(columns=["dur", "totbytes", "srcbytes", "label"])
    result = normalise_columns(df)
    assert "duration"    in result.columns
    assert "total_bytes" in result.columns
    assert "src_bytes"   in result.columns
    assert "label"       in result.columns


def test_check_required_columns_pass():
    df = pd.DataFrame(columns=["duration", "total_bytes", "src_bytes", "label"])
    assert check_required_columns(df) == []


def test_check_required_columns_fail():
    df = pd.DataFrame(columns=["duration", "proto"])
    missing = check_required_columns(df)
    assert "label" in missing
