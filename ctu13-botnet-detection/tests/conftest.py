"""
tests/conftest.py
Shared fixtures for the CTU-13 botnet detection test suite.
Uses synthetic CTU-13-like data — no real dataset needed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src importable from tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ── Synthetic CTU-13-like fixture ──────────────────────────────────────────────

SCENARIO_FAMILY_MAP = {
    "1": "Neris",  "2": "Neris",  "3": "Rbot",
    "5": "Virut",  "6": "Menti",  "7": "Sogou",
    "8": "Murlo",  "12": "NsisAy",
}


def _make_flow(scenario_id: str, label: str, family: str,
               binary: int, rng: np.random.Generator) -> dict:
    dur = max(0.0, rng.exponential(1.5))
    src = max(0.0, rng.exponential(500))
    tot = max(src, src + rng.exponential(200))
    pkts = max(1, int(rng.poisson(10)))
    proto_choices = ["tcp", "udp", "icmp"]
    state_choices = ["SF", "S0", "REJ", "RSTO", "SH"]
    return {
        "start_time":    "2011/08/10 09:46:59",
        "duration":      round(dur, 6),
        "src_bytes":     round(src, 0),
        "total_bytes":   round(tot, 0),
        "total_packets": pkts,
        "proto":         rng.choice(proto_choices),
        "state":         rng.choice(state_choices),
        "src_port":      int(rng.integers(1024, 65535)),
        "dst_port":      int(rng.integers(1, 1024)),
        "direction":     "->",
        "src_tos":       0,
        "dst_tos":       0,
        "label":         label,
        "binary_label":  binary,
        "family":        family,
        "scenario_id":   scenario_id,
        "source_file":   f"scenario_{scenario_id}.binetflow",
    }


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """~500-row synthetic CTU-13-like DataFrame."""
    rng = np.random.default_rng(42)
    rows = []

    # Benign
    for _ in range(150):
        rows.append(_make_flow("1", "flow=Background-Established", "Benign", 0, rng))

    # Known families
    for sid, fam in [("1", "Neris"), ("3", "Rbot"), ("5", "Virut"),
                     ("6", "Menti"), ("7", "Sogou"), ("8", "Murlo")]:
        for _ in range(40):
            rows.append(_make_flow(sid, "flow=From-Botnet", fam, 1, rng))

    # Holdout family
    for _ in range(40):
        rows.append(_make_flow("12", "flow=From-Botnet", "NsisAy", 1, rng))

    df = pd.DataFrame(rows)
    return df


@pytest.fixture(scope="session")
def feature_df(synthetic_df):
    """Features extracted from synthetic_df."""
    from feature_engineering import extract_features
    return extract_features(synthetic_df)


@pytest.fixture(scope="session")
def ubf_df(feature_df):
    """UBF columns extracted from feature_df."""
    from ubf_features import get_ubf
    return get_ubf(feature_df)
