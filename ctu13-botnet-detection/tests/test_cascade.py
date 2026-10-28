"""
tests/test_cascade.py
Tests for cascade routing logic using synthetic data.
"""
import numpy as np
import pandas as pd
import pytest


def _make_mock_engine():
    """Build a minimal CascadeEngine with mock models."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    from cascade_inference import CascadeEngine, FINAL_CATEGORIES

    class _MockBinary:
        def predict(self, X):           return np.array([1])
        def predict_proba(self, X):     return np.array([[0.05, 0.95]])

    class _MockBinaryBenign:
        def predict(self, X):           return np.array([0])
        def predict_proba(self, X):     return np.array([[0.95, 0.05]])

    class _MockMulticlass:
        def predict(self, X):           return np.array([2])  # index of some family
        def predict_proba(self, X):     return np.array([[0.0, 0.0, 0.9, 0.1]])

    class _MockLE:
        classes_ = ["Benign", "Menti", "Neris", "Rbot"]
        def inverse_transform(self, x): return [self.classes_[i] for i in x]

    class _MockAnomaly:
        def predict(self, X):               return np.array([-1])  # anomaly
        def score_samples(self, X):         return np.array([-0.5])

    class _MockAnomalyNormal:
        def predict(self, X):               return np.array([1])   # normal
        def score_samples(self, X):         return np.array([0.3])

    class _MockPre:
        feature_names = []
        _fitted = True
        def transform(self, X):
            import numpy as np
            return np.zeros((len(X), 16))

    return CascadeEngine, _MockBinary, _MockBinaryBenign, _MockMulticlass, _MockLE, _MockAnomaly, _MockAnomalyNormal, _MockPre, FINAL_CATEGORIES


def test_known_botnet_high_confidence():
    CascadeEngine, MockBin, _, MockMC, MockLE, MockAnom, _, MockPre, CATS = _make_mock_engine()
    engine = CascadeEngine.__new__(CascadeEngine)
    engine._binary_model     = MockBin()
    engine._multiclass_model = MockMC()
    engine._le               = MockLE()
    engine._anomaly_model    = MockAnom()
    engine._pre_binary       = MockPre()
    engine._pre_multiclass   = MockPre()
    engine._pre_anomaly      = MockPre()
    engine.confidence_threshold = 0.70
    engine._loaded = True

    flow = {"duration": 1.0, "src_bytes": 500.0, "total_bytes": 800.0,
            "total_packets": 10, "proto": "tcp", "state": "SF"}
    result = engine.predict_one(flow)
    assert result["final_category"] == CATS["KNOWN_BOTNET"]
    assert result["phase2_executed"] is False


def test_benign_flow_stays_benign():
    CascadeEngine, _, MockBenign, MockMC, MockLE, _, MockANormal, MockPre, CATS = _make_mock_engine()
    engine = CascadeEngine.__new__(CascadeEngine)
    engine._binary_model     = MockBenign()
    engine._multiclass_model = MockMC()
    engine._le               = MockLE()
    engine._anomaly_model    = MockANormal()
    engine._pre_binary       = MockPre()
    engine._pre_multiclass   = MockPre()
    engine._pre_anomaly      = MockPre()
    engine.confidence_threshold = 0.70
    engine._loaded = True

    flow = {"duration": 0.5, "src_bytes": 100.0, "total_bytes": 200.0,
            "total_packets": 5, "proto": "udp", "state": "SF"}
    result = engine.predict_one(flow)
    assert result["final_category"] == CATS["BENIGN"]
    assert result["phase2_executed"] is True


def test_novel_behavior_detected():
    CascadeEngine, _, MockBenign, MockMC, MockLE, MockAnom, _, MockPre, CATS = _make_mock_engine()
    engine = CascadeEngine.__new__(CascadeEngine)
    engine._binary_model     = MockBenign()
    engine._multiclass_model = MockMC()
    engine._le               = MockLE()
    engine._anomaly_model    = MockAnom()
    engine._pre_binary       = MockPre()
    engine._pre_multiclass   = MockPre()
    engine._pre_anomaly      = MockPre()
    engine.confidence_threshold = 0.70
    engine._loaded = True

    flow = {"duration": 0.01, "src_bytes": 50000.0, "total_bytes": 50000.0,
            "total_packets": 1000, "proto": "udp", "state": "S0"}
    result = engine.predict_one(flow)
    assert result["final_category"] == CATS["NOVEL_BEHAVIOR"]
    assert result["phase2_executed"] is True
