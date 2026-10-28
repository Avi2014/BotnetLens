# Phase 2 — Anomaly Detection Summary

**Holdout family (zero-day proxy):** NsisAy

## Results

| Algorithm | FPR on Benign | TPR Known Botnet | TPR Holdout | ROC-AUC |
|---|---|---|---|---|
| isolation_forest | 0.0393 | 0.0925 | 0.1481 | 0.5901 |
| one_class_svm | 0.1444 | 0.1153 | 0.0996 | 0.4458 |
| local_outlier_factor | 0.0500 | 0.0616 | 0.3589 | 0.5919 |
| elliptic_envelope | 0.0494 | 0.0864 | 0.0936 | 0.6032 |

## Limitations

- Holdout family evaluation is a **proxy** for zero-day detection.
- CTU-13 families may share behavioral patterns.
- Results are not directly comparable to real-world zero-day detection rates.