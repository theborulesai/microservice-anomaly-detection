"""
evaluator.py
============
Evaluation engine:
  - Train/validation/test splitting (normal-only training → unsupervised)
  - Per-detector threshold selection via validation contamination
  - Full metrics: Precision, Recall, F1, AUC-ROC, AUC-PR, Detection Latency
  - Per-fault-type breakdown
  - Results stored as structured DataFrames
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import time
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── Splits ──────────────────────────────────────────

def make_splits(X: np.ndarray, y: np.ndarray,
                train_ratio: float = 0.6,
                val_ratio:   float = 0.2,
                seed: int = 42):
    """
    Unsupervised protocol:
      Train  : normal samples only (labels=0)
      Val    : mix of normal + anomalies  (for threshold selection)
      Test   : mix of normal + anomalies  (for final evaluation)
    """
    normal_idx  = np.where(y == 0)[0]
    anomaly_idx = np.where(y != 0)[0]

    # Split normal into train / val / test
    n_train = int(len(normal_idx) * train_ratio)
    n_val   = int(len(normal_idx) * val_ratio)
    rng     = np.random.default_rng(seed)
    perm    = rng.permutation(len(normal_idx))

    train_idx = normal_idx[perm[:n_train]]
    val_norm  = normal_idx[perm[n_train:n_train+n_val]]
    test_norm = normal_idx[perm[n_train+n_val:]]

    # Split anomalies between val and test
    a_perm  = rng.permutation(len(anomaly_idx))
    n_a_val = int(len(anomaly_idx) * val_ratio)
    val_anom  = anomaly_idx[a_perm[:n_a_val]]
    test_anom = anomaly_idx[a_perm[n_a_val:]]

    val_idx  = np.concatenate([val_norm,  val_anom])
    test_idx = np.concatenate([test_norm, test_anom])

    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (
        X[train_idx], y[train_idx],
        X[val_idx],   y[val_idx],
        X[test_idx],  y[test_idx],
    )


# ─────────────────────────── Metrics ─────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    scores: np.ndarray,
                    detection_latency_ms: float = 0.0) -> Dict:
    """Compute full metric suite for binary anomaly detection."""
    # Binary labels (0=normal, 1=anomaly)
    y_bin = (y_true != 0).astype(int)

    prec  = precision_score(y_bin, y_pred, zero_division=0)
    rec   = recall_score(y_bin, y_pred, zero_division=0)
    f1    = f1_score(y_bin, y_pred, zero_division=0)
    fpr   = np.sum((y_pred == 1) & (y_bin == 0)) / max(np.sum(y_bin == 0), 1)
    fnr   = np.sum((y_pred == 0) & (y_bin == 1)) / max(np.sum(y_bin == 1), 1)

    try:
        auc_roc = roc_auc_score(y_bin, scores)
    except Exception:
        auc_roc = float("nan")

    try:
        auc_pr = average_precision_score(y_bin, scores)
    except Exception:
        auc_pr = float("nan")

    return {
        "Precision":        round(prec,  4),
        "Recall":           round(rec,   4),
        "F1-Score":         round(f1,    4),
        "AUC-ROC":          round(auc_roc, 4),
        "AUC-PR":           round(auc_pr,  4),
        "FPR":              round(fpr,   4),
        "FNR":              round(fnr,   4),
        "Detection_Lat_ms": round(detection_latency_ms, 2),
    }


def compute_per_fault_metrics(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               scores: np.ndarray,
                               label_names: Dict) -> pd.DataFrame:
    """Per-fault-type precision, recall, F1."""
    rows = []
    unique_labels = np.unique(y_true)
    for lbl in unique_labels:
        if lbl == 0:
            continue
        # Binary problem: this fault vs normal
        mask = (y_true == 0) | (y_true == lbl)
        yt   = (y_true[mask] != 0).astype(int)
        yp   = y_pred[mask]
        sc   = scores[mask]
        prec = precision_score(yt, yp, zero_division=0)
        rec  = recall_score(yt, yp, zero_division=0)
        f1   = f1_score(yt, yp, zero_division=0)
        try:
            auc  = roc_auc_score(yt, sc)
        except Exception:
            auc = float("nan")
        rows.append({
            "Fault_Type": label_names.get(lbl, str(lbl)),
            "Precision":  round(prec, 4),
            "Recall":     round(rec,  4),
            "F1-Score":   round(f1,   4),
            "AUC-ROC":    round(auc,  4),
        })
    return pd.DataFrame(rows)


# ─────────────────────────── Runner ──────────────────────────────────────────

def evaluate_detector(detector, X_train, X_val, y_val,
                      X_test, y_test,
                      contamination: float = 0.1,
                      label_names: Dict = None) -> Tuple[Dict, pd.DataFrame]:
    """
    Full evaluation pipeline for one detector:
      1. Train on X_train (normal only)
      2. Select threshold on X_val
      3. Evaluate on X_test
    Returns (overall_metrics_dict, per_fault_df)
    """
    if label_names is None:
        label_names = {}

    # ─ Train ─
    t0 = time.perf_counter()
    detector.fit(X_train)
    train_time = time.perf_counter() - t0

    # ─ Threshold ─
    threshold = detector.auto_threshold(X_val, contamination=contamination)

    # ─ Score & predict test ─
    t1 = time.perf_counter()
    scores = detector.score(X_test)
    infer_time_ms = (time.perf_counter() - t1) / len(X_test) * 1000  # per sample

    y_pred = (scores >= threshold).astype(int)

    overall = compute_metrics(y_test, y_pred, scores, infer_time_ms)
    overall["Train_Time_s"] = round(train_time, 3)
    overall["Detector"]     = detector.name

    per_fault = compute_per_fault_metrics(y_test, y_pred, scores, label_names)
    per_fault.insert(0, "Detector", detector.name)

    return overall, per_fault
