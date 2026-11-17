"""
statistical_analysis.py
========================
Deep statistical analysis of experimental results:
  - Friedman test + Nemenyi post-hoc for multi-detector comparison
  - Effect size (Cohen's d) between best vs each detector
  - Per-fault detection rate heatmap data
  - Threshold sensitivity analysis
  - FPR-FNR operating-point curves
Saves analysis tables to ../results/
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from itertools import combinations
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size between two arrays."""
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def friedman_and_nemenyi(scores_matrix: np.ndarray,
                          detector_names: list) -> pd.DataFrame:
    """
    Friedman test on detector ranks, then pairwise Wilcoxon signed-rank
    as a Nemenyi-style post-hoc.

    scores_matrix: (n_samples, n_detectors)  anomaly scores (not metrics)
    Returns p-value matrix.
    """
    # Rank each sample's scores across detectors
    n, d = scores_matrix.shape
    rank_matrix = np.zeros_like(scores_matrix, dtype=float)
    for i in range(n):
        rank_matrix[i] = stats.rankdata(scores_matrix[i])

    # Friedman statistic
    k = d
    n_samples = n
    Tj = rank_matrix.sum(axis=0)
    CF = n_samples * k * (k + 1)**2 / 4
    SS_t = sum(Tj**2) / n_samples - n_samples * k * (k+1)**2 / 4
    SS_e = (rank_matrix**2).sum() - n_samples * k * (k+1)**2 / 4
    if SS_e == 0:
        friedman_p = 1.0
        friedman_stat = 0.0
    else:
        friedman_stat = (n_samples - 1) * SS_t / SS_e
        friedman_p = 1 - stats.chi2.cdf(friedman_stat, df=k-1)

    print(f"\n  Friedman statistic = {friedman_stat:.4f}, p = {friedman_p:.6f}")
    if friedman_p < 0.05:
        print("  -> Significant difference among detectors (p < 0.05)")
    else:
        print("  -> No significant difference (p >= 0.05)")

    # Pairwise Wilcoxon signed-rank
    rows = []
    for i, j in combinations(range(d), 2):
        stat, p = stats.wilcoxon(rank_matrix[:, i], rank_matrix[:, j],
                                  alternative='two-sided')
        rows.append({
            "Detector_A": detector_names[i],
            "Detector_B": detector_names[j],
            "Wilcoxon_stat": round(stat, 2),
            "p_value": round(p, 6),
            "Significant": "Yes" if p < 0.05 else "No",
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "pairwise_wilcoxon.csv"), index=False)
    return df


def threshold_sensitivity(detector, X_val, X_test, y_test,
                           contamination_values=None):
    """
    Sweep contamination threshold and measure P / R / F1.
    Returns DataFrame of results.
    """
    if contamination_values is None:
        contamination_values = np.arange(0.05, 0.75, 0.05)

    from sklearn.metrics import precision_score, recall_score, f1_score
    y_bin = (y_test != 0).astype(int)
    scores = detector.score(X_test)

    rows = []
    for c in contamination_values:
        threshold = detector.auto_threshold(X_val, contamination=c)
        y_pred = (scores >= threshold).astype(int)
        rows.append({
            "Contamination": round(c, 2),
            "Threshold": round(threshold, 6),
            "Precision": round(precision_score(y_bin, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_bin, y_pred, zero_division=0), 4),
            "F1-Score":  round(f1_score(y_bin, y_pred, zero_division=0), 4),
            "FPR": round(np.sum((y_pred==1)&(y_bin==0)) / max(np.sum(y_bin==0),1), 4),
        })
    return pd.DataFrame(rows)


def per_seed_stability(build_fn, detector_classes, seeds=(42, 123, 777)):
    """
    Run experiments across multiple seeds; compute mean ± std of F1.
    build_fn() -> (X_train, y_train, X_val, y_val, X_test, y_test)
    detector_classes: list of detector constructor callables
    """
    from evaluator import evaluate_detector
    from data_generator import LABEL_NAMES

    all_rows = []
    for seed in seeds:
        X_train, y_train, X_val, y_val, X_test, y_test = build_fn(seed)
        for det_cls in detector_classes:
            det = det_cls()
            overall, _ = evaluate_detector(
                det, X_train, X_val, y_val, X_test, y_test,
                contamination=0.45, label_names=LABEL_NAMES
            )
            overall["seed"] = seed
            all_rows.append(overall)

    df = pd.DataFrame(all_rows)
    stability = df.groupby("Detector")["F1-Score"].agg(["mean", "std"]).reset_index()
    stability.columns = ["Detector", "F1_Mean", "F1_Std"]
    stability = stability.sort_values("F1_Mean", ascending=False)
    stability.to_csv(os.path.join(RESULTS_DIR, "stability_analysis.csv"), index=False)
    return stability
