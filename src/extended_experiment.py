"""
extended_experiment.py
======================
Phase 3 – Extended analysis:
  1. Multi-seed stability (seeds 42, 123, 777) → stability_analysis.csv
  2. Threshold sensitivity sweep per detector
  3. Per-fault-type deep drill-down (P/R/F1/AUC per fault × detector)
  4. Detector operating point analysis (FPR vs FNR tradeoff table)
  5. All additional plots (threshold curves, stability bar, FPR-FNR scatter)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import time

from data_generator import build_dataset, LABEL_NAMES, FAULT_TYPES
from detectors import (IsolationForestDetector, LOFDetector, OCSVMDetector,
                        AutoencoderDetector, EnsembleDetector)
from evaluator import make_splits, evaluate_detector, compute_metrics, compute_per_fault_metrics
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, f1_score, precision_score, recall_score

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "plots")
SEEDS       = [42, 123, 777]
CONTAMINATION = 0.45

PALETTE = {
    "IsolationForest": "#2196F3",
    "LOF":             "#4CAF50",
    "OneClassSVM":     "#FF9800",
    "Autoencoder":     "#9C27B0",
    "Ensemble":        "#009688",
    "LSTM-AE":         "#F44336",
}

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13,
                      "figure.dpi": 150, "font.family": "DejaVu Sans"})


def savefig(name):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")


# =============================================================================
# PHASE 3A — Multi-Seed Stability
# =============================================================================
def run_stability_analysis():
    print("\n[PHASE 3A] Multi-Seed Stability Analysis")
    print("-" * 50)
    all_rows = []
    det_classes = [
        ("IsolationForest", lambda s: IsolationForestDetector(random_state=s)),
        ("LOF",             lambda s: LOFDetector()),
        ("OneClassSVM",     lambda s: OCSVMDetector()),
        ("Autoencoder",     lambda s: AutoencoderDetector(epochs=50, random_state=s)),
        ("Ensemble",        lambda s: EnsembleDetector()),
    ]
    for seed in SEEDS:
        print(f"  Seed {seed} ...", end="", flush=True)
        df, labels = build_dataset(n_normal=5000, n_anomaly_per_type=400, seed=seed)
        X = df.values; y = labels.values
        X_train, y_train, X_val, y_val, X_test, y_test = make_splits(X, y, seed=seed)
        for name, mk_det in det_classes:
            det = mk_det(seed)
            overall, _ = evaluate_detector(
                det, X_train, X_val, y_val, X_test, y_test,
                contamination=CONTAMINATION, label_names=LABEL_NAMES)
            overall["Seed"] = seed
            all_rows.append(overall)
        print(" done")

    stab_df = pd.DataFrame(all_rows)
    stab_df.to_csv(os.path.join(RESULTS_DIR, "stability_all_seeds.csv"), index=False)

    summary = stab_df.groupby("Detector").agg(
        F1_Mean=("F1-Score", "mean"),
        F1_Std =("F1-Score", "std"),
        P_Mean =("Precision","mean"),
        R_Mean =("Recall","mean"),
        AUC_Mean=("AUC-ROC","mean"),
        AUC_Std =("AUC-ROC","std"),
    ).reset_index().sort_values("F1_Mean", ascending=False)
    summary.to_csv(os.path.join(RESULTS_DIR, "stability_summary.csv"), index=False)
    print("\n  Stability Summary:")
    print(summary.to_string(index=False))

    # Plot: Error-bar chart of F1 across seeds
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary))
    colors = [PALETTE.get(d, "#888") for d in summary["Detector"]]
    bars = ax.bar(x, summary["F1_Mean"], yerr=summary["F1_Std"],
                  color=colors, edgecolor="white", alpha=0.88,
                  capsize=5, error_kw={"linewidth": 2})
    for bar, row in zip(bars, summary.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + row.F1_Std + 0.005,
                f"{row.F1_Mean:.3f}\n±{row.F1_Std:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["Detector"], rotation=15, ha="right")
    ax.set_ylim(0.7, 1.08)
    ax.set_ylabel("F1-Score")
    ax.set_title("Detector Stability Across 3 Random Seeds (Mean ± Std)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    savefig("fig11_stability_analysis.png")
    return summary


# =============================================================================
# PHASE 3B — Threshold Sensitivity
# =============================================================================
def run_threshold_sensitivity(X_train, X_val, y_val, X_test, y_test):
    print("\n[PHASE 3B] Threshold Sensitivity Analysis")
    print("-" * 50)
    contamination_values = np.round(np.arange(0.05, 0.75, 0.05), 2)
    detectors = [
        IsolationForestDetector(random_state=42),
        LOFDetector(),
        OCSVMDetector(),
        AutoencoderDetector(epochs=50, random_state=42),
        EnsembleDetector(),
    ]

    all_rows = []
    y_bin = (y_test != 0).astype(int)

    fig, ax = plt.subplots(figsize=(11, 6))

    for det in detectors:
        det.fit(X_train)
        scores_test = det.score(X_test)
        f1_vals = []
        for c in contamination_values:
            thr = det.auto_threshold(X_val, contamination=c)
            y_pred = (scores_test >= thr).astype(int)
            f1 = f1_score(y_bin, y_pred, zero_division=0)
            prec = precision_score(y_bin, y_pred, zero_division=0)
            rec  = recall_score(y_bin, y_pred, zero_division=0)
            f1_vals.append(f1)
            all_rows.append({"Detector": det.name, "Contamination": c,
                              "F1": round(f1,4), "Precision": round(prec,4),
                              "Recall": round(rec,4)})
        ax.plot(contamination_values, f1_vals, lw=2,
                color=PALETTE.get(det.name, "#888"),
                marker="o", markersize=4, label=det.name)
        print(f"  {det.name}: best F1={max(f1_vals):.4f} at c={contamination_values[np.argmax(f1_vals)]:.2f}")

    ax.set_xlabel("Contamination Threshold (rho)")
    ax.set_ylabel("F1-Score")
    ax.set_title("F1-Score vs Contamination Threshold (Sensitivity Analysis)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    savefig("fig12_threshold_sensitivity.png")

    sens_df = pd.DataFrame(all_rows)
    sens_df.to_csv(os.path.join(RESULTS_DIR, "threshold_sensitivity.csv"), index=False)
    return sens_df


# =============================================================================
# PHASE 3C — Per-Fault Deep Drill-Down
# =============================================================================
def run_per_fault_deep(X_train, X_val, y_val, X_test, y_test):
    print("\n[PHASE 3C] Per-Fault Deep Drill-Down")
    print("-" * 50)
    detectors = [
        IsolationForestDetector(random_state=42),
        LOFDetector(),
        OCSVMDetector(),
        AutoencoderDetector(epochs=50, random_state=42),
        EnsembleDetector(),
    ]

    all_per_fault = []
    for det in detectors:
        det.fit(X_train)
        thr    = det.auto_threshold(X_val, contamination=CONTAMINATION)
        scores = det.score(X_test)
        y_pred = (scores >= thr).astype(int)
        pf     = compute_per_fault_metrics(y_test, y_pred, scores, LABEL_NAMES)
        pf.insert(0, "Detector", det.name)
        all_per_fault.append(pf)
        print(f"  {det.name}: avg fault F1 = {pf['F1-Score'].mean():.4f}")

    pf_df = pd.concat(all_per_fault, ignore_index=True)
    pf_df.to_csv(os.path.join(RESULTS_DIR, "per_fault_deep.csv"), index=False)

    # Grouped bar: F1 per fault type per detector
    fault_types = pf_df["Fault_Type"].unique().tolist()
    det_names   = pf_df["Detector"].unique().tolist()
    x = np.arange(len(fault_types))
    width = 0.15

    fig, ax = plt.subplots(figsize=(15, 6))
    for i, dname in enumerate(det_names):
        sub = pf_df[pf_df["Detector"] == dname].set_index("Fault_Type")
        vals = [sub.loc[ft, "F1-Score"] if ft in sub.index else 0 for ft in fault_types]
        ax.bar(x + (i - len(det_names)/2 + 0.5) * width, vals, width,
               label=dname, color=PALETTE.get(dname, "#888"),
               alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([ft.replace("_", "\n") for ft in fault_types], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1-Score")
    ax.set_title("Per-Fault-Type F1-Score by Detector")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    savefig("fig13_per_fault_grouped_bar.png")
    return pf_df


# =============================================================================
# PHASE 3D — FPR vs FNR Operating Point Scatter
# =============================================================================
def run_fpr_fnr_scatter(X_train, X_val, y_val, X_test, y_test):
    print("\n[PHASE 3D] FPR vs FNR Operating Points")
    print("-" * 50)
    detectors = [
        IsolationForestDetector(random_state=42),
        LOFDetector(),
        OCSVMDetector(),
        AutoencoderDetector(epochs=50, random_state=42),
        EnsembleDetector(),
    ]
    y_bin = (y_test != 0).astype(int)
    rows = []

    fig, ax = plt.subplots(figsize=(8, 7))
    contamination_sweep = np.round(np.arange(0.05, 0.75, 0.05), 2)

    for det in detectors:
        det.fit(X_train)
        sc = det.score(X_test)
        fprs, fnrs = [], []
        for c in contamination_sweep:
            thr    = det.auto_threshold(X_val, contamination=c)
            y_pred = (sc >= thr).astype(int)
            fpr    = np.sum((y_pred==1)&(y_bin==0)) / max(np.sum(y_bin==0),1)
            fnr    = np.sum((y_pred==0)&(y_bin==1)) / max(np.sum(y_bin==1),1)
            fprs.append(fpr)
            fnrs.append(fnr)
        color = PALETTE.get(det.name, "#888")
        ax.plot(fprs, fnrs, lw=2, color=color, alpha=0.7)
        # Mark optimal (min Euclidean from (0,0))
        dists = [np.sqrt(f**2 + n**2) for f, n in zip(fprs, fnrs)]
        best  = np.argmin(dists)
        ax.scatter(fprs[best], fnrs[best], s=100, color=color,
                   zorder=5, label=f"{det.name} (opt c={contamination_sweep[best]:.2f})")
        rows.append({"Detector": det.name, "Optimal_FPR": round(fprs[best],4),
                     "Optimal_FNR": round(fnrs[best],4),
                     "Optimal_Contamination": contamination_sweep[best]})
        print(f"  {det.name}: optimal FPR={fprs[best]:.4f}, FNR={fnrs[best]:.4f}")

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("False Negative Rate (FNR)")
    ax.set_title("FPR vs FNR Operating Point Curves\n(dots = optimal operating point)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    savefig("fig14_fpr_fnr_operating_points.png")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "optimal_operating_points.csv"), index=False)


# =============================================================================
# PHASE 3E — Anomaly Type Classification Confusion (Multiclass)
# =============================================================================
def run_multiclass_analysis(X_train, X_val, y_val, X_test, y_test):
    print("\n[PHASE 3E] Multiclass Fault Classification Analysis")
    print("-" * 50)
    # Use best tabular detector (LOF) for multiclass breakdown
    det = LOFDetector()
    det.fit(X_train)
    thr    = det.auto_threshold(X_val, contamination=CONTAMINATION)
    scores = det.score(X_test)
    y_pred_bin = (scores >= thr).astype(int)

    # For each fault type, compute detection rate = TP / (TP + FN)
    rows = []
    for label_int, label_name in LABEL_NAMES.items():
        if label_int == 0:
            continue
        mask = y_test == label_int
        if mask.sum() == 0:
            continue
        detected = y_pred_bin[mask].sum()
        total    = mask.sum()
        rate     = detected / total
        rows.append({
            "Fault_Type": label_name,
            "Total_Samples": int(total),
            "Detected": int(detected),
            "Detection_Rate": round(rate, 4),
            "Missed": int(total - detected),
        })
        print(f"  {label_name:25s}: {detected}/{total} detected ({rate:.1%})")

    dr_df = pd.DataFrame(rows).sort_values("Detection_Rate", ascending=True)
    dr_df.to_csv(os.path.join(RESULTS_DIR, "detection_rate_by_fault.csv"), index=False)

    # Horizontal bar chart: detection rate per fault
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#4CAF50" if r >= 0.9 else "#FF9800" if r >= 0.7 else "#F44336"
              for r in dr_df["Detection_Rate"]]
    bars = ax.barh(dr_df["Fault_Type"].str.replace("_", " ").str.title(),
                   dr_df["Detection_Rate"],
                   color=colors, edgecolor="white", alpha=0.88)
    for bar, row in zip(bars, dr_df.itertuples()):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{row.Detection_Rate:.1%} ({row.Detected}/{row.Total_Samples})",
                va="center", fontsize=10)
    ax.axvline(0.9, color="green", lw=1.5, ls="--", alpha=0.6, label="90% target")
    ax.axvline(0.7, color="orange", lw=1.5, ls="--", alpha=0.6, label="70% threshold")
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Detection Rate (Recall per Fault Type)")
    ax.set_title("LOF Detection Rate per Fault/Attack Type")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    savefig("fig15_detection_rate_by_fault.png")
    return dr_df


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  PHASE 3 — EXTENDED STATISTICAL & DIAGNOSTIC ANALYSIS")
    print("=" * 70)
    t0 = time.time()

    # Build base dataset (seed=42) for phases 3B-3E
    print("\nPreparing base dataset (seed=42)...")
    df, labels = build_dataset(n_normal=5000, n_anomaly_per_type=400, seed=42)
    X = df.values; y = labels.values
    X_train, y_train, X_val, y_val, X_test, y_test = make_splits(X, y, seed=42)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    stab  = run_stability_analysis()
    sens  = run_threshold_sensitivity(X_train, X_val, y_val, X_test, y_test)
    pf    = run_per_fault_deep(X_train, X_val, y_val, X_test, y_test)
    run_fpr_fnr_scatter(X_train, X_val, y_val, X_test, y_test)
    dr    = run_multiclass_analysis(X_train, X_val, y_val, X_test, y_test)

    print(f"\n[OK] Phase 3 complete in {time.time()-t0:.1f}s")
    print(f"     Results -> {RESULTS_DIR}")
    print(f"     Plots   -> {PLOTS_DIR}")
