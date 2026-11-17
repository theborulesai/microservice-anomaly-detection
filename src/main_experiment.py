"""
main_experiment.py
==================
Master experiment runner.
Executes the full pipeline:
  1. Generate synthetic microservice telemetry dataset
  2. Split into train / val / test (unsupervised protocol)
  3. Train and evaluate all detectors
  4. Generate all plots
  5. Save CSV results tables
  6. Print a formatted summary table
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import time

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

from data_generator import build_dataset, build_windowed_dataset, LABEL_NAMES, FAULT_TYPES
from detectors import (
    IsolationForestDetector, LOFDetector, OCSVMDetector,
    AutoencoderDetector, LSTMAutoencoderDetector, EnsembleDetector
)
from evaluator import make_splits, evaluate_detector
from visualizer import (
    plot_overall_metrics, plot_fault_heatmap,
    plot_roc_curves, plot_pr_curves,
    plot_score_distributions, plot_detection_latency,
    plot_radar_chart, plot_training_time,
    plot_confusion_matrices, plot_score_timeline,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

SEED         = 42
CONTAMINATION = 0.45   # validation set has ~50% anomalies; set high to balance P-R


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 – TABULAR DETECTORS
# ═══════════════════════════════════════════════════════════════════════════════

def run_tabular_experiments():
    print("\n" + "="*70)
    print("  PHASE 1 – TABULAR DETECTORS")
    print("="*70)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/6] Generating dataset …")
    df, labels = build_dataset(n_normal=5000, n_anomaly_per_type=400, seed=SEED)
    X = df.values
    y = labels.values
    print(f"      Dataset: {X.shape[0]:,} samples × {X.shape[1]} features")
    print(f"      Class distribution:\n{pd.Series(y).map(LABEL_NAMES).value_counts().to_string()}")

    # ── Split ─────────────────────────────────────────────────────────────────
    print("\n[2/6] Splitting …")
    X_train, y_train, X_val, y_val, X_test, y_test = make_splits(X, y, seed=SEED)
    print(f"      Train: {len(X_train):,} (all normal)")
    print(f"      Val  : {len(X_val):,}  | Test: {len(X_test):,}")

    # ── Detectors ────────────────────────────────────────────────────────────
    detectors = [
        IsolationForestDetector(n_estimators=200, random_state=SEED),
        LOFDetector(n_neighbors=20),
        OCSVMDetector(nu=0.05),
        AutoencoderDetector(epochs=50, random_state=SEED),
        EnsembleDetector(),
    ]

    all_overall, all_per_fault = [], []
    scores_dict, roc_data, pr_data, cm_dict = {}, {}, {}, {}

    print("\n[3/6] Training & evaluating detectors …")
    for det in detectors:
        print(f"\n  ▶ {det.name} …")
        t0 = time.time()
        overall, per_fault = evaluate_detector(
            det, X_train, X_val, y_val, X_test, y_test,
            contamination=CONTAMINATION,
            label_names=LABEL_NAMES
        )
        elapsed = time.time() - t0
        print(f"    F1={overall['F1-Score']:.4f}  AUC-ROC={overall['AUC-ROC']:.4f}  "
              f"Precision={overall['Precision']:.4f}  Recall={overall['Recall']:.4f}  "
              f"[{elapsed:.1f}s]")

        all_overall.append(overall)
        all_per_fault.append(per_fault)

        # Scores for plots
        scores = det.score(X_test)
        scores_dict[det.name] = scores
        y_bin = (y_test != 0).astype(int)
        threshold = det.auto_threshold(X_val, CONTAMINATION)

        # ROC
        fpr_arr, tpr_arr, _ = roc_curve(y_bin, scores)
        roc_data[det.name]   = (fpr_arr, tpr_arr, overall["AUC-ROC"])

        # PR
        prec_arr, rec_arr, _ = precision_recall_curve(y_bin, scores)
        pr_data[det.name]    = (prec_arr, rec_arr, overall["AUC-PR"])

        # Confusion matrix
        y_pred = (scores >= threshold).astype(int)
        cm_dict[det.name] = confusion_matrix(y_bin, y_pred)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    print("\n[4/6] Saving result tables …")
    summary_df    = pd.DataFrame(all_overall)
    per_fault_df  = pd.concat(all_per_fault, ignore_index=True)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "overall_metrics.csv"), index=False)
    per_fault_df.to_csv(os.path.join(RESULTS_DIR, "per_fault_metrics.csv"), index=False)
    print(f"      Saved to {RESULTS_DIR}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[5/6] Generating plots …")
    plot_overall_metrics(summary_df)
    plot_fault_heatmap(per_fault_df, metric="F1-Score")
    plot_fault_heatmap(per_fault_df, metric="AUC-ROC")
    plot_roc_curves(roc_data)
    plot_pr_curves(pr_data)
    plot_score_distributions(scores_dict, y_test)
    plot_detection_latency(summary_df)
    plot_radar_chart(summary_df)
    plot_training_time(summary_df)
    plot_confusion_matrices(cm_dict)

    # Score timeline for best detector
    best_name = summary_df.loc[summary_df["F1-Score"].idxmax(), "Detector"]
    best_det = next(d for d in detectors if d.name == best_name)
    best_scores = scores_dict[best_name]
    best_threshold = best_det.auto_threshold(X_val, CONTAMINATION)
    plot_score_timeline(best_scores, y_test, best_threshold, best_name)

    # ── Summary Table ─────────────────────────────────────────────────────────
    print("\n[6/6] Results Summary")
    print("="*80)
    print(summary_df[["Detector","Precision","Recall","F1-Score",
                       "AUC-ROC","AUC-PR","FPR","Detection_Lat_ms",
                       "Train_Time_s"]].to_string(index=False))
    print("="*80)

    return summary_df, per_fault_df, scores_dict, y_test


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 – LSTM SEQUENCE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_lstm_experiment():
    print("\n" + "="*70)
    print("  PHASE 2 – LSTM AUTOENCODER (Sequence Model)")
    print("="*70)

    print("\n[1/4] Building windowed dataset …")
    X_win, y_win, feat_names = build_windowed_dataset(
        window_size=30, n_windows_normal=600, n_windows_per_fault=80, seed=SEED
    )
    print(f"      X: {X_win.shape}, y: {y_win.shape}")
    print(f"      Features: {feat_names}")

    # Unsupervised split
    from sklearn.model_selection import train_test_split
    normal_idx  = np.where(y_win == 0)[0]
    anomaly_idx = np.where(y_win != 0)[0]

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(normal_idx))
    n_train = int(len(normal_idx) * 0.6)
    train_idx = normal_idx[perm[:n_train]]
    rest_norm = normal_idx[perm[n_train:]]
    val_anom  = anomaly_idx[:len(anomaly_idx)//2]
    test_anom = anomaly_idx[len(anomaly_idx)//2:]
    val_idx   = np.concatenate([rest_norm[:len(rest_norm)//2], val_anom])
    test_idx  = np.concatenate([rest_norm[len(rest_norm)//2:], test_anom])

    X_tr = X_win[train_idx]; y_tr = y_win[train_idx]
    X_v  = X_win[val_idx];   y_v  = y_win[val_idx]
    X_te = X_win[test_idx];  y_te = y_win[test_idx]

    print(f"\n[2/4] Training LSTM-AE …")
    lstm_det = LSTMAutoencoderDetector(
        hidden_size=32, num_layers=2,
        epochs=30, batch_size=32, random_state=SEED
    )
    t0 = time.time()
    lstm_det.fit(X_tr)
    train_time = time.time() - t0
    print(f"      Trained in {train_time:.1f}s  (backend: {lstm_det._backend})")

    threshold = lstm_det.auto_threshold(X_v, contamination=CONTAMINATION)
    scores    = lstm_det.score(X_te)
    y_pred    = (scores >= threshold).astype(int)
    y_bin     = (y_te != 0).astype(int)

    from evaluator import compute_metrics, compute_per_fault_metrics
    metrics = compute_metrics(y_te, y_pred, scores,
                               detection_latency_ms=train_time/len(X_te)*1000)
    metrics["Detector"]    = "LSTM-AE"
    metrics["Train_Time_s"]= round(train_time, 3)
    per_fault = compute_per_fault_metrics(y_te, y_pred, scores, LABEL_NAMES)
    per_fault.insert(0, "Detector", "LSTM-AE")

    print("\n[3/4] LSTM-AE Results:")
    for k, v in metrics.items():
        if k not in ("Detector",):
            print(f"      {k}: {v}")

    lstm_summary = pd.DataFrame([metrics])
    lstm_summary.to_csv(os.path.join(RESULTS_DIR, "lstm_overall_metrics.csv"), index=False)
    per_fault.to_csv(os.path.join(RESULTS_DIR, "lstm_per_fault_metrics.csv"), index=False)

    print("\n[4/4] LSTM score timeline …")
    plot_score_timeline(scores, y_te, threshold, "LSTM-AE")

    return lstm_summary, per_fault


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 – COMPREHENSIVE COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_final_comparison(tab_summary: pd.DataFrame,
                            lstm_summary: pd.DataFrame):
    all_df = pd.concat([tab_summary, lstm_summary], ignore_index=True)
    all_df.to_csv(os.path.join(RESULTS_DIR, "final_comparison.csv"), index=False)

    print("\n" + "="*90)
    print("  FINAL COMPARISON – ALL DETECTORS")
    print("="*90)
    display_cols = ["Detector","Precision","Recall","F1-Score",
                    "AUC-ROC","AUC-PR","FPR","FNR","Train_Time_s","Detection_Lat_ms"]
    print(all_df[display_cols].sort_values("F1-Score", ascending=False).to_string(index=False))
    print("="*90)

    # Rank table
    rank_cols = ["Precision","Recall","F1-Score","AUC-ROC","AUC-PR"]
    rank_df = all_df[["Detector"] + rank_cols].copy()
    for col in rank_cols:
        rank_df[col + "_rank"] = rank_df[col].rank(ascending=False).astype(int)
    rank_df["Avg_Rank"] = rank_df[[c+"_rank" for c in rank_cols]].mean(axis=1)
    rank_df = rank_df.sort_values("Avg_Rank")

    print("\n  RANKING TABLE (by average rank, lower = better)")
    print("-"*60)
    print(rank_df[["Detector","Avg_Rank"] + [c+"_rank" for c in rank_cols]].to_string(index=False))

    rank_df.to_csv(os.path.join(RESULTS_DIR, "ranking_table.csv"), index=False)
    return all_df


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  Unsupervised Anomaly Detection for Microservices")
    print("  Comparative Experimental Evaluation")
    print("="*70)
    t_total = time.time()

    tab_summary, per_fault_df, scores_dict, y_test = run_tabular_experiments()
    lstm_summary, lstm_per_fault                   = run_lstm_experiment()
    final_df = print_final_comparison(tab_summary, lstm_summary)

    print(f"\n[OK] All experiments complete in {time.time()-t_total:.1f}s")
    print(f"   Results → {RESULTS_DIR}")
    print(f"   Plots   → {PLOTS_DIR}")
