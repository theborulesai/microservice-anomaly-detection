"""
visualizer.py
=============
All plots for the experimental results section of the paper.
Saves publication-quality figures to ../plots/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

PALETTE = {
    "IsolationForest": "#2196F3",
    "LOF":             "#4CAF50",
    "OneClassSVM":     "#FF9800",
    "Autoencoder":     "#9C27B0",
    "LSTM-AE":         "#F44336",
    "Ensemble":        "#009688",
}

FAULT_COLORS = {
    "cpu_stress":           "#E53935",
    "memory_leak":          "#FB8C00",
    "pod_failure":          "#8E24AA",
    "network_delay":        "#1E88E5",
    "ddos_attack":          "#00897B",
    "cryptomining":         "#F4511E",
    "privilege_escalation": "#6D4C41",
    "sql_injection":        "#3949AB",
}

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize":10,
    "figure.dpi":     150,
})


# ─────────────────────────── Helper ──────────────────────────────────────────

def savefig(name: str):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")
    return path


# ─────────────────────────── Fig 1: Overall Bar Chart ────────────────────────

def plot_overall_metrics(summary_df: pd.DataFrame) -> str:
    """Grouped bar chart of Precision / Recall / F1 / AUC-ROC per detector."""
    metrics = ["Precision", "Recall", "F1-Score", "AUC-ROC"]
    detectors = summary_df["Detector"].tolist()
    x = np.arange(len(detectors))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = summary_df[metric].values
        bars = ax.bar(x + (i - 1.5) * width, vals, width,
                      label=metric, color=color, alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(detectors, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Overall Detection Performance by Detector")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return savefig("fig1_overall_metrics.png")


# ─────────────────────────── Fig 2: Per-Fault Heatmap ────────────────────────

def plot_fault_heatmap(per_fault_df: pd.DataFrame, metric: str = "F1-Score") -> str:
    """Heatmap: rows=fault types, cols=detectors, cell=metric value."""
    pivot = per_fault_df.pivot(index="Fault_Type", columns="Detector", values=metric)

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = "YlOrRd"
    im = ax.imshow(pivot.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([ft.replace("_", " ").title() for ft in pivot.index])

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(metric)
    ax.set_title(f"Per-Fault-Type {metric} Heatmap")
    plt.tight_layout()
    return savefig(f"fig2_fault_heatmap_{metric.lower().replace('-','_')}.png")


# ─────────────────────────── Fig 3: ROC Curves ───────────────────────────────

def plot_roc_curves(roc_data: dict) -> str:
    """
    roc_data: {detector_name: (fpr_array, tpr_array, auc_val)}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0,1],[0,1],"k--", lw=1, label="Random (AUC=0.50)")

    for name, (fpr, tpr, auc) in roc_data.items():
        color = PALETTE.get(name, "#555")
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{name} (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – All Detectors")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return savefig("fig3_roc_curves.png")


# ─────────────────────────── Fig 4: PR Curves ────────────────────────────────

def plot_pr_curves(pr_data: dict) -> str:
    """pr_data: {detector_name: (prec_array, rec_array, auc_pr)}"""
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, (prec, rec, auc_pr) in pr_data.items():
        color = PALETTE.get(name, "#555")
        ax.plot(rec, prec, lw=2, color=color,
                label=f"{name} (AP={auc_pr:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves – All Detectors")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return savefig("fig4_pr_curves.png")


# ─────────────────────────── Fig 5: Anomaly Score Distributions ──────────────

def plot_score_distributions(scores_dict: dict,
                              y_true: np.ndarray) -> str:
    """
    scores_dict: {detector_name: score_array}
    Box plots of anomaly scores split by normal/anomaly.
    """
    n_det = len(scores_dict)
    fig, axes = plt.subplots(1, n_det, figsize=(3.5 * n_det, 5))
    if n_det == 1:
        axes = [axes]

    is_anomaly = (y_true != 0)
    for ax, (name, scores) in zip(axes, scores_dict.items()):
        # Normalize scores to [0,1] for comparability
        mn, mx = scores.min(), scores.max()
        sc_norm = (scores - mn) / (mx - mn + 1e-9)
        norm_scores = sc_norm[~is_anomaly]
        anom_scores = sc_norm[is_anomaly]

        bp = ax.boxplot([norm_scores, anom_scores],
                   patch_artist=True,
                   medianprops=dict(color="black", linewidth=2),
                   boxprops=dict(facecolor=PALETTE.get(name, "#aaa"), alpha=0.7))
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Normal", "Anomaly"])
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("Norm. Score" if ax == axes[0] else "")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Anomaly Score Distributions: Normal vs Anomaly", fontsize=13, y=1.02)
    plt.tight_layout()
    return savefig("fig5_score_distributions.png")


# ─────────────────────────── Fig 6: Detection Latency ────────────────────────

def plot_detection_latency(summary_df: pd.DataFrame) -> str:
    """Horizontal bar chart of per-sample inference latency (ms)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    detectors = summary_df["Detector"].tolist()
    lats = summary_df["Detection_Lat_ms"].values
    colors = [PALETTE.get(d, "#888") for d in detectors]

    bars = ax.barh(detectors, lats, color=colors, edgecolor="white", alpha=0.88)
    for bar, v in zip(bars, lats):
        ax.text(v + 0.001, bar.get_y() + bar.get_height()/2,
                f"{v:.4f} ms", va="center", fontsize=10)

    ax.set_xlabel("Per-sample Inference Latency (ms)")
    ax.set_title("Detection Latency Comparison")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return savefig("fig6_detection_latency.png")


# ─────────────────────────── Fig 7: Radar Chart ──────────────────────────────

def plot_radar_chart(summary_df: pd.DataFrame) -> str:
    """Radar chart comparing detectors across 5 dimensions."""
    metrics = ["Precision", "Recall", "F1-Score", "AUC-ROC", "AUC-PR"]
    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for _, row in summary_df.iterrows():
        name   = row["Detector"]
        values = [row[m] for m in metrics]
        values += values[:1]
        color  = PALETTE.get(name, "#888")
        ax.plot(angles, values, lw=2, color=color, label=name)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], size=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), framealpha=0.9)
    ax.set_title("Multi-Metric Radar – All Detectors", size=13, pad=20)
    plt.tight_layout()
    return savefig("fig7_radar_chart.png")


# ─────────────────────────── Fig 8: Training Time ────────────────────────────

def plot_training_time(summary_df: pd.DataFrame) -> str:
    """Log-scale bar chart of training times."""
    fig, ax = plt.subplots(figsize=(8, 4))
    detectors = summary_df["Detector"].tolist()
    times = summary_df["Train_Time_s"].values
    colors = [PALETTE.get(d, "#888") for d in detectors]

    bars = ax.bar(detectors, times, color=colors, edgecolor="white", alpha=0.88)
    for bar, v in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                f"{v:.2f}s", ha="center", va="bottom", fontsize=10)

    ax.set_yscale("log")
    ax.set_ylabel("Training Time (s, log scale)")
    ax.set_title("Training Time Comparison")
    ax.set_xticklabels(detectors, rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return savefig("fig8_training_time.png")


# ─────────────────────────── Fig 9: Confusion Matrices ───────────────────────

def plot_confusion_matrices(cm_dict: dict) -> str:
    """cm_dict: {detector_name: confusion_matrix_2x2}"""
    n = len(cm_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, cm_dict.items()):
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[i,j] > cm.max()/2 else "black")
        ax.set_xticks([0,1])
        ax.set_xticklabels(["Pred Normal","Pred Anomaly"], rotation=15)
        ax.set_yticks([0,1])
        ax.set_yticklabels(["True Normal","True Anomaly"])
        ax.set_title(name, fontsize=11)
    fig.suptitle("Confusion Matrices", fontsize=13)
    plt.tight_layout()
    return savefig("fig9_confusion_matrices.png")


# ─────────────────────────── Fig 10: Score Timeline ──────────────────────────

def plot_score_timeline(scores: np.ndarray, y_true: np.ndarray,
                        threshold: float, detector_name: str) -> str:
    """Line plot of anomaly scores over time with threshold and ground truth shading."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    t = np.arange(len(scores))

    # Normalize
    mn, mx = scores.min(), scores.max()
    sc_n = (scores - mn) / (mx - mn + 1e-9)
    thr_n = (threshold - mn) / (mx - mn + 1e-9)

    ax1.plot(t, sc_n, lw=0.8, color="#2196F3", alpha=0.85, label="Anomaly Score")
    ax1.axhline(thr_n, color="red", lw=1.5, ls="--", label=f"Threshold ({thr_n:.2f})")

    is_anom = (y_true != 0)
    for start, end in _consecutive_ranges(is_anom):
        ax1.axvspan(start, end, alpha=0.15, color="red", label="_nolegend_")

    ax1.set_ylabel("Norm. Anomaly Score")
    ax1.set_title(f"Anomaly Score Timeline – {detector_name}")
    ax1.legend(loc="upper left")
    ax1.grid(linestyle="--", alpha=0.3)

    # Ground truth bar
    ax2.fill_between(t, is_anom.astype(int), step="mid",
                     color="red", alpha=0.5, label="True Anomaly")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Normal", "Anomaly"])
    ax2.set_xlabel("Time Step")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    return savefig(f"fig10_score_timeline_{detector_name.lower().replace(' ','_')}.png")


def _consecutive_ranges(mask):
    """Yield (start, end) index pairs for True runs in a boolean array."""
    ranges = []
    in_range, start = False, 0
    for i, v in enumerate(mask):
        if v and not in_range:
            start = i
            in_range = True
        elif not v and in_range:
            ranges.append((start, i))
            in_range = False
    if in_range:
        ranges.append((start, len(mask)))
    return ranges
