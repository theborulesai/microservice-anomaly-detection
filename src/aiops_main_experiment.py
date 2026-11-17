"""
aiops_main_experiment.py
Complete experiment pipeline using real AIOps Challenge 2020 dataset.
Runs all 6 detectors + evaluation on real production data.
Authors: ANUSHREE S (23BEC007) & SAINATH (23BCS115)
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, average_precision_score
)
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'

BASE_DIR = Path(r'C:\Users\Sainath S Borule\Downloads\microservice-anomaly-detection')
sys.path.insert(0, str(BASE_DIR / 'src'))

from aiops_loader import load_all_days, prepare_train_test
# ── Correct class names from detectors.py ────────────────────────────────────
from detectors import (IsolationForestDetector, LOFDetector, OCSVMDetector,
                       AutoencoderDetector, LSTMAutoencoderDetector, EnsembleDetector)

RESULTS_DIR = BASE_DIR / 'results'
PLOTS_DIR   = BASE_DIR / 'plots'
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load AIOps Dataset
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset():
    cache = BASE_DIR / 'data' / 'aiops_combined.csv'
    if cache.exists():
        print('[DATA] Loading from cache...')
        combined = pd.read_csv(cache, encoding='utf-8-sig')
        combined['timestamp']  = pd.to_numeric(combined['timestamp'],  errors='coerce')
        combined['label']      = pd.to_numeric(combined['label'],      errors='coerce').fillna(0).astype(int)
        combined['fault_type'] = combined['fault_type'].fillna('normal')
        n = len(combined)
        na = combined['label'].sum()
        nf = combined.shape[1] - 3
        print(f'[DATA] Cache: {n} windows, {na} anomalous ({100*na/n:.1f}%), {nf} features')
    else:
        print('[DATA] Loading from raw day files...')
        combined = load_all_days(verbose=True)
        combined.to_csv(cache, index=False, encoding='utf-8-sig')

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Evaluate one detector (tabular — 2D input)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_detector(name, detector, X_train_s, X_val_s, X_test_s,
                      y_val, y_test):
    """
    All X inputs are already scaled (2D numpy arrays).
    Uses detector.fit(X) / detector.score(X) interface from BaseDetector.
    """
    print(f'\n  [{name}]', end='', flush=True)

    # Train
    t0 = time.time()
    detector.fit(X_train_s)
    train_time = time.time() - t0

    # Score on val → pick threshold
    scores_val  = detector.score(X_val_s)
    best_f1, best_thr = 0.0, float(np.percentile(scores_val, 85))
    for pct in np.linspace(5, 95, 181):
        thr = np.percentile(scores_val, pct)
        preds = (scores_val >= thr).astype(int)
        if y_val.sum() > 0:
            f = f1_score(y_val, preds, zero_division=0)
            if f > best_f1:
                best_f1, best_thr = f, thr

    # Score on test
    t0 = time.time()
    scores_test = detector.score(X_test_s)
    lat = (time.time() - t0) * 1000 / max(len(X_test_s), 1)

    y_pred = (scores_test >= best_thr).astype(int)

    # Metrics
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        'name':       name,
        'f1':         f1_score(y_test, y_pred, zero_division=0),
        'auc_roc':    roc_auc_score(y_test, scores_test) if y_test.sum() > 0 else 0.0,
        'auc_pr':     average_precision_score(y_test, scores_test) if y_test.sum() > 0 else 0.0,
        'precision':  precision_score(y_test, y_pred, zero_division=0),
        'recall':     recall_score(y_test, y_pred, zero_division=0),
        'accuracy':   accuracy_score(y_test, y_pred),
        'fpr':        fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'fnr':        fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        'train_time': train_time,
        'latency_ms': lat,
        'threshold':  best_thr,
    }
    print(f' F1={metrics["f1"]:.4f} AUC={metrics["auc_roc"]:.4f} '
          f'Train={train_time:.2f}s Lat={lat:.3f}ms')
    return metrics, y_pred, scores_test


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2b: Evaluate LSTM-AE (3D input: N, T, F)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_lstm(name, detector, X_train_raw, X_val_raw, X_test_raw,
                  y_val, y_test, window_size=5):
    """
    LSTMAutoencoderDetector.fit() expects (N, T, F) — it handles scaling internally.
    We create sliding windows from the raw (unscaled) 2D arrays.
    """
    print(f'\n  [{name}]', end='', flush=True)

    def make_windows(X, ws):
        N, F = X.shape
        if N < ws:
            # Pad with zeros if too short
            pad = np.zeros((ws - N, F))
            X = np.vstack([pad, X])
            N = ws
        seqs = np.stack([X[i:i+ws] for i in range(N - ws + 1)], axis=0)
        return seqs  # (N-ws+1, ws, F)

    Xtr3 = make_windows(X_train_raw, window_size)
    Xva3 = make_windows(X_val_raw,   window_size)
    Xte3 = make_windows(X_test_raw,  window_size)

    # Labels — align to window end position
    n_val_w  = Xva3.shape[0]
    n_test_w = Xte3.shape[0]
    # Pad labels if windows fewer than original
    yv = y_val[-n_val_w:]  if len(y_val)  >= n_val_w  else np.concatenate([np.zeros(n_val_w  - len(y_val),  dtype=int), y_val])
    yt = y_test[-n_test_w:] if len(y_test) >= n_test_w else np.concatenate([np.zeros(n_test_w - len(y_test), dtype=int), y_test])

    # Train
    t0 = time.time()
    detector.fit(Xtr3)
    train_time = time.time() - t0

    # Score val
    scores_val = detector.score(Xva3)
    best_f1, best_thr = 0.0, float(np.percentile(scores_val, 85))
    for pct in np.linspace(5, 95, 181):
        thr = np.percentile(scores_val, pct)
        preds = (scores_val >= thr).astype(int)
        if yv.sum() > 0:
            f = f1_score(yv, preds, zero_division=0)
            if f > best_f1:
                best_f1, best_thr = f, thr

    # Score test
    t0 = time.time()
    scores_test = detector.score(Xte3)
    lat = (time.time() - t0) * 1000 / max(len(Xte3), 1)

    y_pred = (scores_test >= best_thr).astype(int)

    cm = confusion_matrix(yt, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        'name':       name,
        'f1':         f1_score(yt, y_pred, zero_division=0),
        'auc_roc':    roc_auc_score(yt, scores_test) if yt.sum() > 0 else 0.0,
        'auc_pr':     average_precision_score(yt, scores_test) if yt.sum() > 0 else 0.0,
        'precision':  precision_score(yt, y_pred, zero_division=0),
        'recall':     recall_score(yt, y_pred, zero_division=0),
        'accuracy':   accuracy_score(yt, y_pred),
        'fpr':        fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'fnr':        fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        'train_time': train_time,
        'latency_ms': lat,
        'threshold':  best_thr,
    }
    print(f' F1={metrics["f1"]:.4f} AUC={metrics["auc_roc"]:.4f} '
          f'Train={train_time:.2f}s Lat={lat:.3f}ms')

    # Expand scores back to original test length for consistency
    full_scores = np.full(len(y_test), np.nan)
    offset = len(y_test) - len(scores_test)
    full_scores[offset:] = scores_test
    # fill leading NaN with min score
    if np.any(np.isnan(full_scores)):
        full_scores = np.where(np.isnan(full_scores), np.nanmin(full_scores), full_scores)

    full_pred = np.zeros(len(y_test), dtype=int)
    full_pred[offset:] = y_pred

    return metrics, full_pred, full_scores


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Per-fault analysis
# ─────────────────────────────────────────────────────────────────────────────
def per_fault_analysis(y_pred_all, y_test, ft_test):
    fault_types = [ft for ft in np.unique(ft_test) if ft != 'normal']
    rows = []
    for name, y_pred in y_pred_all.items():
        for ft in fault_types:
            mask = (ft_test == ft)
            if mask.sum() == 0:
                continue
            yf    = y_test[mask]
            yp    = y_pred[mask]
            det   = int(yp[yf == 1].sum()) if yf.sum() > 0 else 0
            total = int(yf.sum())
            rows.append({
                'detector':   name,
                'fault_type': ft,
                'f1':         f1_score(yf, yp, zero_division=0),
                'detected':   det,
                'total':      total,
                'rate':       det / total if total > 0 else 0,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Visualize results
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(results_df, per_fault_df, y_test, scores_all):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    det_names = results_df['name'].tolist()
    colors    = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # ── Fig 1: Overall Metrics Bar Chart ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('AIOps 2020: Overall Detector Performance', fontsize=14, fontweight='bold')
    for ax, metric, title in zip(axes,
            ['f1', 'auc_roc', 'precision'],
            ['F1 Score', 'AUC-ROC', 'Precision']):
        vals = results_df[metric].tolist()
        bars = ax.bar(det_names, vals, color=colors[:len(det_names)])
        ax.set_title(title); ax.set_ylim(0, 1.15)
        ax.set_xticklabels(det_names, rotation=20, ha='right')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'aiops_fig1_overall_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved fig1_overall_metrics')

    # ── Fig 2: Per-fault F1 Heatmap ───────────────────────────────────────────
    if not per_fault_df.empty:
        pivot = per_fault_df.pivot(index='detector', columns='fault_type', values='f1').fillna(0)
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=25, ha='right')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        plt.colorbar(im, ax=ax, label='F1 Score')
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f'{pivot.values[i,j]:.2f}',
                        ha='center', va='center', fontsize=8,
                        color='black' if pivot.values[i,j] > 0.3 else 'white')
        ax.set_title('AIOps 2020: Per-Fault F1 Score Heatmap', fontweight='bold')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'aiops_fig2_per_fault_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print('  Saved fig2_per_fault_heatmap')

    # ── Fig 3: ROC Curves ─────────────────────────────────────────────────────
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, scores) in enumerate(scores_all.items()):
        if y_test.sum() > 0:
            # Replace any NaN
            scores_clean = np.nan_to_num(scores, nan=float(np.nanmin(scores)))
            fpr_c, tpr_c, _ = roc_curve(y_test, scores_clean)
            auc = roc_auc_score(y_test, scores_clean)
            ax.plot(fpr_c, tpr_c, color=colors[i % len(colors)],
                    label=f'{name} (AUC={auc:.3f})', lw=2)
    ax.plot([0,1],[0,1],'k--', lw=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('AIOps 2020: ROC Curves', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'aiops_fig3_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved fig3_roc_curves')

    # ── Fig 4: FPR vs FNR scatter ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, row in results_df.iterrows():
        ax.scatter(row['fpr'], row['fnr'], s=150, color=colors[i % len(colors)], zorder=5)
        ax.annotate(row['name'], (row['fpr'], row['fnr']),
                    textcoords='offset points', xytext=(6, 4), fontsize=9)
    ax.set_xlabel('FPR (False Positive Rate)'); ax.set_ylabel('FNR (False Negative Rate)')
    ax.set_title('AIOps 2020: FPR vs FNR Operating Points', fontweight='bold')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.1, color='gray', ls='--', lw=0.8)
    ax.axvline(0.1, color='gray', ls='--', lw=0.8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'aiops_fig4_fpr_fnr.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved fig4_fpr_fnr')

    # ── Fig 5: Training Time ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(det_names, results_df['train_time'].tolist(), color=colors[:len(det_names)])
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('AIOps 2020: Detector Training Time', fontweight='bold')
    ax.set_xticklabels(det_names, rotation=20, ha='right')
    for bar, v in zip(bars, results_df['train_time']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{v:.2f}s', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'aiops_fig5_training_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved fig5_training_time')

    print(f'\n  All plots saved to: {PLOTS_DIR}')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print('=' * 65)
    print('  AIOps Challenge 2020 — Anomaly Detector Benchmark')
    print('  Authors: ANUSHREE S (23BEC007) & SAINATH (23BCS115)')
    print('=' * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    print('\n[1] Loading AIOps 2020 dataset...')
    combined = load_dataset()

    print('\n[2] Preparing train/val/test splits...')
    (X_train, X_val, X_test,
     y_val,   y_test,
     ft_val,  ft_test,
     feat_cols) = prepare_train_test(combined)

    n_feat = X_train.shape[1]
    print(f'\n    Features:         {n_feat}')
    print(f'    Train windows:    {len(X_train)} (all normal)')
    print(f'    Val windows:      {len(X_val)} ({y_val.sum()} anomalous)')
    print(f'    Test windows:     {len(X_test)} ({y_test.sum()} anomalous)')
    print(f'    Fault types test: {np.unique(ft_test[ft_test!="normal"])}')

    # ── Global scaler (fit on train only) ─────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # ─────────────────────────────────────────────────────────────────────────
    # Tabular detectors — receive pre-scaled 2D arrays
    # NOTE: BaseDetector.fit() ALSO scales internally via self.scaler.
    # To avoid double-scaling we pass raw (unscaled) arrays to fit/score.
    # They scale themselves; we only pre-scale for the threshold search loop.
    # Actually simpler: pass raw arrays; detectors scale themselves. ──────────
    # ─────────────────────────────────────────────────────────────────────────

    tabular_detectors = {
        'IF':       IsolationForestDetector(n_estimators=200, contamination=0.1, random_state=SEED),
        'LOF':      LOFDetector(n_neighbors=20, contamination=0.1),
        'OCSVM':    OCSVMDetector(nu=0.1, kernel='rbf', gamma='scale'),
        'AE':       AutoencoderDetector(epochs=50, batch_size=32),
        'Ensemble': EnsembleDetector(),
    }

    # ── Run tabular detectors ─────────────────────────────────────────────────
    print('\n[3] Running tabular detectors on AIOps data...')
    all_results = []
    y_pred_all  = {}
    scores_all  = {}

    for name, det in tabular_detectors.items():
        # Pass raw (unscaled) arrays — detectors have internal scalers
        metrics, y_pred, scores = evaluate_detector(
            name, det, X_train, X_val, X_test, y_val, y_test
        )
        all_results.append(metrics)
        y_pred_all[name]  = y_pred
        scores_all[name]  = scores

    # ── Run LSTM-AE separately (needs 3D input) ───────────────────────────────
    print('\n[4] Running LSTM-AE detector...')
    lstm_det = LSTMAutoencoderDetector(hidden_size=32, num_layers=2,
                                       epochs=20, batch_size=64,
                                       random_state=SEED)
    metrics_lstm, y_pred_lstm, scores_lstm = evaluate_lstm(
        'LSTM-AE', lstm_det,
        X_train, X_val, X_test,
        y_val, y_test, window_size=5
    )
    all_results.append(metrics_lstm)
    y_pred_all['LSTM-AE']  = y_pred_lstm
    scores_all['LSTM-AE']  = scores_lstm

    results_df = pd.DataFrame(all_results)

    # ── Per-fault analysis ────────────────────────────────────────────────────
    print('\n[5] Per-fault analysis...')
    per_fault_df = per_fault_analysis(y_pred_all, y_test, ft_test)

    # ── Save results ──────────────────────────────────────────────────────────
    print('\n[6] Saving results...')
    results_df.to_csv(RESULTS_DIR / 'aiops_overall_metrics.csv', index=False)
    per_fault_df.to_csv(RESULTS_DIR / 'aiops_per_fault_metrics.csv', index=False)
    print('  Saved aiops_overall_metrics.csv')
    print('  Saved aiops_per_fault_metrics.csv')

    # ── Print results table ───────────────────────────────────────────────────
    print('\n' + '=' * 75)
    print('  FINAL RESULTS ON AIOps CHALLENGE 2020 DATASET')
    print('=' * 75)
    print(f"{'Detector':<12} {'F1':>7} {'AUC-ROC':>8} {'Precision':>10} "
          f"{'Recall':>8} {'FPR':>7} {'Train(s)':>9} {'Lat(ms)':>8}")
    print('-' * 75)
    for _, r in results_df.sort_values('f1', ascending=False).iterrows():
        print(f"{r['name']:<12} {r['f1']:>7.4f} {r['auc_roc']:>8.4f} "
              f"{r['precision']:>10.4f} {r['recall']:>8.4f} {r['fpr']:>7.4f} "
              f"{r['train_time']:>9.2f} {r['latency_ms']:>8.3f}")

    if not per_fault_df.empty:
        print('\n  Per-Fault F1 Summary:')
        piv = per_fault_df.pivot(index='detector', columns='fault_type', values='f1')
        print(piv.round(3).to_string())

    # ── Visualize ─────────────────────────────────────────────────────────────
    print('\n[7] Generating plots...')
    plot_results(results_df, per_fault_df, y_test, scores_all)

    print('\n' + '=' * 65)
    best = results_df.loc[results_df['f1'].idxmax()]
    print(f'  BEST DETECTOR: {best["name"]} (F1={best["f1"]:.4f}, AUC={best["auc_roc"]:.4f})')
    print('  Dataset: AIOps Challenge 2020 (Real Production Data)')
    print(f'  Total windows: {len(combined)} | Anomalous: {combined["label"].sum()}')
    print('=' * 65)


if __name__ == '__main__':
    main()
