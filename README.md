# Microservice Anomaly Detection on AIOps Challenge 2020

> **Authors:** ANUSHREE S (23BEC007) · SAINATH S (23BCS115)

Unsupervised anomaly detection benchmark using real production telemetry from the **AIOps Challenge 2020** dataset (Zhejiang Mobile). Six detectors evaluated across five fault types on 15 days of microservice data.

---

## 📊 Results Summary

| Detector | F1 | AUC-ROC | Recall | Train Time |
|----------|-----|---------|--------|-----------|
| **OCSVM** | **0.2960** | **0.4878** | 0.6406 | 0.01s |
| AE | 0.2945 | 0.4353 | **0.7500** | 9.13s |
| Ensemble | 0.2941 | 0.4534 | 0.7031 | 0.63s |
| IF | 0.2836 | 0.4559 | 0.5938 | 0.22s |
| LSTM-AE | 0.2080 | 0.3790 | 0.4194 | 1.22s |
| LOF | 0.1887 | 0.4444 | 0.2344 | 0.05s |

### Per-Fault F1 Scores

| Detector | cpu_fault | db_conn_limit | network_delay | network_loss |
|----------|-----------|---------------|---------------|-------------|
| **AE** | **0.909** | **0.909** | 0.833 | **0.839** |
| Ensemble | **0.909** | **0.909** | 0.809 | 0.759 |
| OCSVM | 0.800 | **0.909** | **0.809** | 0.667 |
| IF | 0.737 | **0.909** | 0.727 | 0.714 |
| LSTM-AE | 0.800 | 0.500 | 0.400 | 0.667 |
| LOF | 0.286 | 0.667 | 0.353 | 0.364 |

---

## 📁 Project Structure

```
microservice-anomaly-detection/
│
├── src/
│   ├── aiops_loader.py           # Data loader for AIOps 2020 dataset
│   ├── aiops_main_experiment.py  # Main experiment — runs all 6 detectors
│   ├── detectors.py              # IF, LOF, OCSVM, AE, LSTM-AE, Ensemble
│   ├── extract_data.py           # ZIP extraction utility
│   └── ...
│
├── plots/
│   ├── aiops_fig1_overall_metrics.png
│   ├── aiops_fig2_per_fault_heatmap.png
│   ├── aiops_fig3_roc_curves.png
│   ├── aiops_fig4_fpr_fnr.png
│   └── aiops_fig5_training_time.png
│
├── results/
│   ├── aiops_overall_metrics.csv
│   └── aiops_per_fault_metrics.csv
│
├── overleaf_upload/              # Ready-to-upload Overleaf project
│   ├── RESEARCH_PAPER.tex
│   ├── RESULTS_REPORT.tex
│   └── plots/
│
├── RESEARCH_PAPER.tex            # Full research paper (LaTeX)
├── RESULTS_REPORT.tex            # Experimental results report (LaTeX)
└── requirements.txt
```

---

## 🗂️ Dataset

**AIOps Challenge 2020** — Zhejiang Mobile production microservice telemetry

| Property | Value |
|----------|-------|
| Days | 15 (Apr 11 + Apr 20 – May 31, 2020) |
| Windows (5-min) | 1,081 |
| Anomalous windows | 161 (14.9%) |
| Features | 408 |
| Fault types | 5 |

**Fault types:** `cpu_fault`, `network_delay`, `network_loss`, `db_connection_limit`, `db_close`

> **Note:** The raw dataset (~3 GB) is not included. Download from [AIOps Challenge 2020](https://competition.aiops-challenge.com).

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install scikit-learn==1.5.2 numpy pandas matplotlib scipy torch
```

### 2. Place dataset
Extract `AIOps挑战赛2020预赛数据.zip` into the project root.

### 3. Run the experiment
```bash
cd microservice-anomaly-detection
python src/aiops_main_experiment.py
```

Results saved to `results/` and `plots/`.

---

## 🔬 Detectors

| # | Detector | Type | Key Idea |
|---|----------|------|----------|
| 1 | **Isolation Forest** | Tree ensemble | Anomalies isolated in fewer splits |
| 2 | **Local Outlier Factor** | Density-based | Compares local density to neighbours |
| 3 | **One-Class SVM** | Kernel boundary | Min-volume sphere around normal data |
| 4 | **Autoencoder** | Deep learning | High reconstruction error = anomaly |
| 5 | **LSTM-Autoencoder** | Sequential DL | Temporal sequence reconstruction |
| 6 | **Ensemble** | Rank-averaging | Combines IF + LOF + AE votes |

---

## 📄 Reports

Compile `RESEARCH_PAPER.tex` and `RESULTS_REPORT.tex` on [Overleaf](https://overleaf.com).  
Or upload the `overleaf_upload/` folder directly as a zip.

---

## ⚙️ Environment

- Python 3.11
- scikit-learn 1.5.2
- PyTorch (CPU, with PCA fallback)
- Windows 11

---

## 📜 Citation

```
ANUSHREE S, SAINATH S. Unsupervised Anomaly Detection in Microservice
Architectures: A Comparative Evaluation on the AIOps Challenge 2020 Dataset. 2024.
```
