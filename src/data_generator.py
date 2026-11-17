"""
data_generator.py
=================
Synthetic microservice telemetry data generator with realistic fault
and attack injection patterns.

Simulates a 10-service microservice application (inspired by Sock Shop /
Online Boutique) and generates CPU, memory, latency, error-rate, and
network I/O metrics under:
  - Normal operation
  - Fault injection  : CPU stress, memory leak, pod failure, network delay
  - Attack injection : DDoS, cryptomining, privilege escalation, SQL injection
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
import os

# ─────────────────────────── Configuration ──────────────────────────────────

SERVICES = [
    "frontend", "user-service", "product-service", "cart-service",
    "order-service", "payment-service", "shipping-service",
    "recommendation-service", "search-service", "notification-service"
]

FAULT_TYPES = {
    "normal":               0,
    "cpu_stress":           1,
    "memory_leak":          2,
    "pod_failure":          3,
    "network_delay":        4,
    "ddos_attack":          5,
    "cryptomining":         6,
    "privilege_escalation": 7,
    "sql_injection":        8,
}

LABEL_NAMES = {v: k for k, v in FAULT_TYPES.items()}

# ─────────────────────────── Base Signal Generator ──────────────────────────

class MetricProfile:
    """Baseline statistical profile for a microservice."""
    def __init__(self, name: str, rng: np.random.Generator):
        self.name = name
        # Service-specific baseline noise
        seed_val = sum(ord(c) for c in name)
        self.cpu_base    = rng.uniform(10, 30)
        self.mem_base    = rng.uniform(200, 600)   # MB
        self.lat_base    = rng.uniform(20, 150)    # ms
        self.err_base    = rng.uniform(0.001, 0.02)
        self.net_in_base = rng.uniform(1, 10)      # MB/s
        self.net_out_base= rng.uniform(0.5, 8)


def generate_normal(profile: MetricProfile, n: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Generate n timesteps of normal multivariate metric data."""
    t = np.arange(n)
    # Diurnal pattern (24-hour cycle) on a 1-min resolution
    diurnal = np.sin(2 * np.pi * t / 1440) * 0.15 + 1.0

    cpu    = profile.cpu_base    * diurnal + rng.normal(0, 2.0, n)
    mem    = profile.mem_base    * diurnal + rng.normal(0, 15,  n)
    lat    = profile.lat_base    * diurnal + rng.normal(0, 8,   n)
    err    = profile.err_base    + rng.uniform(0, 0.005, n)
    net_in = profile.net_in_base * diurnal + rng.normal(0, 0.5, n)
    net_out= profile.net_out_base* diurnal + rng.normal(0, 0.3, n)

    # Clip to realistic ranges
    cpu    = np.clip(cpu,  0, 100)
    mem    = np.clip(mem,  50, 8192)
    lat    = np.clip(lat,  1, 5000)
    err    = np.clip(err,  0, 1)
    net_in = np.clip(net_in,  0, 1000)
    net_out= np.clip(net_out, 0, 1000)

    return np.stack([cpu, mem, lat, err, net_in, net_out], axis=1)


# ─────────────────────────── Anomaly Injectors ──────────────────────────────

def inject_cpu_stress(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """CPU spikes to 85–99%, memory slightly elevated."""
    d = data.copy()
    spike = rng.uniform(85, 99, len(d))
    d[:, 0] = np.clip(d[:, 0] + spike, 0, 100)
    d[:, 1] = np.clip(d[:, 1] * rng.uniform(1.1, 1.3, len(d)), 0, 8192)
    d[:, 2] = np.clip(d[:, 2] * rng.uniform(1.5, 3.0, len(d)), 0, 10000)
    return d


def inject_memory_leak(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gradual memory growth, latency degradation."""
    d = data.copy()
    growth = np.linspace(1.0, 4.5, len(d))
    d[:, 1] = np.clip(d[:, 1] * growth, 0, 8192)
    d[:, 2] = np.clip(d[:, 2] * growth * 0.8, 0, 10000)
    d[:, 0] = np.clip(d[:, 0] + growth * 5, 0, 100)
    return d


def inject_pod_failure(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pod restarts: sudden drops + spikes in error rate and latency."""
    d = data.copy()
    n = len(d)
    failure_times = rng.integers(0, n, size=rng.integers(2, 6))
    for ft in failure_times:
        window = min(10, n - ft)
        d[ft:ft+window, 2] = np.clip(d[ft:ft+window, 2] * 10, 0, 10000)
        d[ft:ft+window, 3] = np.clip(d[ft:ft+window, 3] + 0.5, 0, 1)
        d[ft:ft+window, 4] = 0  # network drops to 0
        d[ft:ft+window, 5] = 0
    return d


def inject_network_delay(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Network latency spikes, elevated error rate."""
    d = data.copy()
    d[:, 2] = np.clip(d[:, 2] * rng.uniform(5, 20, len(d)), 0, 10000)
    d[:, 3] = np.clip(d[:, 3] + rng.uniform(0.05, 0.3, len(d)), 0, 1)
    d[:, 4] = np.clip(d[:, 4] * rng.uniform(0.1, 0.4, len(d)), 0, 1000)
    return d


def inject_ddos(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """DDoS: massive spike in network in, CPU, error rate, latency."""
    d = data.copy()
    d[:, 0] = np.clip(d[:, 0] + rng.uniform(40, 70, len(d)), 0, 100)
    d[:, 2] = np.clip(d[:, 2] * rng.uniform(10, 50, len(d)), 0, 50000)
    d[:, 3] = np.clip(d[:, 3] + rng.uniform(0.3, 0.9, len(d)), 0, 1)
    d[:, 4] = np.clip(d[:, 4] * rng.uniform(10, 100, len(d)), 0, 10000)
    return d


def inject_cryptomining(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cryptomining: sustained high CPU, steady net out, memory stable."""
    d = data.copy()
    d[:, 0] = np.clip(d[:, 0] + rng.uniform(55, 75, len(d)), 0, 100)
    d[:, 5] = np.clip(d[:, 5] * rng.uniform(2, 5, len(d)), 0, 1000)
    # Stealth: minimal effect on latency and error rate
    d[:, 2] = np.clip(d[:, 2] * rng.uniform(1.1, 1.5, len(d)), 0, 10000)
    return d


def inject_privilege_escalation(data: np.ndarray,
                                 rng: np.random.Generator) -> np.ndarray:
    """Privilege escalation: anomalous auth calls, brief CPU/net spikes."""
    d = data.copy()
    n = len(d)
    # Intermittent suspicious probes
    probe_times = rng.integers(0, n, size=rng.integers(5, 15))
    for pt in probe_times:
        window = min(3, n - pt)
        d[pt:pt+window, 0] = np.clip(d[pt:pt+window, 0] + 30, 0, 100)
        d[pt:pt+window, 5] = np.clip(d[pt:pt+window, 5] * 5, 0, 1000)
        d[pt:pt+window, 3] = np.clip(d[pt:pt+window, 3] + 0.1, 0, 1)
    return d


def inject_sql_injection(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """SQL injection: bursts of DB queries, errors, latency degradation."""
    d = data.copy()
    n = len(d)
    bursts = rng.integers(0, n, size=rng.integers(3, 10))
    for b in bursts:
        window = min(8, n - b)
        d[b:b+window, 2] = np.clip(d[b:b+window, 2] * rng.uniform(3, 8, window), 0, 10000)
        d[b:b+window, 3] = np.clip(d[b:b+window, 3] + rng.uniform(0.1, 0.5, window), 0, 1)
        d[b:b+window, 1] = np.clip(d[b:b+window, 1] * rng.uniform(1.2, 2.0, window), 0, 8192)
    return d


INJECTORS = {
    "cpu_stress":           inject_cpu_stress,
    "memory_leak":          inject_memory_leak,
    "pod_failure":          inject_pod_failure,
    "network_delay":        inject_network_delay,
    "ddos_attack":          inject_ddos,
    "cryptomining":         inject_cryptomining,
    "privilege_escalation": inject_privilege_escalation,
    "sql_injection":        inject_sql_injection,
}

# ─────────────────────────── Dataset Builder ────────────────────────────────

def build_dataset(n_normal: int = 5000,
                  n_anomaly_per_type: int = 400,
                  seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a labelled dataset of microservice telemetry.

    Returns
    -------
    df : pd.DataFrame  – feature matrix
    labels : pd.Series – integer labels (0 = normal, 1-8 = anomaly types)
    """
    rng = np.random.default_rng(seed)
    feature_names = ["cpu_%", "mem_mb", "latency_ms", "error_rate",
                     "net_in_mbps", "net_out_mbps"]
    service_feature_names = []
    for svc in SERVICES:
        for feat in feature_names:
            service_feature_names.append(f"{svc}_{feat}")

    rows, label_list = [], []

    # ── Normal data ──
    profiles = {svc: MetricProfile(svc, rng) for svc in SERVICES}
    for _ in range(n_normal):
        row = []
        for svc in SERVICES:
            row.extend(generate_normal(profiles[svc], 1, rng)[0].tolist())
        rows.append(row)
        label_list.append(FAULT_TYPES["normal"])

    # ── Anomalous data ──
    for fault_name, injector in INJECTORS.items():
        fault_label = FAULT_TYPES[fault_name]
        for _ in range(n_anomaly_per_type):
            row = []
            # Pick 1–4 services to inject the fault into
            n_affected = rng.integers(1, 5)
            affected = rng.choice(SERVICES, size=n_affected, replace=False)
            for svc in SERVICES:
                base = generate_normal(profiles[svc], 1, rng)
                if svc in affected:
                    base = injector(base, rng)
                row.extend(base[0].tolist())
            rows.append(row)
            label_list.append(fault_label)

    df = pd.DataFrame(rows, columns=service_feature_names)
    labels = pd.Series(label_list, name="label")
    return df, labels


def build_windowed_dataset(window_size: int = 30,
                            n_windows_normal: int = 300,
                            n_windows_per_fault: int = 50,
                            seed: int = 42):
    """
    Build a time-windowed version for LSTM/sequence models.
    Returns X (N, window_size, n_features), y (N,) for one representative service.
    """
    rng = np.random.default_rng(seed)
    n_features = 6
    feature_names = ["cpu_%", "mem_mb", "latency_ms",
                     "error_rate", "net_in_mbps", "net_out_mbps"]

    profile = MetricProfile("frontend", rng)
    X_list, y_list = [], []

    # Normal windows
    for _ in range(n_windows_normal):
        win = generate_normal(profile, window_size, rng)
        X_list.append(win)
        y_list.append(0)

    # Anomalous windows
    for fault_name, injector in INJECTORS.items():
        fault_label = FAULT_TYPES[fault_name]
        for _ in range(n_windows_per_fault):
            base = generate_normal(profile, window_size, rng)
            win  = injector(base, rng)
            X_list.append(win)
            y_list.append(fault_label)

    X = np.stack(X_list)   # (N, window_size, n_features)
    y = np.array(y_list)
    return X, y, feature_names


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    print("Building tabular dataset …")
    df, labels = build_dataset()
    df["label"] = labels
    df["label_name"] = labels.map(LABEL_NAMES)
    df.to_csv(os.path.join(out_dir, "telemetry_tabular.csv"), index=False)
    print(f"  Saved {len(df):,} rows × {df.shape[1]} cols")
    print(df["label_name"].value_counts())

    print("\nBuilding windowed dataset …")
    X, y, feat_names = build_windowed_dataset()
    np.save(os.path.join(out_dir, "telemetry_X_windowed.npy"), X)
    np.save(os.path.join(out_dir, "telemetry_y_windowed.npy"), y)
    print(f"  Saved X: {X.shape}, y: {y.shape}")
