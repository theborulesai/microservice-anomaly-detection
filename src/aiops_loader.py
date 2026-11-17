"""
aiops_loader.py — Load and preprocess the AIOps Challenge 2020 dataset.

Data structure per day:
  平台指标/  → db_oracle_11g.csv, dcos_container.csv, dcos_docker.csv,
              mw_redis.csv, os_linux.csv
  业务指标/  → esb.csv
  调用链指标/→ trace_csf.csv, trace_fly_remote.csv, trace_jdbc.csv,
              trace_local.csv, trace_osb.csv, trace_remote_process.csv
Ground truth: 故障整理（预赛）.csv

Output: feature matrix X (n_samples × n_features) + labels y
"""
import os, glob, re
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR  = Path(r'C:\Users\Sainath S Borule\Downloads\microservice-anomaly-detection')
DATA_DIR  = BASE_DIR / 'data'
GT_CSV    = next((BASE_DIR / 'data' / 'outer').glob('*.csv'), None)   # 故障整理（预赛）.csv

WINDOW_SEC = 300          # 5-minute aggregation window (matches fault duration)
SEED       = 42

# ─── Fault-type mapping ───────────────────────────────────────────────────────
FAULT_MAP = {
    'CPU fault':            'cpu_fault',
    'network delay':        'network_delay',
    'network loss':         'network_loss',
    'db connection limit':  'db_connection_limit',
    'db  close':            'db_close',
    'db close':             'db_close',
}

# ─── Load ground truth ────────────────────────────────────────────────────────
def load_ground_truth() -> pd.DataFrame:
    """Return DataFrame with columns: start_ts (unix ms), end_ts, fault_type, cmdb_id.

    The dataset has two kinds of fault records:
    - Original April 11 faults: use log_time (no start_time)
    - Replayed faults in later days: use start_time column

    All times are in China Standard Time (CST = UTC+8).
    Data timestamps are in unix ms (also CST-based).
    So we parse datetimes as naive CST and convert: subtract 8h to get UTC ms,
    but since the platform timestamps are ALSO naive CST-based ms, we keep same basis.
    """
    df = pd.read_csv(GT_CSV, encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]

    rows = []
    for _, r in df.iterrows():
        # Use start_time if available (replayed), else log_time (original)
        time_str = r.get('start_time') if pd.notna(r.get('start_time', None)) \
                   else r.get('log_time', None)
        if pd.isna(time_str):
            continue
        # Parse as CST naive datetime → unix ms (CST epoch, same as data timestamps)
        try:
            dt = pd.to_datetime(str(time_str).strip(), format='%Y/%m/%d %H:%M')
        except Exception:
            try:
                dt = pd.to_datetime(str(time_str).strip())
            except Exception:
                continue
        # GT times are written as CST (UTC+8) naive strings.
        # Data timestamps are stored as UTC unix-ms.
        # → subtract 8 hours (28800 s) to convert CST→UTC ms.
        ts_ms = int(dt.timestamp() * 1000) - 8 * 3600 * 1000
        # Fault window: 5 min duration
        rows.append({
            'start_ts':   ts_ms,
            'end_ts':     ts_ms + 5 * 60 * 1000,
            'fault_type': FAULT_MAP.get(str(r.get('fault_desrcibtion', '')).strip(),
                                         str(r.get('fault_desrcibtion', '')).strip()),
            'cmdb_id':    r.get('container', r.get('name', ''))
        })
    result = pd.DataFrame(rows)
    return result.dropna(subset=['start_ts'])


# ─── Platform metric loaders ──────────────────────────────────────────────────
def _load_kpi_csv(path: str, metric_col='name', val_col='value',
                  ts_col='timestamp', id_col='cmdb_id') -> pd.DataFrame:
    """Generic KPI CSV loader → long format (timestamp, cmdb_id, metric, value)."""
    try:
        df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
        df = df[[ts_col, id_col, metric_col, val_col]].copy()
        df.columns = ['timestamp', 'cmdb_id', 'metric', 'value']
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['value'])
        return df
    except Exception as e:
        return pd.DataFrame(columns=['timestamp', 'cmdb_id', 'metric', 'value'])


def _load_esb(path: str) -> pd.DataFrame:
    """Business KPI: esb.csv → success_rate, avg_time, num per service per window."""
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
        df['timestamp'] = pd.to_numeric(df['startTime'], errors='coerce')
        rows = []
        for _, r in df.iterrows():
            svc = str(r.get('serviceName', 'esb'))
            ts  = r['timestamp']
            rows.append({'timestamp': ts, 'cmdb_id': svc, 'metric': 'success_rate',
                         'value': float(r.get('succee_rate', 1.0))})
            rows.append({'timestamp': ts, 'cmdb_id': svc, 'metric': 'avg_time',
                         'value': float(r.get('avg_time', 0.0))})
            rows.append({'timestamp': ts, 'cmdb_id': svc, 'metric': 'num_requests',
                         'value': float(r.get('num', 0))})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=['timestamp', 'cmdb_id', 'metric', 'value'])


def _load_traces(path: str) -> pd.DataFrame:
    """Trace CSV → error_rate, avg_elapsed per cmdb_id per timestamp."""
    try:
        df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
        df['timestamp'] = pd.to_numeric(df['startTime'], errors='coerce')
        df['elapsed']   = pd.to_numeric(df['elapsedTime'], errors='coerce')
        df['success']   = df['success'].astype(str).str.lower().map(
            {'true': 1, 'false': 0, '1': 1, '0': 0}
        ).fillna(1)
        df['cmdb_id'] = df['cmdb_id'].fillna('unknown')
        # Aggregate per (timestamp-window, cmdb_id)
        df['win'] = (df['timestamp'] // (WINDOW_SEC * 1000)) * (WINDOW_SEC * 1000)
        grp = df.groupby(['win', 'cmdb_id']).agg(
            avg_elapsed=('elapsed', 'mean'),
            error_rate=('success', lambda x: 1 - x.mean()),
            call_count=('elapsed', 'count')
        ).reset_index()
        rows = []
        for _, r in grp.iterrows():
            base = {'timestamp': r['win'], 'cmdb_id': r['cmdb_id']}
            rows.append({**base, 'metric': 'trace_avg_elapsed', 'value': r['avg_elapsed']})
            rows.append({**base, 'metric': 'trace_error_rate',  'value': r['error_rate']})
            rows.append({**base, 'metric': 'trace_call_count',  'value': r['call_count']})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=['timestamp', 'cmdb_id', 'metric', 'value'])


# ─── Load one day's data ──────────────────────────────────────────────────────
def load_day(day_dir: Path) -> pd.DataFrame:
    """Load all CSVs for one day → unified long DataFrame."""
    sub = list(day_dir.iterdir())
    # Find the actual day subfolder (e.g. 2020_04_11/2020_04_11/)
    inner = [d for d in sub if d.is_dir()]
    base  = inner[0] if inner else day_dir

    # Find platform/business/trace sub-folders by content
    all_csvs = list(base.rglob('*.csv'))
    frames = []

    for csv in all_csvs:
        name = csv.name.lower()
        if name == 'esb.csv':
            frames.append(_load_esb(str(csv)))
        elif name in ('db_oracle_11g.csv', 'dcos_container.csv',
                      'dcos_docker.csv', 'mw_redis.csv', 'os_linux.csv'):
            frames.append(_load_kpi_csv(str(csv)))
        elif name.startswith('trace_'):
            frames.append(_load_traces(str(csv)))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ─── Window aggregation ───────────────────────────────────────────────────────
def aggregate_to_windows(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long → wide: one row per 5-min window.
    Features = (cmdb_id × metric) pivoted, then aggregated across cmdb_ids.
    """
    if long_df.empty:
        return pd.DataFrame()

    long_df['timestamp'] = pd.to_numeric(long_df['timestamp'], errors='coerce')
    long_df = long_df.dropna(subset=['timestamp'])
    long_df['win'] = (long_df['timestamp'] // (WINDOW_SEC * 1000)) * (WINDOW_SEC * 1000)

    # Aggregate value per (win, cmdb_id, metric)
    grp = long_df.groupby(['win', 'cmdb_id', 'metric'])['value'].mean().reset_index()

    # Pivot: columns = metric (aggregated across all cmdb_ids per window)
    pivot = grp.groupby(['win', 'metric'])['value'].agg(['mean', 'max', 'std']).reset_index()
    pivot.columns = ['win', 'metric', 'mean_val', 'max_val', 'std_val']

    wide = pivot.pivot(index='win', columns='metric',
                       values=['mean_val', 'max_val', 'std_val'])
    wide.columns = [f'{agg}__{met}' for agg, met in wide.columns]
    wide = wide.reset_index().rename(columns={'win': 'timestamp'})
    wide = wide.fillna(0)
    return wide


# ─── Label assignment ─────────────────────────────────────────────────────────
def assign_labels(wide_df: pd.DataFrame, gt_df: pd.DataFrame):
    """
    For each time window, label=1 if any fault overlaps the window.
    Returns (wide_df with 'label' column, fault_type column).
    """
    wide_df = wide_df.copy()
    wide_df['label']      = 0
    wide_df['fault_type'] = 'normal'

    win_ms = WINDOW_SEC * 1000
    for _, fault in gt_df.iterrows():
        mask = (
            (wide_df['timestamp'] >= fault['start_ts']) &
            (wide_df['timestamp'] <  fault['end_ts'] + win_ms)
        )
        wide_df.loc[mask, 'label']      = 1
        wide_df.loc[mask, 'fault_type'] = fault['fault_type']

    return wide_df


# ─── Main loader ──────────────────────────────────────────────────────────────
def load_all_days(verbose=True) -> pd.DataFrame:
    """Load all extracted days → combined labeled feature DataFrame."""
    day_dirs = sorted([d for d in DATA_DIR.iterdir()
                       if d.is_dir() and re.match(r'2020_\d{2}_\d{2}', d.name)])

    if verbose:
        print(f'[AIOps Loader] Found {len(day_dirs)} day directories')

    gt_df = load_ground_truth()
    if verbose:
        print(f'[AIOps Loader] Ground truth: {len(gt_df)} fault events')

    all_days = []
    for day_dir in day_dirs:
        if verbose:
            print(f'  Loading {day_dir.name}...', end='', flush=True)
        long_df = load_day(day_dir)
        if long_df.empty:
            if verbose: print(' EMPTY')
            continue
        wide_df = aggregate_to_windows(long_df)
        if wide_df.empty:
            if verbose: print(' EMPTY after aggregation')
            continue
        labeled = assign_labels(wide_df, gt_df)
        all_days.append(labeled)
        n_anom = labeled['label'].sum()
        if verbose:
            print(f' {len(labeled)} windows ({n_anom} anomalous)')

    if not all_days:
        raise RuntimeError('No data loaded! Check extraction.')

    combined = pd.concat(all_days, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    if verbose:
        total    = len(combined)
        n_anom   = combined['label'].sum()
        n_feat   = combined.shape[1] - 3   # minus timestamp, label, fault_type
        print(f'\n[AIOps Loader] Total: {total} windows, {n_anom} anomalous '
              f'({100*n_anom/total:.1f}%), {n_feat} features')
        print(f'[AIOps Loader] Fault types: {combined["fault_type"].value_counts().to_dict()}')

    return combined


def prepare_train_test(combined_df: pd.DataFrame, train_ratio=0.5, val_ratio=0.2):
    """
    Split into train (normal only), val (mixed), test (mixed).
    Returns X_train, X_val, X_test, y_val, y_test, fault_val, fault_test, feature_cols.
    """
    feature_cols = [c for c in combined_df.columns
                    if c not in ('timestamp', 'label', 'fault_type')]

    # Sort chronologically for time-series integrity
    df = combined_df.sort_values('timestamp').reset_index(drop=True)

    n = len(df)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train:n_train + n_val]
    test_df  = df.iloc[n_train + n_val:]

    # Training: use ONLY normal windows (unsupervised protocol)
    train_normal = train_df[train_df['label'] == 0]

    X_train = train_normal[feature_cols].values.astype(np.float32)
    X_val   = val_df[feature_cols].values.astype(np.float32)
    X_test  = test_df[feature_cols].values.astype(np.float32)
    y_val   = val_df['label'].values
    y_test  = test_df['label'].values
    ft_val  = val_df['fault_type'].values
    ft_test = test_df['fault_type'].values

    # Remove NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val   = np.nan_to_num(X_val,   nan=0.0, posinf=0.0, neginf=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)

    print(f'  Train (normal only): {len(X_train)} windows')
    print(f'  Val  (mixed):        {len(X_val)} windows  ({y_val.sum()} anomalous)')
    print(f'  Test (mixed):        {len(X_test)} windows ({y_test.sum()} anomalous)')
    print(f'  Feature dimensions: {X_train.shape[1]}')

    return X_train, X_val, X_test, y_val, y_test, ft_val, ft_test, feature_cols


if __name__ == '__main__':
    combined = load_all_days(verbose=True)
    combined.to_csv(BASE_DIR / 'data' / 'aiops_combined.csv', index=False, encoding='utf-8-sig')
    print('Saved aiops_combined.csv')
    X_train, X_val, X_test, y_val, y_test, ft_val, ft_test, fcols = prepare_train_test(combined)
    print(f'Ready: X_train={X_train.shape}, X_test={X_test.shape}')
