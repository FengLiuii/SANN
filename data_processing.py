import os
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import torch
from torch import Tensor
from datetime import timedelta, datetime


def load_dataset(
    csv_path: str,
    label_col: str,
    drop_cols: Optional[List[str]] = None,
    group_col: Optional[str] = None,
    features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """
    Load tabular dataset from CSV and return features, labels, and optional group labels.

    Parameters
    - csv_path: CSV file path
    - label_col: name of the label/target column
    - drop_cols: columns to drop from features (e.g., identifiers)
    - group_col: optional column for grouping (e.g., basin/domain)
    """
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV columns: {list(df.columns)}")

    drop_cols = drop_cols or []
    if features is not None:
        # Use explicit feature subset as per original notebook
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns in CSV: {missing}")
        feature_df = df[features].copy()
        # Coerce to numeric and fill NaN as in original code
        feature_df = feature_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    else:
        cols_to_drop = set(drop_cols + [label_col])
        if group_col is not None:
            cols_to_drop.add(group_col)
        feature_df = df.drop(columns=list(cols_to_drop), errors="ignore")
    label_series = df[label_col]
    group_series = df[group_col] if group_col is not None and group_col in df.columns else None
    return feature_df, label_series, group_series


def standardize_features(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values.astype(np.float32))
    return X_scaled, scaler


def build_knn_hyperedges(
    X: np.ndarray,
    k_neighbors: int = 10,
    include_self: bool = True,
) -> List[List[int]]:
    """
    Build hyperedges from KNN neighborhoods. Each node i defines a hyperedge consisting
    of its k nearest neighbors (and itself if include_self=True).

    Returns a list of hyperedges, where each hyperedge is a list of node indices.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D array: [num_nodes, num_features]")

    num_nodes = X.shape[0]
    k = min(k_neighbors, max(1, num_nodes - 1))

    nbrs = NearestNeighbors(n_neighbors=k + (0 if not include_self else 1), algorithm="auto")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)

    hyperedges: List[List[int]] = []
    for node_idx in range(num_nodes):
        neigh_list = indices[node_idx].tolist()
        if not include_self:
            if node_idx in neigh_list:
                neigh_list.remove(node_idx)
        hyperedges.append(sorted(set(neigh_list + ([node_idx] if include_self and node_idx not in neigh_list else []))))

    return hyperedges


def hyperedges_to_index(hyperedges: List[List[int]]) -> Tensor:
    """
    Convert list of hyperedges to hyperedge_index format expected by PyG HypergraphConv.
    hyperedge_index has shape [2, num_entries], rows are [node_index; hyperedge_id].
    """
    node_indices: List[int] = []
    hyperedge_ids: List[int] = []
    for he_id, nodes in enumerate(hyperedges):
        for n in nodes:
            node_indices.append(int(n))
            hyperedge_ids.append(he_id)
    if len(node_indices) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    he_index = torch.tensor([node_indices, hyperedge_ids], dtype=torch.long)
    return he_index


def build_knn_edge_index(
    X: np.ndarray,
    k_neighbors: int = 3,
    include_self: bool = True,
    make_undirected: bool = True,
) -> Tensor:
    """
    Build graph edge_index from KNN neighborhoods similar to torch_cluster.knn_graph.
    Returns edge_index [2, E] with directed edges i->j for j in N_k(i). If make_undirected=True,
    both directions are included (and self-loops if include_self).
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D array: [num_nodes, num_features]")
    num_nodes = X.shape[0]
    k = min(k_neighbors, max(1, num_nodes - 1))

    n_self = 1 if include_self else 0
    nbrs = NearestNeighbors(n_neighbors=k + n_self, algorithm="auto")
    nbrs.fit(X)
    _, indices = nbrs.kneighbors(X)

    src_list: List[int] = []
    dst_list: List[int] = []
    for i in range(num_nodes):
        neigh = indices[i].tolist()
        if not include_self:
            neigh = [j for j in neigh if j != i]
        for j in neigh:
            src_list.append(i)
            dst_list.append(j)
            if make_undirected and j != i:
                src_list.append(j)
                dst_list.append(i)

    if len(src_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    return edge_index


def train_val_test_masks(
    num_nodes: int,
    y: np.ndarray,
    val_size: float = 0.2,
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = 42,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Create boolean masks for train/val/test on a transductive hypergraph.
    """
    indices = np.arange(num_nodes)
    strat = y if stratify else None
    idx_train, idx_tmp, _, y_tmp = train_test_split(
        indices, y, test_size=val_size + test_size, stratify=strat, random_state=random_state
    )
    rel_test_size = test_size / (val_size + test_size) if (val_size + test_size) > 0 else 0.5
    idx_val, idx_test, _, _ = train_test_split(
        idx_tmp, y_tmp, test_size=rel_test_size, stratify=y_tmp if stratify else None, random_state=random_state
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    return train_mask, val_mask, test_mask


def make_hypergraph_data(
    X: np.ndarray,
    y: np.ndarray,
    hyperedge_index: Tensor,
    masks: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
) -> Dict[str, Tensor]:
    """
    Create a plain dict representing a hypergraph suitable for models in this project.
    We intentionally avoid depending on a specific geometric library Data object to keep it lightweight.
    """
    data: Dict[str, Tensor] = {
        "x": torch.tensor(X, dtype=torch.float32),
        "y": torch.tensor(y, dtype=torch.long),
        "hyperedge_index": hyperedge_index.long(),
    }
    if masks is not None:
        train_mask, val_mask, test_mask = masks
        data["train_mask"] = train_mask
        data["val_mask"] = val_mask
        data["test_mask"] = test_mask
    return data


def prepare_data_from_csv(
    csv_path: str,
    label_col: str,
    drop_cols: Optional[List[str]] = None,
    group_col: Optional[str] = None,
    features: Optional[List[str]] = None,
    k_neighbors: int = 10,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dict[str, Tensor], StandardScaler]:
    """
    End-to-end data preparation:
    - load CSV
    - standardize features
    - build KNN hypergraph
    - split masks
    - construct data dict
    """
    X_df, y_s, _ = load_dataset(
        csv_path,
        label_col=label_col,
        drop_cols=drop_cols,
        group_col=group_col,
        features=features,
    )
    X_scaled, scaler = standardize_features(X_df)
    y = y_s.values.astype(np.int64)

    hyperedges = build_knn_hyperedges(X_scaled, k_neighbors=k_neighbors, include_self=True)
    he_index = hyperedges_to_index(hyperedges)

    masks = train_val_test_masks(num_nodes=X_scaled.shape[0], y=y, val_size=val_size, test_size=test_size, stratify=True, random_state=random_state)
    data = make_hypergraph_data(X_scaled, y, he_index, masks)
    return data, scaler


__all__ = [
    "load_dataset",
    "standardize_features",
    "build_knn_hyperedges",
    "hyperedges_to_index",
    "train_val_test_masks",
    "make_hypergraph_data",
    "prepare_data_from_csv",
]


# ============================
# Temporal sequence utilities
# ============================

def _nearest_sample_index(times: List[pd.Timestamp], target_time: pd.Timestamp) -> Optional[int]:
    if len(times) == 0:
        return None
    # times assumed sorted
    left = 0
    right = len(times) - 1
    # binary search
    while left <= right:
        mid = (left + right) // 2
        if times[mid] < target_time:
            left = mid + 1
        elif times[mid] > target_time:
            right = mid - 1
        else:
            return mid
    # choose nearest by absolute difference, prefer previous (right) if exists
    candidates = []
    if 0 <= right < len(times):
        candidates.append((abs((times[right] - target_time).days), right))
    if 0 <= left < len(times):
        candidates.append((abs((times[left] - target_time).days), left))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], -1))
    return candidates[0][1]


def _parse_monitoring_time(series: pd.Series, default_year: int = 2024) -> pd.Series:
    """Parse Monitoring_time; if year is missing like '09-14 12:00', prepend default year."""
    def _parse_one(v: object) -> Optional[pd.Timestamp]:
        if pd.isna(v):
            return None
        s = str(v).strip()
        # Try fast path
        try:
            return pd.to_datetime(s, errors='raise')
        except Exception:
            pass
        # If matches 'MM-DD HH:MM' or 'M-D H:MM'
        try:
            dt = datetime.strptime(f"{default_year}-" + s, "%Y-%m-%d %H:%M")
            return pd.Timestamp(dt)
        except Exception:
            return None
    parsed = series.apply(_parse_one)
    # Fill any None by forward/backward fill after converting to Series of dtype 'datetime64[ns]'
    parsed = pd.to_datetime(parsed, errors='coerce')
    return parsed


def build_temporal_sequences(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    feature_cols: List[str],
    interval_days: int,
    steps: int = 4,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    For each row (station-time), construct a sequence of features at specified backward intervals.
    Filling rule: prefer previous observation at the exact lag; if missing, use nearest previous; if none previous exists, use next future observation.

    Returns:
    - X_seq: [N, T, F]
    - X_curr: [N, F]
    - order_indices: mapping from sequence index to original df row index
    """
    df = df.copy()
    # robust time parsing and daily aggregation (keep last record per day)
    times = _parse_monitoring_time(df[time_col])
    df["__parsed_time__"] = times
    # drop rows without valid time
    df = df[~df["__parsed_time__"].isna()].copy()
    df["__date__"] = df["__parsed_time__"].dt.normalize()
    # keep last record per day per station (by time, then take last)
    df.sort_values([id_col, "__parsed_time__"], inplace=True)
    df_day = df.groupby([id_col, "__date__"], as_index=False).tail(1).copy()
    df_day = df_day.sort_values([id_col, "__date__"]).copy()

    groups = df_day.groupby(id_col)
    rows_seq: List[np.ndarray] = []
    rows_curr: List[np.ndarray] = []
    order_indices: List[int] = []
    T = max(1, int(steps))
    for station_id, g in groups:
        # ensure pandas Timestamps
        g_times: List[pd.Timestamp] = list(pd.to_datetime(g["__date__"].values))
        g_feats = g[feature_cols].values.astype(np.float32)
        g_idx = list(g.index.values)
        for idx_within, (t, feat_now) in enumerate(zip(g_times, g_feats)):
            if not isinstance(t, pd.Timestamp):
                t = pd.Timestamp(t)
            seq_feats: List[np.ndarray] = []
            for k in range(T):
                d = interval_days * k
                target_time = t - timedelta(days=int(d))
                j = _nearest_sample_index(g_times, target_time)
                if j is None:
                    seq_feats.append(feat_now)
                else:
                    seq_feats.append(g_feats[j])
            rows_seq.append(np.stack(seq_feats, axis=0))
            rows_curr.append(feat_now)
            order_indices.append(int(g_idx[idx_within]))

    X_seq = np.stack(rows_seq, axis=0)  # [N, T, F]
    X_curr = np.stack(rows_curr, axis=0)  # [N, F]
    return X_seq, X_curr, order_indices


def prepare_temporal_data_from_csv(
    csv_path: str,
    label_col: str,
    id_col: str,
    time_col: str,
    features: List[str],
    interval_days: int,
    steps: int = -1,
    k_neighbors: int = 3,
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_state: int = 42,
    group_col: Optional[str] = None,
) -> Tuple[Dict[str, Tensor], StandardScaler, np.ndarray]:
    """
    End-to-end temporal preparation producing sequences and spatial KNN graph on current features.
    Returns (data_dict, scaler, basins_array)
    """
    df = pd.read_csv(csv_path)
    # numeric features coercion & fill
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in CSV: {missing}")
    df[features] = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    # labels and groups
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}'")
    y = df[label_col].values.astype(np.int64)
    basins = df[group_col].values if group_col and group_col in df.columns else np.array([None] * len(df))

    # If steps <= 0, infer automatically per-station by date span and use min steps across stations
    auto_steps = steps
    if steps <= 0:
        # compute per-station span
        span_steps = []
        for sid, g in df.groupby(id_col):
            ts = _parse_monitoring_time(g[time_col]).dropna()
            if len(ts) == 0:
                continue
            d0 = ts.min().normalize()
            d1 = ts.max().normalize()
            total_days = max(0, int((d1 - d0).days))
            span_steps.append(max(1, total_days // max(1, int(interval_days))))
        auto_steps = max(1, min(span_steps) if span_steps else 1)

    # build sequences with single interval and multiple steps
    X_seq_raw, X_curr_raw, order_idx = build_temporal_sequences(
        df,
        id_col=id_col,
        time_col=time_col,
        feature_cols=features,
        interval_days=int(interval_days),
        steps=int(auto_steps),
    )
    # reorder labels and groups accordingly
    y = y[np.array(order_idx, dtype=np.int64)]
    basins = basins[np.array(order_idx, dtype=np.int64)]

    # standardize per feature using current features only (match original scaling)
    scaler = StandardScaler()
    X_curr = scaler.fit_transform(X_curr_raw.astype(np.float32))
    # apply same scaler to each time step
    X_seq = X_seq_raw.reshape(X_seq_raw.shape[0] * X_seq_raw.shape[1], X_seq_raw.shape[2])
    X_seq = scaler.transform(X_seq)
    X_seq = X_seq.reshape(-1, X_seq_raw.shape[1], X_seq_raw.shape[2])

    # spatial graph from current features
    edge_index = build_knn_edge_index(X_curr, k_neighbors=k_neighbors, include_self=True, make_undirected=True)

    # masks
    train_mask, val_mask, test_mask = train_val_test_masks(num_nodes=X_curr.shape[0], y=y, val_size=val_size, test_size=test_size, stratify=True, random_state=random_state)

    # total days (global) over daily-aggregated data
    # recompute daily-aggregated dates to avoid dependency on local variables
    times_all = _parse_monitoring_time(df[time_col]).dropna()
    if len(times_all) > 0:
        dmin = times_all.min().normalize()
        dmax = times_all.max().normalize()
        total_days = int((dmax - dmin).days) + 1
    else:
        total_days = 0

    data: Dict[str, Tensor] = {
        "x": torch.tensor(X_curr, dtype=torch.float32),
        "x_seq": torch.tensor(X_seq, dtype=torch.float32),
        "y": torch.tensor(y, dtype=torch.long),
        "hyperedge_index": edge_index.long(),
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }
    # attach total days as int for downstream use (non-tensor for convenience)
    data["total_days"] = torch.tensor(total_days, dtype=torch.long)
    return data, scaler, basins



