"""
CSV preprocessing for RF and XGB .
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fwd_len_col(df):
    if "Total Length of Fwd Packets" in df.columns:
        return "Total Length of Fwd Packets"
    if "Fwd Packets Length Total" in df.columns:
        return "Fwd Packets Length Total"
    return None


def _bwd_len_col(df):
    if "Total Length of Bwd Packets" in df.columns:
        return "Total Length of Bwd Packets"
    if "Bwd Packets Length Total" in df.columns:
        return "Bwd Packets Length Total"
    return None


def _bwd_pkts_col(df):
    if "Total Bwd Packets" in df.columns:
        return "Total Bwd Packets"
    if "Total Backward Packets" in df.columns:
        return "Total Backward Packets"
    return None


def add_engineered_features_rf(df):
    """RF training uses full engineered features."""
    df = df.copy()
    fwd_len = _fwd_len_col(df)
    bwd_len = _bwd_len_col(df)
    bwd_pkts = _bwd_pkts_col(df)
    if fwd_len and bwd_len:
        df["total_bytes"] = df[fwd_len] + df[bwd_len]
        df["bytes_ratio"] = df[fwd_len] / (df[bwd_len] + 1)
    if "Total Fwd Packets" in df.columns and bwd_pkts:
        df["total_packets"] = df["Total Fwd Packets"] + df[bwd_pkts]
        df["packets_ratio"] = df["Total Fwd Packets"] / (df[bwd_pkts] + 1)
    if "Flow Duration" in df.columns:
        if "total_bytes" in df.columns:
            df["data_transfer_rate"] = df["total_bytes"] / (df["Flow Duration"] + 1)
        if "total_packets" in df.columns:
            df["packets_per_second"] = df["total_packets"] / (df["Flow Duration"] + 1)
    if "Total Fwd Packets" in df.columns and "Flow Duration" in df.columns:
        df["fwd_packets_rate"] = df["Total Fwd Packets"] / (df["Flow Duration"] + 1)
    if bwd_pkts and "Flow Duration" in df.columns:
        df["bwd_packets_rate"] = df[bwd_pkts] / (df["Flow Duration"] + 1)
    return df


def add_engineered_features_xgb(df):
    """XGB training uses minimal engineered features."""
    df = df.copy()
    fwd_len = _fwd_len_col(df)
    bwd_len = _bwd_len_col(df)
    bwd_pkts = _bwd_pkts_col(df)
    if fwd_len and bwd_len:
        df["total_bytes"] = df[fwd_len] + df[bwd_len]
    if "Total Fwd Packets" in df.columns and bwd_pkts:
        df["total_packets"] = df["Total Fwd Packets"] + df[bwd_pkts]
    return df


COLUMNS_TO_DROP = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "SimillarHTTP",
    "Inbound",
    "Fwd Header Length.1",
]

COLUMN_ALIAS_MAP = {
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Fwd Packets Length Total",
    "TotLen Bwd Pkts": "Bwd Packets Length Total",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Fwd IAT Tot": "Fwd IAT Total",
    "Bwd IAT Tot": "Bwd IAT Total",
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Min": "Packet Length Min",
    "Pkt Len Max": "Packet Length Max",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "Pkt Size Avg": "Avg Packet Size",
    "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "URG Flag Cnt": "URG Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",
    "Init Fwd Win Byts": "Init Fwd Win Bytes",
    "Init Bwd Win Byts": "Init Bwd Win Bytes",
    "Dst Port": "Destination Port",
}


def normalize_upload_columns(df):
    """Rename columns to CICIDS-2017 training names. Only renames columns present in df."""
    rename = {k: v for k, v in COLUMN_ALIAS_MAP.items() if k in df.columns and k != v}
    if rename:
        df = df.rename(columns=rename)
    return df


def _align_and_fill(X_df, expected_columns, fill_value=0):
    out = pd.DataFrame(index=X_df.index)
    for c in expected_columns:
        if c in X_df.columns:
            out[c] = X_df[c].values
        else:
            out[c] = fill_value
    return out


def _encode_categoricals(X_df, label_encoders):
    X = X_df.copy()
    for feat, le in label_encoders.items():
        if feat not in X.columns:
            continue
        vals = X[feat].astype(str)
        cmap = {c: le.transform([c])[0] for c in le.classes_}
        X[feat] = vals.map(lambda v: cmap.get(v, cmap[le.classes_[0]])).astype(int)
    return X


def preprocess_for_rf(df, scaler_rf, selector_rf, label_encoders_rf):
    """Preprocess DataFrame through RF pipeline. """
    df = add_engineered_features_rf(df)
    for col in COLUMNS_TO_DROP:
        if col in df.columns:
            df = df.drop(col, axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = _encode_categoricals(df, label_encoders_rf)
    if hasattr(scaler_rf, "feature_names_in_"):
        expected = list(scaler_rf.feature_names_in_)
    else:
        expected = list(df.columns)
    X_aligned = _align_and_fill(df, expected)
    return selector_rf.transform(scaler_rf.transform(X_aligned))


def preprocess_for_xgb(df, scaler_xgb, selector_xgb, label_encoders_xgb):
    """Preprocess DataFrame through XGB pipeline."""
    df = add_engineered_features_xgb(df)
    for col in COLUMNS_TO_DROP:
        if col in df.columns:
            df = df.drop(col, axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = _encode_categoricals(df, label_encoders_xgb)
    if hasattr(scaler_xgb, "feature_names_in_"):
        expected = list(scaler_xgb.feature_names_in_)
    else:
        expected = list(df.columns)
    X_aligned = _align_and_fill(df, expected)
    return selector_xgb.transform(scaler_xgb.transform(X_aligned))
