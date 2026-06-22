"""
Batch RF + XGB predictions on a CSV (same preprocess as the dashboard).

Usage: python run_inference_on_csv.py [path/to/file.csv]
Default input: test_2018.csv

Posts a Teams summary when TEAMS_WEBHOOK_URL is set.
"""
import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

BASE = Path(__file__).resolve().parent
from cicids2017_preprocess import (
    normalize_upload_columns,
    preprocess_for_rf,
    preprocess_for_xgb,
)
MODEL_BASE = BASE / "SecondModel"

from teams_executive_summary import resolve_teams_webhook_url, send_executive_summary


def load_models():
    rf_model = joblib.load(MODEL_BASE / "rf_model_binary_cicids2017.pkl")
    rf_scaler = joblib.load(MODEL_BASE / "scaler_rf_cicids2017.pkl")
    rf_selector = joblib.load(MODEL_BASE / "feature_selector_rf_cicids2017.pkl")
    rf_label_encoders = joblib.load(MODEL_BASE / "label_encoders_rf_cicids2017.pkl")
    rf_target_encoder = joblib.load(MODEL_BASE / "target_encoder_rf_cicids2017.pkl")

    xgb_model = joblib.load(MODEL_BASE / "xgb_model_binary_cicids2017_95percent.pkl")
    xgb_scaler = joblib.load(MODEL_BASE / "scaler_xgb_cicids2017_95percent.pkl")
    xgb_selector = joblib.load(MODEL_BASE / "feature_selector_xgb_cicids2017_95percent.pkl")
    xgb_label_encoders = joblib.load(MODEL_BASE / "label_encoders_xgb_cicids2017_95percent.pkl")
    xgb_target_encoder = joblib.load(MODEL_BASE / "target_encoder_xgb_cicids2017_95percent.pkl")

    return {
        "rf_model": rf_model,
        "rf_scaler": rf_scaler,
        "rf_selector": rf_selector,
        "rf_label_encoders": rf_label_encoders,
        "rf_target_encoder": rf_target_encoder,
        "xgb_model": xgb_model,
        "xgb_scaler": xgb_scaler,
        "xgb_selector": xgb_selector,
        "xgb_label_encoders": xgb_label_encoders,
        "xgb_target_encoder": xgb_target_encoder,
    }


def main():
    parser = argparse.ArgumentParser(description="Run RF/XGB on CSV (CICIDS2017/2018 columns)")
    parser.add_argument("csv_path", nargs="?", default="test_2018.csv", help="Path to CSV (default: test_2018.csv)")
    args = parser.parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = BASE / csv_path
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        print("Save your CSV as test_2018.csv in the project root, or pass: python run_inference_on_csv.py <path>")
        sys.exit(1)

    print("Loading models...")
    models = load_models()
    rf_te = models["rf_target_encoder"]
    xgb_te = models["xgb_target_encoder"]
    class_names_rf = list(rf_te.classes_)
    class_names_xgb = list(xgb_te.classes_)

    print(f"Reading CSV: {csv_path}")
    t0 = time.perf_counter()
    raw = pd.read_csv(csv_path, sep=None, engine="python")
    if raw.empty:
        print("ERROR: CSV is empty.")
        sys.exit(1)
    n_rows = len(raw)
    raw = normalize_upload_columns(raw)

    # label column name is case-insensitive
    label_col = None
    for c in raw.columns:
        if str(c).strip().lower() == "label":
            label_col = c
            break
    has_label = label_col is not None
    if has_label:
        true_labels = raw[label_col].copy()
        true_labels_norm = true_labels.apply(
            lambda x: "Normal" if str(x).lower() in ["benign", "normal"] else "Attack"
        )
        df_for_preprocess = raw.drop(columns=[label_col])
    else:
        true_labels_norm = None
        df_for_preprocess = raw.copy()

    print("Preprocessing...")
    try:
        X_rf = preprocess_for_rf(
            df_for_preprocess,
            models["rf_scaler"],
            models["rf_selector"],
            models["rf_label_encoders"],
        )
        X_xgb = preprocess_for_xgb(
            df_for_preprocess,
            models["xgb_scaler"],
            models["xgb_selector"],
            models["xgb_label_encoders"],
        )
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        sys.exit(1)

    rf_pred = models["rf_model"].predict(X_rf)
    rf_proba = models["rf_model"].predict_proba(X_rf)
    xgb_pred = models["xgb_model"].predict(X_xgb)
    xgb_proba = models["xgb_model"].predict_proba(X_xgb)
    elapsed = time.perf_counter() - t0

    if resolve_teams_webhook_url():
        ok, msg = send_executive_summary(
            csv_path.name, class_names_rf, class_names_xgb, rf_pred, xgb_pred, elapsed
        )
        print(f"Teams executive summary: {'sent' if ok else 'failed'} ({msg})")

    print()
    print("=" * 60)
    print("PREDICTIONS (first 20 rows)")
    print("=" * 60)
    for i in range(min(20, n_rows)):
        rf_c = class_names_rf[rf_pred[i]]
        rf_conf = float(rf_proba[i].max())
        xgb_c = class_names_xgb[xgb_pred[i]]
        xgb_conf = float(xgb_proba[i].max())
        true_str = f"  True: {true_labels_norm.iloc[i]}" if has_label else ""
        print(f"  Row {i+1:3}:  RF={rf_c:6} ({rf_conf:.2%})  XGB={xgb_c:6} ({xgb_conf:.2%}){true_str}")
    if n_rows > 20:
        print(f"  ... ({n_rows - 20} more rows)")
    print()

    if has_label and true_labels_norm is not None:
        rf_ok = (
            ((true_labels_norm == "Normal") & (rf_pred == 1))
            | ((true_labels_norm == "Attack") & (rf_pred == 0))
        ).sum()
        xgb_ok = (
            ((true_labels_norm == "Normal") & (xgb_pred == 1))
            | ((true_labels_norm == "Attack") & (xgb_pred == 0))
        ).sum()
        # Attack=0, Normal=1 in saved encoders
        y_true = rf_te.transform(true_labels_norm)

        print("=" * 60)
        print("CONFUSION MATRIX & CLASSIFICATION REPORT")
        print("=" * 60)
        print("\n--- RF ---")
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, rf_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, rf_pred, target_names=list(rf_te.classes_)))

        print("\n--- XGB ---")
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, xgb_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, xgb_pred, target_names=list(xgb_te.classes_)))
        print()

        rf_acc = rf_ok / n_rows * 100
        xgb_acc = xgb_ok / n_rows * 100
        print("=" * 60)
        print("ACCURACY (when Label column present)")
        print("=" * 60)
        print(f"  RF:  {rf_ok}/{n_rows} correct = {rf_acc:.2f}%")
        print(f"  XGB: {xgb_ok}/{n_rows} correct = {xgb_acc:.2f}%")
        print()
    print("Done.")


if __name__ == "__main__":
    main()
