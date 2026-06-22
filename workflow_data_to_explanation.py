"""
CSV → preprocess → predict → SHAP report (CLI, no Streamlit).

Usage:
  python workflow_data_to_explanation.py [csv] [--rows 3] [--out report.txt]
"""

import sys
import argparse
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from cicids2017_preprocess import (
    normalize_upload_columns,
    preprocess_for_rf,
    preprocess_for_xgb,
)
MODEL_BASE = BASE / "SecondModel"

from teams_executive_summary import resolve_teams_webhook_url, send_executive_summary


def _safe_float(x):
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return float(s)


def load_models_and_feature_names():
    """Load RF/XGB models, preprocessors, and feature names (if available)."""
    out = {}
    # RF
    out["rf_model"] = joblib.load(MODEL_BASE / "rf_model_binary_cicids2017.pkl")
    out["rf_scaler"] = joblib.load(MODEL_BASE / "scaler_rf_cicids2017.pkl")
    out["rf_selector"] = joblib.load(MODEL_BASE / "feature_selector_rf_cicids2017.pkl")
    out["rf_label_encoders"] = joblib.load(MODEL_BASE / "label_encoders_rf_cicids2017.pkl")
    out["rf_target_encoder"] = joblib.load(MODEL_BASE / "target_encoder_rf_cicids2017.pkl")
    try:
        out["rf_feature_names"] = joblib.load(MODEL_BASE / "feature_names_rf_cicids2017.pkl")
    except Exception:
        out["rf_feature_names"] = None  # set from X_rf.shape[1] later

    # XGB
    out["xgb_model"] = joblib.load(MODEL_BASE / "xgb_model_binary_cicids2017_95percent.pkl")
    out["xgb_scaler"] = joblib.load(MODEL_BASE / "scaler_xgb_cicids2017_95percent.pkl")
    out["xgb_selector"] = joblib.load(MODEL_BASE / "feature_selector_xgb_cicids2017_95percent.pkl")
    out["xgb_label_encoders"] = joblib.load(MODEL_BASE / "label_encoders_xgb_cicids2017_95percent.pkl")
    out["xgb_target_encoder"] = joblib.load(MODEL_BASE / "target_encoder_xgb_cicids2017_95percent.pkl")
    try:
        out["xgb_feature_names"] = joblib.load(MODEL_BASE / "feature_names_xgb_cicids2017_95percent.pkl")
    except Exception:
        out["xgb_feature_names"] = None  # set from X_xgb.shape[1] later
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Workflow: CSV → preprocess → predict → SHAP explanations"
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="test_2018.csv",
        help="Path to CSV (default: test_2018.csv)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=3,
        help="Number of rows to explain with SHAP (default: 3)",
    )
    parser.add_argument(
        "--out",
        default="workflow_explanations.txt",
        help="Output path for explanation report (default: workflow_explanations.txt)",
    )
    args = parser.parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = BASE / csv_path
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    print("Loading models and preprocessors...")
    models = load_models_and_feature_names()
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

    # Predictions summary CSV
    pred_df = pd.DataFrame({
        "row": range(1, n_rows + 1),
        "RF_pred": [class_names_rf[p] for p in rf_pred],
        "RF_conf": [float(rf_proba[i].max()) for i in range(n_rows)],
        "XGB_pred": [class_names_xgb[p] for p in xgb_pred],
        "XGB_conf": [float(xgb_proba[i].max()) for i in range(n_rows)],
    })
    if has_label:
        pred_df["True"] = true_labels_norm.values
        pred_df["RF_match"] = pred_df["RF_pred"] == pred_df["True"]
        pred_df["XGB_match"] = pred_df["XGB_pred"] == pred_df["True"]
    pred_path = BASE / "workflow_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions written to {pred_path}")

    n_explain = min(args.rows, n_rows)
    if n_explain <= 0:
        print("No rows to explain.")
        elapsed = time.perf_counter() - t0
        if resolve_teams_webhook_url():
            ok, msg = send_executive_summary(
                csv_path.name, class_names_rf, class_names_xgb, rf_pred, xgb_pred, elapsed
            )
            print(f"Teams executive summary: {'sent' if ok else 'failed'} ({msg})")
        return

    print("Building SHAP explainers...")
    import shap
    shap_rf = shap.TreeExplainer(models["rf_model"])
    shap_xgb = shap.TreeExplainer(models["xgb_model"])
    n_rf_f = X_rf.shape[1]
    n_xgb_f = X_xgb.shape[1]
    rf_fnames = models["rf_feature_names"]
    xgb_fnames = models["xgb_feature_names"]
    if rf_fnames is None or len(rf_fnames) != n_rf_f:
        rf_fnames = [f"rf_f{i}" for i in range(n_rf_f)]
    if xgb_fnames is None or len(xgb_fnames) != n_xgb_f:
        xgb_fnames = [f"xgb_f{i}" for i in range(n_xgb_f)]

    lines = []
    lines.append("=" * 70)
    lines.append("WORKFLOW: DATA INPUT → EXPLANATION OUTPUT")
    lines.append("=" * 70)
    lines.append(f"CSV: {csv_path.name}  |  Rows: {n_rows}  |  Explained: {n_explain}")
    lines.append("")

    for idx in range(n_explain):
        rf_input = X_rf[idx : idx + 1]
        xgb_input = X_xgb[idx : idx + 1]
        rf_p = int(rf_pred[idx])
        xgb_p = int(xgb_pred[idx])
        rf_conf = float(rf_proba[idx].max())
        xgb_conf = float(xgb_proba[idx].max())
        pred_rf = class_names_rf[rf_p]
        pred_xgb = class_names_xgb[xgb_p]

        lines.append("-" * 70)
        lines.append(f"ROW {idx + 1}")
        lines.append("-" * 70)
        lines.append(f"  RF:  {pred_rf}  (confidence {rf_conf:.2%})")
        lines.append(f"  XGB: {pred_xgb}  (confidence {xgb_conf:.2%})")
        if has_label:
            lines.append(f"  True label: {true_labels_norm.iloc[idx]}")
        lines.append("")

        # SHAP RF
        try:
            sv_rf = shap_rf.shap_values(rf_input)
            if isinstance(sv_rf, list):
                safe_i = min(rf_p, len(sv_rf) - 1)
                shap_vals = (
                    np.array(sv_rf[safe_i][0]).flatten()
                    if len(sv_rf[safe_i]) else np.array(sv_rf[0][0]).flatten()
                )
            else:
                shap_vals = (
                    sv_rf[0, :, min(rf_p, sv_rf.shape[2] - 1)]
                    if len(sv_rf.shape) == 3 else sv_rf[0]
                )
            shap_vals = np.asarray(shap_vals).flatten()
            if len(shap_vals) != n_rf_f:
                shap_vals = np.pad(shap_vals, (0, max(0, n_rf_f - len(shap_vals))))[:n_rf_f]
            shap_vals = np.array([_safe_float(v) for v in shap_vals])
            top_rf = sorted(
                [(rf_fnames[i], _safe_float(shap_vals[i])) for i in range(n_rf_f)],
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:10]
            lines.append("  SHAP (RF) – top features driving this prediction:")
            for f, v in top_rf:
                direction = "↑ Normal" if v > 0 else "↑ Attack"
                lines.append(f"    {f}: {v:+.4f}  ({direction})")
        except Exception as e:
            lines.append(f"  SHAP (RF) error: {e}")
        lines.append("")

        # SHAP XGB
        try:
            sv_xgb = shap_xgb.shap_values(xgb_input)
            if isinstance(sv_xgb, list):
                safe_i = min(xgb_p, len(sv_xgb) - 1)
                class_vals = sv_xgb[safe_i]
                shap_vals = (
                    np.array(class_vals[0]).flatten()
                    if hasattr(class_vals, "__len__") and len(class_vals) else np.array(sv_xgb[0][0]).flatten()
                )
            else:
                sv_xgb = np.array(sv_xgb)
                shap_vals = (
                    sv_xgb[0, :, min(xgb_p, sv_xgb.shape[2] - 1)]
                    if len(sv_xgb.shape) == 3 else sv_xgb[0]
                )
            shap_vals = np.asarray(shap_vals).flatten()
            if len(shap_vals) != n_xgb_f:
                shap_vals = np.pad(shap_vals, (0, max(0, n_xgb_f - len(shap_vals))))[:n_xgb_f]
            shap_vals = np.array([_safe_float(v) for v in shap_vals])
            top_xgb = sorted(
                [(xgb_fnames[i], _safe_float(shap_vals[i])) for i in range(n_xgb_f)],
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:10]
            lines.append("  SHAP (XGB) – top features driving this prediction:")
            for f, v in top_xgb:
                direction = "↑ Normal" if v > 0 else "↑ Attack"
                lines.append(f"    {f}: {v:+.4f}  ({direction})")
        except Exception as e:
            lines.append(f"  SHAP (XGB) error: {e}")

        # Short natural-language summary
        lines.append("")
        agree = pred_rf == pred_xgb
        lines.append(
            f"  Summary: Both models predict {pred_rf}."
            if agree
            else f"  Summary: RF predicts {pred_rf}, XGB predicts {pred_xgb} (disagreement)."
        )
        lines.append("")

    lines.append("=" * 70)
    lines.append("END OF EXPLANATION REPORT")
    lines.append("=" * 70)

    report = "\n".join(lines)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = BASE / out_path
    out_path.write_text(report, encoding="utf-8")
    print(f"Explanation report written to {out_path}")
    print()
    print(report[: min(1500, len(report))] + ("..." if len(report) > 1500 else ""))

    elapsed = time.perf_counter() - t0
    if resolve_teams_webhook_url():
        ok, msg = send_executive_summary(
            csv_path.name, class_names_rf, class_names_xgb, rf_pred, xgb_pred, elapsed
        )
        print(f"Teams executive summary: {'sent' if ok else 'failed'} ({msg})")

    print("Done.")


if __name__ == "__main__":
    main()
