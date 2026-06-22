"""
Generate dual-model evidence for CICIDS-2017: bar chart, confusion matrices, metrics table.
Uses saved models and test tensors under SecondModel/ (same as streamlit_dashboard).

Official / thesis results: evaluate on ALL rows in the saved X_test / y_test arrays
(no --sample). Those arrays should be your full 15% test split from training.

Optional --sample is for quick drafts only; use --stratified to preserve class balance.

Outputs: Presentation_Evidence/CICIDS2017_Dual_Model_Evidence/
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = Path(__file__).resolve().parent / "SecondModel"
OUT_DIR = Path(__file__).resolve().parent / "Presentation_Evidence" / "CICIDS2017_Dual_Model_Evidence"


def attack_positive_tp_tn_fp_fn(cm: np.ndarray, classes: np.ndarray) -> tuple[int, int, int, int]:
    """Rows/columns = actual/predicted order per sklearn; positive class = Attack."""
    order = {str(c): i for i, c in enumerate(classes)}
    if "Attack" not in order or "Normal" not in order:
        raise ValueError(f"Expected Attack and Normal in classes_, got {list(classes)}")
    a, n = order["Attack"], order["Normal"]
    tp = int(cm[a, a])
    fn = int(cm[a, n])
    fp = int(cm[n, a])
    tn = int(cm[n, n])
    return tp, tn, fp, fn


def apply_sample(
    X: np.ndarray,
    y: np.ndarray,
    max_rows: int | None,
    stratified: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (X, y) possibly subsampled; third value is a tag for filenames."""
    n = X.shape[0]
    if max_rows is None or n <= max_rows:
        return X, y, "full"
    if stratified:
        n_take = min(max_rows, n)
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_take, random_state=seed)
        idx_train, _ = next(sss.split(np.zeros((n, 1)), y))
        return X[idx_train], y[idx_train], f"sample{n_take}_stratified_s{seed}"
    return X[:max_rows], y[:max_rows], f"sample{max_rows}_head"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CICIDS-2017 RF + XGB evidence from SecondModel/ saved test tensors."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Use at most N rows per model (default: use all rows in each .npy file).",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="With --sample, draw a stratified subset (Attack/Normal) instead of first N rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: Presentation_Evidence/CICIDS2017_Dual_Model_Evidence/).",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rf_paths = {
        "model": BASE / "rf_model_binary_cicids2017.pkl",
        "X": BASE / "X_test_rf_cicids2017.npy",
        "y": BASE / "y_test_rf_cicids2017.npy",
        "te": BASE / "target_encoder_rf_cicids2017.pkl",
    }
    xgb_paths = {
        "model": BASE / "xgb_model_binary_cicids2017_95percent.pkl",
        "X": BASE / "X_test_xgb_cicids2017_95percent.npy",
        "y": BASE / "y_test_xgb_cicids2017_95percent.npy",
        "te": BASE / "target_encoder_xgb_cicids2017_95percent.pkl",
    }

    for name, paths in [("RF", rf_paths), ("XGB", xgb_paths)]:
        for k, p in paths.items():
            if not p.exists():
                print(f"Missing required file: {p}")
                print("Place CICIDS-2017 trained artifacts under SecondModel/ and re-run.")
                sys.exit(1)

    rf_model = joblib.load(rf_paths["model"])
    rf_te = joblib.load(rf_paths["te"])
    X_rf = np.load(rf_paths["X"])
    y_rf = np.asarray(np.load(rf_paths["y"])).ravel()
    if X_rf.shape[0] != len(y_rf):
        print("Error: RF X and y row counts differ.")
        sys.exit(1)

    xgb_model = joblib.load(xgb_paths["model"])
    xgb_te = joblib.load(xgb_paths["te"])
    X_xgb = np.load(xgb_paths["X"])
    y_xgb = np.asarray(np.load(xgb_paths["y"])).ravel()
    if X_xgb.shape[0] != len(y_xgb):
        print("Error: XGB X and y row counts differ.")
        sys.exit(1)

    X_rf, y_rf, tag_rf = apply_sample(X_rf, y_rf, args.sample, args.stratified, args.seed)
    X_xgb, y_xgb, tag_xgb = apply_sample(X_xgb, y_xgb, args.sample, args.stratified, args.seed)
    eval_tag = tag_rf if tag_rf == tag_xgb else f"{tag_rf}__{tag_xgb}"
    if args.sample is not None:
        print(
            f"Note: --sample {args.sample} active ({'stratified' if args.stratified else 'head'}). "
            "For thesis, prefer full test set (omit --sample)."
        )

    file_suffix = "" if eval_tag == "full" else f"_{eval_tag}"

    rf_pred = rf_model.predict(X_rf)
    rf_proba = rf_model.predict_proba(X_rf)
    xgb_pred = xgb_model.predict(X_xgb)
    xgb_proba = xgb_model.predict_proba(X_xgb)

    rf_classes = rf_te.classes_
    xgb_classes = xgb_te.classes_
    rf_labels = list(range(len(rf_classes)))
    xgb_labels = list(range(len(xgb_classes)))

    rf_cm = confusion_matrix(y_rf, rf_pred, labels=rf_labels)
    xgb_cm = confusion_matrix(y_xgb, xgb_pred, labels=xgb_labels)

    def metrics_block(y_true, y_pred, proba, classes) -> dict:
        acc = accuracy_score(y_true, y_pred)
        p_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        r_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        atk = list(classes).index("Attack")
        p_a = precision_score(y_true, y_pred, pos_label=atk, zero_division=0)
        r_a = recall_score(y_true, y_pred, pos_label=atk, zero_division=0)
        f_a = f1_score(y_true, y_pred, pos_label=atk, zero_division=0)
        # ROC/PR need y in {0,1} with score = P(Attack); encoder may use Attack=0.
        y_bin = (np.asarray(y_true) == atk).astype(np.int32)
        pa = np.asarray(proba)[:, atk].astype(np.float64)
        roc = roc_auc_score(y_bin, pa)
        pr = average_precision_score(y_bin, pa)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
        tp, tn, fp, fn = attack_positive_tp_tn_fp_fn(cm, classes)
        return {
            "accuracy": acc,
            "precision_attack": p_a,
            "recall_attack": r_a,
            "f1_attack": f_a,
            "precision_w": p_w,
            "recall_w": r_w,
            "f1_w": f_w,
            "roc": roc,
            "pr": pr,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "n": len(y_true),
        }

    m_rf = metrics_block(y_rf, rf_pred, rf_proba, rf_classes)
    m_xgb = metrics_block(y_xgb, xgb_pred, xgb_proba, xgb_classes)

    col_rf = "Random Forest (CICIDS-2017)"
    col_xgb = "XGBoost (CICIDS-2017)"
    table_data = {
        "Metric": [
            "Test subset size (N)",
            "Accuracy",
            "Precision (Attack)",
            "Recall (Attack)",
            "F1-Score (Attack)",
            "Precision (Weighted)",
            "Recall (Weighted)",
            "F1-Score (Weighted)",
            "ROC-AUC",
            "PR-AUC",
            "True Positives (TP)",
            "True Negatives (TN)",
            "False Positives (FP)",
            "False Negatives (FN)",
        ],
        col_rf: [
            f"{m_rf['n']:,}",
            f"{m_rf['accuracy']*100:.2f}%",
            f"{m_rf['precision_attack']*100:.2f}%",
            f"{m_rf['recall_attack']*100:.2f}%",
            f"{m_rf['f1_attack']*100:.2f}%",
            f"{m_rf['precision_w']*100:.2f}%",
            f"{m_rf['recall_w']*100:.2f}%",
            f"{m_rf['f1_w']*100:.2f}%",
            f"{m_rf['roc']:.4f}",
            f"{m_rf['pr']:.4f}",
            f"{m_rf['tp']:,}",
            f"{m_rf['tn']:,}",
            f"{m_rf['fp']:,}",
            f"{m_rf['fn']:,}",
        ],
        col_xgb: [
            f"{m_xgb['n']:,}",
            f"{m_xgb['accuracy']*100:.2f}%",
            f"{m_xgb['precision_attack']*100:.2f}%",
            f"{m_xgb['recall_attack']*100:.2f}%",
            f"{m_xgb['f1_attack']*100:.2f}%",
            f"{m_xgb['precision_w']*100:.2f}%",
            f"{m_xgb['recall_w']*100:.2f}%",
            f"{m_xgb['f1_w']*100:.2f}%",
            f"{m_xgb['roc']:.4f}",
            f"{m_xgb['pr']:.4f}",
            f"{m_xgb['tp']:,}",
            f"{m_xgb['tn']:,}",
            f"{m_xgb['fp']:,}",
            f"{m_xgb['fn']:,}",
        ],
    }
    df_table = pd.DataFrame(table_data)
    csv_path = out_dir / f"CICIDS2017_DUAL_MODEL_METRICS{file_suffix}.csv"
    df_table.to_csv(csv_path, index=False)
    # Same data, familiar name (matches old NSL+XGB workflow style) — new approach = both on CICIDS-2017
    if file_suffix == "":
        df_table.to_csv(out_dir / "METRICS_COMPARISON_TABLE_CICIDS2017_DUAL_MODEL.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")

    # Standalone metrics table figure
    fig_t, ax_t = plt.subplots(figsize=(16, 10))
    ax_t.axis("tight")
    ax_t.axis("off")
    tbl = ax_t.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.35, 0.325, 0.325],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.2)
    for i in range(3):
        tbl[(0, i)].set_facecolor("#2E86AB")
        tbl[(0, i)].set_text_props(weight="bold", color="white", size=12)
    for i in range(1, len(df_table) + 1):
        for j in range(3):
            tbl[(i, j)].set_facecolor("#F0F0F0" if i % 2 == 0 else "white")
    for row_idx in [1, 2, 3, 4]:
        for j in range(3):
            tbl[(row_idx, j)].set_facecolor("#E8F4F8")
            tbl[(row_idx, j)].set_text_props(weight="bold", size=11)
    fig_t.suptitle("Detailed Performance Metrics Comparison", fontsize=18, fontweight="bold", y=0.98)
    tbl_sub = (
        "Random Forest & XGBoost on CICIDS-2017 • Fixed 70/15/15 split • 15% test set • Positive class = Attack"
    )
    fig_t.text(0.5, 0.94, tbl_sub, ha="center", fontsize=12, style="italic", alpha=0.85)
    plt.tight_layout()
    fig_t.savefig(out_dir / f"CICIDS2017_DUAL_MODEL_METRICS_TABLE{file_suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig_t)

    # -------------------------------------------------------------------------
    # Separate figures for PowerPoint (one visual per slide)
    # -------------------------------------------------------------------------
    ppt_footer = f"CICIDS-2017 • Fixed 70/15/15 • Test set N = {m_rf['n']:,} • Positive class = Attack"

    # Slide 1: Accuracy comparison bar chart
    fig_b, ax_b = plt.subplots(figsize=(10, 6.5))
    models_pp = ["Random Forest\n(CICIDS-2017)", "XGBoost\n(CICIDS-2017)"]
    accs_pp = [m_rf["accuracy"] * 100, m_xgb["accuracy"] * 100]
    colors_pp = ["#2E86AB", "#F24236"]
    bars_b = ax_b.bar(models_pp, accs_pp, color=colors_pp, edgecolor="black", linewidth=2.2, width=0.52)
    ax_b.set_ylabel("Test accuracy (%)", fontsize=14, fontweight="bold")
    ax_b.set_title(
        "Dual-model intrusion detection — test set accuracy",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax_b.set_ylim(0, 105)
    ax_b.axhline(y=90, color="red", linestyle="--", alpha=0.65, linewidth=2, label="90% target")
    ax_b.legend(loc="upper right", fontsize=11)
    ax_b.grid(axis="y", alpha=0.35, linestyle="--")
    for bar, acc in zip(bars_b, accs_pp):
        h = bar.get_height()
        ax_b.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.2,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )
    fig_b.text(0.5, 0.02, ppt_footer, ha="center", fontsize=10, style="italic", color="#444444")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig_b.savefig(
        out_dir / f"CICIDS2017_SLIDE_1_ACCURACY_BAR{file_suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_b)

    # Slide 2: Random Forest confusion matrix only
    fig_rf, ax_rf = plt.subplots(figsize=(8, 6.8))
    sns.heatmap(
        rf_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax_rf,
        xticklabels=list(rf_classes),
        yticklabels=list(rf_classes),
        cbar_kws={"label": "Count"},
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax_rf.set_title(
        f"Random Forest (CICIDS-2017)\nConfusion matrix — 15% test set\nAccuracy: {m_rf['accuracy']*100:.2f}%",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax_rf.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax_rf.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    fig_rf.text(0.5, 0.02, ppt_footer, ha="center", fontsize=10, style="italic", color="#444444")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig_rf.savefig(
        out_dir / f"CICIDS2017_SLIDE_2_CONFUSION_MATRIX_RF{file_suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_rf)

    # Slide 3: XGBoost confusion matrix only
    fig_xg, ax_xg = plt.subplots(figsize=(8, 6.8))
    sns.heatmap(
        xgb_cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        ax=ax_xg,
        xticklabels=list(xgb_classes),
        yticklabels=list(xgb_classes),
        cbar_kws={"label": "Count"},
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax_xg.set_title(
        f"XGBoost (CICIDS-2017)\nConfusion matrix — 15% test set\nAccuracy: {m_xgb['accuracy']*100:.2f}%",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax_xg.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax_xg.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    fig_xg.text(0.5, 0.02, ppt_footer, ha="center", fontsize=10, style="italic", color="#444444")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig_xg.savefig(
        out_dir / f"CICIDS2017_SLIDE_3_CONFUSION_MATRIX_XGB{file_suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_xg)

    # One-sheet "Results Evidence" layout (same structure as old RF-NSL + XGB-CICIDS slide;
    # new approach: both columns are CICIDS-2017).
    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.28, height_ratios=[0.85, 1.05, 2.35])
    fig.suptitle(
        "Dual-Model Intrusion Detection System\nModel Performance Comparison",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.955,
        "Random Forest (CICIDS-2017) & XGBoost (CICIDS-2017) • Fixed 70/15/15 split • 15% held-out test set",
        ha="center",
        fontsize=11,
        style="italic",
        color="#333333",
    )

    ax_bar = fig.add_subplot(gs[0, :])
    models = ["Random Forest\n(CICIDS-2017)", "XGBoost\n(CICIDS-2017)"]
    accs = [m_rf["accuracy"] * 100, m_xgb["accuracy"] * 100]
    colors = ["#2E86AB", "#F24236"]
    bars = ax_bar.bar(models, accs, color=colors, edgecolor="black", linewidth=2.2, width=0.55)
    ax_bar.set_ylabel("Accuracy (%)", fontsize=13, fontweight="bold")
    ax_bar.set_title("Test set accuracy", fontsize=14, fontweight="bold", pad=10)
    ax_bar.set_ylim(0, 105)
    ax_bar.axhline(y=90, color="red", linestyle="--", alpha=0.6, linewidth=2, label="90% Target")
    ax_bar.legend(fontsize=10)
    ax_bar.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, acc in zip(bars, accs):
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2.0, h + 1, f"{acc:.2f}%", ha="center", va="bottom", fontsize=15, fontweight="bold")

    ax1 = fig.add_subplot(gs[1, 0])
    sns.heatmap(
        rf_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax1,
        xticklabels=list(rf_classes),
        yticklabels=list(rf_classes),
        cbar_kws={"label": "Count"},
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax1.set_title(
        f"{col_rf}\nFixed 70/15/15 Split — Test Set\nAccuracy: {m_rf['accuracy']*100:.2f}%",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_ylabel("Actual", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Predicted", fontsize=11, fontweight="bold")

    ax2 = fig.add_subplot(gs[1, 1])
    sns.heatmap(
        xgb_cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        ax=ax2,
        xticklabels=list(xgb_classes),
        yticklabels=list(xgb_classes),
        cbar_kws={"label": "Count"},
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax2.set_title(
        f"{col_xgb}\nFixed 70/15/15 Split — Test Set\nAccuracy: {m_xgb['accuracy']*100:.2f}%",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_ylabel("Actual", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Predicted", fontsize=11, fontweight="bold")

    ax_tab = fig.add_subplot(gs[2, :])
    ax_tab.axis("tight")
    ax_tab.axis("off")
    t2 = ax_tab.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.36, 0.32, 0.32],
    )
    t2.auto_set_font_size(False)
    t2.set_fontsize(9)
    t2.scale(1, 1.85)
    for i in range(3):
        t2[(0, i)].set_facecolor("#2E86AB")
        t2[(0, i)].set_text_props(weight="bold", color="white", size=10)
    for i in range(1, len(df_table) + 1):
        for j in range(3):
            t2[(i, j)].set_facecolor("#F0F0F0" if i % 2 == 0 else "white")
    for row_idx in [1, 2, 3, 4]:
        for j in range(3):
            t2[(row_idx, j)].set_facecolor("#E8F4F8")
            t2[(row_idx, j)].set_text_props(weight="bold", size=9)
    ax_tab.set_title(
        "Detailed Performance Metrics Comparison\nPositive class = Attack",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    comb_png = out_dir / f"CICIDS2017_DUAL_MODEL_EVIDENCE_COMBINED{file_suffix}.png"
    plt.savefig(comb_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    # Friendly alias — equivalent to your old "Results Evidence" one-pager
    if file_suffix == "":
        shutil.copy2(comb_png, out_dir / "RESULTS_EVIDENCE_CICIDS2017_DUAL_MODEL.png")

    summary = f"""CICIDS-2017 dual-model evidence (saved SecondModel/ artifacts)
Evaluation: {eval_tag}
Generated from: X_test_rf_cicids2017.npy / y_test_rf_cicids2017.npy
                X_test_xgb_cicids2017_95percent.npy / y_test_xgb_cicids2017_95percent.npy
Positive class for precision/recall/F1 (Attack): sklearn pos_label=Attack

Random Forest:  accuracy {m_rf['accuracy']*100:.2f}%  N={m_rf['n']:,}
XGBoost:        accuracy {m_xgb['accuracy']*100:.2f}%  N={m_xgb['n']:,}

Output: {out_dir}
  {csv_path.name}
  METRICS_COMPARISON_TABLE_CICIDS2017_DUAL_MODEL.csv (if full eval)
  RESULTS_EVIDENCE_CICIDS2017_DUAL_MODEL.png (if full eval — combined one-pager)
  CICIDS2017_SLIDE_1_ACCURACY_BAR{file_suffix}.png — PPT slide 1
  CICIDS2017_SLIDE_2_CONFUSION_MATRIX_RF{file_suffix}.png — PPT slide 2
  CICIDS2017_SLIDE_3_CONFUSION_MATRIX_XGB{file_suffix}.png — PPT slide 3
  CICIDS2017_DUAL_MODEL_METRICS_TABLE{file_suffix}.png — optional 4th slide (table only)
  CICIDS2017_DUAL_MODEL_EVIDENCE_COMBINED{file_suffix}.png
"""
    (out_dir / f"SUMMARY{file_suffix}.txt").write_text(summary, encoding="utf-8")

    print(summary)
    print("[OK] Done.")


if __name__ == "__main__":
    main()
