"""
Assemble Final_PPT_Evidence/ for viva / PowerPoint.

Copies:
  - RF & XGB global feature-importance bar charts (from training)
  - XAI validation text reports (from evaluate_xai.py)

Usage:
  python build_final_ppt_evidence.py           # run XAI eval if outputs missing (N=500)
  python build_final_ppt_evidence.py --n100    # faster XAI eval (N=100)
  python build_final_ppt_evidence.py --skip-eval   # copy PNGs only (XAI txt must exist)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Final_PPT_Evidence"
SECOND = ROOT / "SecondModel"
PRES = ROOT / "Presentation_Evidence"

IMAGE_COPIES = [
    (
        SECOND / "rf_feature_importance_cicids2017.png",
        OUT / "01_RF_global_feature_importance.png",
    ),
    (
        SECOND / "xgb_feature_importance_cicids2017_95percent.png",
        OUT / "02_XGB_global_feature_importance.png",
    ),
]

XAI_TEXT_FILES = [
    "XAI_VALIDATION_PROGRESS_PPT.txt",
    "xai_thesis_table_a_metrics.txt",
    "xai_thesis_table_b_agreement.txt",
    "xai_thesis_evaluation_results.txt",
]


def _xai_outputs_present(use_n100: bool) -> bool:
    suffix = "_n100" if use_n100 else ""
    names = [
        f"XAI_VALIDATION_PROGRESS_PPT{suffix}.txt",
        f"xai_thesis_table_a_metrics{suffix}.txt",
        f"xai_thesis_table_b_agreement{suffix}.txt",
        f"xai_thesis_evaluation_results{suffix}.txt",
    ]
    return all((PRES / n).exists() for n in names)


def _run_xai_eval(use_n100: bool) -> None:
    cmd = [sys.executable, str(ROOT / "evaluate_xai.py")]
    if use_n100:
        cmd.append("--n100")
    print("Running XAI evaluation (this may take several minutes)...")
    subprocess.run(cmd, cwd=ROOT, check=True)


def _copy_xai_text(use_n100: bool) -> None:
    suffix = "_n100" if use_n100 else ""
    mapping = [
        (f"XAI_VALIDATION_PROGRESS_PPT{suffix}.txt", "03_XAI_validation_PPT_snippet.txt"),
        (f"xai_thesis_table_a_metrics{suffix}.txt", "04_XAI_table_a_metrics.txt"),
        (f"xai_thesis_table_b_agreement{suffix}.txt", "05_XAI_table_b_agreement.txt"),
        (f"xai_thesis_evaluation_results{suffix}.txt", "06_XAI_full_evaluation_results.txt"),
    ]
    for src_name, dst_name in mapping:
        src = PRES / src_name
        if not src.exists():
            raise FileNotFoundError(f"Missing {src}. Run evaluate_xai.py first.")
        shutil.copy2(src, OUT / dst_name)
        print(f"  copied {dst_name}")


TABLE_A_COLUMNS = [
    "LIME_Spearman_mean",
    "LIME_Spearman_pct_below_0_5",
    "SHAP_flips_k3",
    "SHAP_acc_drop_pct_k3",
    "LIME_flips_k3",
    "LIME_acc_drop_pct_k3",
    "ELI5_flips_k3",
    "ELI5_acc_drop_pct_k3",
    "SHAP_flips_k5",
    "SHAP_acc_drop_pct_k5",
    "LIME_flips_k5",
    "LIME_acc_drop_pct_k5",
    "ELI5_flips_k5",
    "ELI5_acc_drop_pct_k5",
    "SHAP_mean_s",
    "LIME_mean_s",
    "ELI5_one_off_s",
]


def _parse_table_a(path: Path) -> dict:
    """Parse Table A into RF/XGB metric dicts."""
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = {}
    for row_line in lines:
        if row_line.strip().startswith(("Model", "Table", "=")):
            continue
        parts = row_line.split()
        if parts[0] not in ("RF", "XGB"):
            continue
        model = parts[0]
        values = parts[1:]
        if len(values) != len(TABLE_A_COLUMNS):
            raise ValueError(f"Table A row for {model} has {len(values)} values, expected {len(TABLE_A_COLUMNS)}")
        out[model] = dict(zip(TABLE_A_COLUMNS, values))
    if "RF" not in out or "XGB" not in out:
        raise ValueError(f"Could not parse RF/XGB rows from {path}")
    return out


def _parse_table_b(path: Path) -> dict:
    """Parse agreement means from Table B."""
    text = path.read_text(encoding="utf-8")
    out: dict = {"RF": {}, "XGB": {}}
    current = None
    for line in text.splitlines():
        line = line.strip()
        if line == "RF:":
            current = "RF"
        elif line == "XGB:":
            current = "XGB"
        elif current and "Spearman: mean=" in line:
            key = line.split(" Spearman:")[0].strip()
            mean = line.split("mean=")[1].split(",")[0]
            out[current][f"{key}_spearman"] = mean
        elif current and "Jaccard: mean=" in line:
            key = line.split(" Jaccard:")[0].strip()
            mean = line.split("mean=")[1].split(",")[0]
            out[current][f"{key}_jaccard"] = mean
    return out


def _write_slide_one_page(use_n100: bool) -> None:
    """Single file mapping slide bullets to exact evidence numbers."""
    n_label = "100" if use_n100 else "500"
    table_a = OUT / "04_XAI_table_a_metrics.txt"
    table_b = OUT / "05_XAI_table_b_agreement.txt"
    if not table_a.exists() or not table_b.exists():
        raise FileNotFoundError("Run XAI copy step before writing slide summary.")

    ta = _parse_table_a(table_a)
    tb = _parse_table_b(table_b)
    rf, xg = ta["RF"], ta["XGB"]

    rf_stab = float(rf["LIME_Spearman_mean"])
    xg_stab = float(xg["LIME_Spearman_mean"])

    def _max_jaccard(model: str) -> float:
        keys = [k for k in tb[model] if k.endswith("_jaccard")]
        return max(float(tb[model][k]) for k in keys) if keys else 0.0

    rf_max_j = _max_jaccard("RF")
    xg_max_j = _max_jaccard("XGB")
    max_jaccard = max(rf_max_j, xg_max_j)

    shap_lime_rf = float(tb["RF"].get("SHAP_vs_LIME_spearman", 0))
    shap_lime_xg = float(tb["XGB"].get("SHAP_vs_LIME_spearman", 0))

    lines = [
        "XAI VALIDATION — ONE-PAGE SLIDE SUPPORT",
        "Dual-Model Intrusion Detection (Random Forest + XGBoost)",
        "=" * 64,
        "",
        f"Sample: N={n_label} stratified held-out test rows, seed=42",
        "Methods: SHAP, LIME, ELI5",
        "",
        "-" * 64,
        "SLIDE TEXT (PowerPoint)                 |  EVIDENCE (this run)",
        "-" * 64,
        "",
        "1) STABILITY",
        f"   RF: {rf_stab:.2f} and XGB: {xg_stab:.2f}",
        "   High consistency of explanations",
        f"   Evidence: RF LIME Spearman = {rf_stab:.4f} ({rf['LIME_Spearman_pct_below_0_5']} rows < 0.5)",
        f"             XGB LIME Spearman = {xg_stab:.4f} ({xg['LIME_Spearman_pct_below_0_5']} rows < 0.5)",
        "",
        "2) FIDELITY",
        "   Feature removal changed predictions",
        "   SHAP & LIME show strong influence",
        f"   RF  SHAP k=3: {rf['SHAP_flips_k3']} flips, {rf['SHAP_acc_drop_pct_k3']}% acc drop | "
        f"LIME k=3: {rf['LIME_flips_k3']} flips, {rf['LIME_acc_drop_pct_k3']}% acc drop",
        f"   RF  SHAP k=5: {rf['SHAP_flips_k5']} flips, {rf['SHAP_acc_drop_pct_k5']}% acc drop | "
        f"LIME k=5: {rf['LIME_flips_k5']} flips, {rf['LIME_acc_drop_pct_k5']}% acc drop",
        f"   XGB SHAP k=3: {xg['SHAP_flips_k3']} flips, {xg['SHAP_acc_drop_pct_k3']}% acc drop | "
        f"LIME k=3: {xg['LIME_flips_k3']} flips, {xg['LIME_acc_drop_pct_k3']}% acc drop",
        f"   XGB SHAP k=5: {xg['SHAP_flips_k5']} flips, {xg['SHAP_acc_drop_pct_k5']}% acc drop | "
        f"LIME k=5: {xg['LIME_flips_k5']} flips, {xg['LIME_acc_drop_pct_k5']}% acc drop",
        "",
        "3) RUNTIME (mean seconds)",
        "   SHAP (RF): Fast | LIME: Moderate | ELI5: Slower",
        f"   RF  SHAP {rf['SHAP_mean_s']} s | LIME {rf['LIME_mean_s']} s | ELI5 {rf['ELI5_one_off_s']} s",
        f"   XGB SHAP {xg['SHAP_mean_s']} s | LIME {xg['LIME_mean_s']} s | ELI5 {xg['ELI5_one_off_s']} s",
        "",
        "4) AGREEMENT",
        f"   Jaccard up to {max_jaccard:.2f}",
        f"   Spearman around 0.4 to 0.5 (SHAP vs LIME)",
        "   Together they provide broader model understanding",
        f"   RF  SHAP vs LIME: Spearman {shap_lime_rf:.4f}, Jaccard {tb['RF'].get('SHAP_vs_LIME_jaccard', 'n/a')}",
        f"   XGB SHAP vs LIME: Spearman {shap_lime_xg:.4f}, Jaccard {tb['XGB'].get('SHAP_vs_LIME_jaccard', 'n/a')}",
        f"   Highest Jaccard: {max_jaccard:.4f} (XGB LIME vs ELI5 = {tb['XGB'].get('LIME_vs_ELI5_jaccard', 'n/a')})",
        "",
        "-" * 64,
        "QUICK TABLE FOR VIVA / EXAMINER",
        "-" * 64,
        "",
        f"{'Metric':<28} | {'RF':<12} | {'XGBoost':<12}",
        f"{'-'*28}-+-{'-'*12}-+-{'-'*12}",
        f"{'LIME stability (mean)':<28} | {rf_stab:<12.2f} | {xg_stab:<12.2f}",
        f"{'SHAP flips (k=3)':<28} | {rf['SHAP_flips_k3']:<12} | {xg['SHAP_flips_k3']:<12}",
        f"{'LIME flips (k=3)':<28} | {rf['LIME_flips_k3']:<12} | {xg['LIME_flips_k3']:<12}",
        f"{'SHAP runtime mean (s)':<28} | {rf['SHAP_mean_s']:<12} | {xg['SHAP_mean_s']:<12}",
        f"{'LIME runtime mean (s)':<28} | {rf['LIME_mean_s']:<12} | {xg['LIME_mean_s']:<12}",
        f"{'SHAP vs LIME Spearman':<28} | {shap_lime_rf:<12.2f} | {shap_lime_xg:<12.2f}",
        f"{'Best Jaccard (top-10)':<28} | {rf_max_j:<12.2f} | {xg_max_j:<12.2f}",
        "",
        "Global training importance charts: 01_RF_...png, 02_XGB_...png",
        "",
        "Regenerate: python build_final_ppt_evidence.py",
    ]
    out_path = OUT / "07_XAI_SLIDE_RESULTS_ONE_PAGE.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {out_path.name}")


def _write_readme(use_n100: bool) -> None:
    n_label = "100" if use_n100 else "500"
    readme = OUT / "README.txt"
    readme.write_text(
        "Final PPT Evidence - Dual-Model Intrusion Detection\n"
        "---------------------------------------------------\n\n"
        "Global feature importance (training):\n"
        "  01_RF_global_feature_importance.png\n"
        "  02_XGB_global_feature_importance.png\n\n"
        f"XAI validation (N={n_label} stratified test sample):\n"
        "  07_XAI_SLIDE_RESULTS_ONE_PAGE.txt  <- START HERE for slides/viva\n"
        "  03_XAI_validation_PPT_snippet.txt\n"
        "  04_XAI_table_a_metrics.txt\n"
        "  05_XAI_table_b_agreement.txt\n"
        "  06_XAI_full_evaluation_results.txt\n\n"
        "Regenerate: python build_final_ppt_evidence.py\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Final_PPT_Evidence folder")
    parser.add_argument("--n100", action="store_true", help="Use N=100 XAI sample (faster)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluate_xai.py")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    PRES.mkdir(parents=True, exist_ok=True)

    print("Copying global feature-importance charts...")
    for src, dst in IMAGE_COPIES:
        if not src.exists():
            raise FileNotFoundError(f"Missing {src}. Run training scripts first.")
        shutil.copy2(src, dst)
        print(f"  copied {dst.name}")

    if not args.skip_eval and not _xai_outputs_present(args.n100):
        _run_xai_eval(args.n100)

    print("Copying XAI validation reports...")
    _copy_xai_text(args.n100)
    _write_slide_one_page(args.n100)
    _write_readme(args.n100)
    print(f"\nDone. Evidence folder: {OUT}")


if __name__ == "__main__":
    main()
