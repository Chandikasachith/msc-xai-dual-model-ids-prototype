# Prototype Workflow: Data Input → Explanation Output

End-to-end pipeline for the dual-model (RF + XGB) CICIDS-2017 intrusion detection system with explainable AI.

---

## Overview

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  Data       │───▶│  Preprocessing    │───▶│  Prediction     │───▶│  Explanation     │
│  Input      │    │  (normalize,      │    │  (RF + XGB      │    │  (SHAP, LIME,     │
│  (CSV)      │    │   encode, scale,  │    │   Normal/Attack)│    │   text report)    │
│             │    │   select)         │    │                 │    │                  │
└─────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
```

---

## 1. Data input

- **Format:** CSV with CICIDS-2017–style columns (or CICIDS-2018 aliases; see `cicids2017_preprocess.py` for mapping).
- **Optional column:** `Label` (values like Benign/Normal → Normal, others → Attack). If present, accuracy and match counts are reported.
- **Sources:** Uploaded file, or built-in samples (e.g. `test_2018.csv`, `cicids2017_holdout_200.csv`).

---

## 2. Preprocessing

Same logic as the dashboard and `run_inference_on_csv.py`:

1. **Normalize columns** – Rename 2018/alternate names to 2017 training names.
2. **Drop non-features** – e.g. Flow ID, Source IP, Destination IP, Timestamp.
3. **Engineer features** – RF: full set (e.g. `total_bytes`, `bytes_ratio`, `data_transfer_rate`, …). XGB: minimal set (e.g. `total_bytes`, `total_packets`).
4. **Encode categoricals** – Using saved label encoders per model.
5. **Align columns** – Match scaler’s expected features; missing → 0.
6. **Scale** – RF scaler and XGB scaler (fit at training time).
7. **Feature selection** – RF and XGB selectors (e.g. variance threshold) → final feature matrix per model.

Output: `X_rf` (n × p_rf), `X_xgb` (n × p_xgb), ready for the two models.

---

## 3. Prediction

- **RF:** `rf_model.predict(X_rf)`, `predict_proba(X_rf)` → class (Normal/Attack) and confidence.
- **XGB:** `xgb_model.predict(X_xgb)`, `predict_proba(X_xgb)` → class and confidence.

Optional: compare with true `Label` and report accuracy and confusion matrix.

---

## 4. Explanation output

For selected rows (e.g. first 3 or user-specified indices):

- **SHAP** – Per-row feature contributions (TreeExplainer). Top positive/negative drivers of Normal vs Attack.
- **LIME** (optional) – Local linear approximation; which features push toward Attack vs Normal.
- **ELI5** (optional) – Global permutation importance (model-level, not per row).

Outputs:

- **Console** – Short summary per explained row.
- **Report file** – Human-readable text: prediction, confidence, top SHAP features, and a one-paragraph explanation.

---

## 5. How to run the prototype

**Inference only (no XAI):**

```bash
python run_inference_on_csv.py [path/to/file.csv]
# Default: test_2018.csv
```

**Full workflow (data → prediction → explanations):**

```bash
python workflow_data_to_explanation.py [path/to/file.csv] [--rows 3] [--out report.txt]
# Default CSV: test_2018.csv; explains first 3 rows; writes workflow_explanations.txt
```

**Prerequisites:** Train and save models first so that `SecondModel/` contains:
`rf_model_binary_cicids2017.pkl`, `scaler_rf_cicids2017.pkl`, `feature_selector_rf_cicids2017.pkl`,
`xgb_model_binary_cicids2017_95percent.pkl`, `scaler_xgb_cicids2017_95percent.pkl`,
`feature_selector_xgb_cicids2017_95percent.pkl`, and the corresponding `target_encoder_*.pkl` files.
Optional: `feature_names_rf_cicids2017.pkl` and `feature_names_xgb_cicids2017_95percent.pkl` for readable feature names in the report.

Training also saves **full** `X_test_*` / `y_test_*` tensors (entire 15% test fold). To regenerate dual-model figures and metrics for the thesis, run from the project root: `python generate_cicids2017_dual_model_evidence.py` (outputs under `Presentation_Evidence/CICIDS2017_Dual_Model_Evidence/`).

**XAI validation (progress report / PPT):** Run `python evaluate_xai.py`. It writes `Presentation_Evidence/xai_thesis_table_a_metrics.txt`, `xai_thesis_table_b_agreement.txt`, and **`XAI_VALIDATION_PROGRESS_PPT.txt`** (ready-to-paste validation wording + latest numbers).

**OpenAI row advisory:** Add `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`) to `.streamlit/secrets.toml`. In `streamlit_dashboard.py`, after uploading a CSV, use **Generate AI analyst suggestions** when the row has **Attack from either model** or **RF/XGB disagree**. Sends raw row fields plus SHAP/LIME summaries only; output is advisory.

**Microsoft Teams executive summary:** Prefer **`.streamlit/secrets.toml`**: copy `secrets.toml.example` to `secrets.toml` and set `TEAMS_WEBHOOK_URL = "https://..."` (this file is gitignored). The same file is read by **Streamlit, CLI scripts, and `test_teams_webhook.py`** via `teams_executive_summary.resolve_teams_webhook_url()`. Optionally override with the `TEAMS_WEBHOOK_URL` environment variable. Then:

- `run_inference_on_csv.py` and `workflow_data_to_explanation.py` post a short summary after a successful run.
- `streamlit_dashboard.py`: after CSV upload and batch predictions, an **executive summary is sent to Teams automatically once** (per upload; reruns are deduplicated). Optional **Resend summary to Teams** if you need another post.

---

## 6. Files involved

| Step           | Script / component                                      | Artifacts (SecondModel/)                                                                 |
|----------------|---------------------------------------------------------|-------------------------------------------------------------------------------------------|
| Load           | `run_inference_on_csv.py`, `workflow_data_to_explanation.py` | `*_model_*.pkl`, `scaler_*.pkl`, `feature_selector_*.pkl`, `label_encoders_*.pkl`, `target_encoder_*.pkl`, optional `feature_names_*.pkl` |
| Preprocess     | `cicids2017_preprocess.py` (`normalize_upload_columns`, `preprocess_for_rf`, `preprocess_for_xgb`) | —                                                                                         |
| Predict        | RF and XGB `.predict` / `predict_proba`                 | —                                                                                         |
| Explain        | SHAP `TreeExplainer`, optional LIME/ELI5                | Optional: `X_test_rf_cicids2017.npy`, `X_test_xgb_cicids2017_95percent.npy` for LIME/ELI5 |
| Teams summary  | `teams_executive_summary.py`                            | Env: `TEAMS_WEBHOOK_URL` |

---

## 7. Summary

- **Input:** CSV (CICIDS-2017/2018 style).
- **Preprocessing:** Normalize → engineer → encode → scale → select (per model).
- **Prediction:** RF and XGB binary (Normal/Attack) + confidence.
- **Explanation:** SHAP (and optionally LIME/ELI5) with a text report for selected rows.

This defines the **prototype workflow from data input to explanation output** used for the thesis/presentation.
