"""
Build OpenAI prompts from model outputs, raw CSV row, and XAI text.
Advisory only for human review.
"""

from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from xai_insight_language import (
    format_eli5_line_for_llm,
    format_lime_line_for_llm,
    format_shap_line_for_llm,
)

SYSTEM_PROMPT = """You are a network security analyst writing brief, clear notes for engineers.

Your output is **AI-generated advisory text** for human review — not an automatic verdict. The reader must verify with investigation before enforcement.

Context:
- Two ML models only output **Attack** or **Normal**. They do **not** output attack type.
- You receive **dashboard XAI summaries** (plain language, per model), **raw CSV field values**, and **explainability detail** (SHAP/LIME/ELI5).
- Align your narrative with the dashboard summaries first; use SHAP/LIME only as supporting detail.
- SHAP/LIME numbers are **explanation contributions**, NOT the raw measurement from the flow. Never describe a SHAP value as if it were bytes, packets, or milliseconds.

Writing style:
- Short sentences. Plain English. No jargon dumps.
- Lead with what matters, then evidence.
- Do not repeat the same point in every section.
- Be confident about evidence; be cautious about attack labels.

Rules (no hallucination):
- Use ONLY facts in the user message. Do not invent IPs, CVEs, malware, tools, or campaigns.
- When citing a field, use the **raw CSV value** from the message. When citing explainability, say it is from SHAP/LIME/ELI5.
- If attack type cannot be justified, say so clearly in one sentence. Do not invent an attack family.

Output Markdown in exactly this order:

## Situation
2–3 sentences: what both models predicted, confidence, whether they agree, and the single most important pattern in plain language.

## What stands out
3–5 bullets. Each bullet: plain-language observation → cite **raw field name + value** and/or explainability driver. Separate raw values from SHAP/LIME scores.

## Behaviour pattern (interpretive)
If evidence supports a plausible pattern (e.g. volume-heavy, scan-like, irregular return traffic): one cautious sentence naming it, then 2–3 short evidence bullets.
Otherwise: one clear sentence that you cannot determine attack type from this row alone, plus why.
Do not label options as (A) or (B) — write natural prose only.

## Recommended actions
Numbered list, 4–6 items. Practical steps for this row (SIEM lookup, baseline comparison, monitor host, PCAP if repeats). If models disagree, prioritise verification over blocking.

End with this exact line on its own:
*This note is AI-generated advisory text — confirm with investigation before taking enforcement action.*
"""


def _read_secret_from_toml(key: str) -> str:
    roots = (Path.cwd(), Path(__file__).resolve().parent)
    for root in roots:
        path = root / ".streamlit" / "secrets.toml"
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or not line.upper().startswith(key.upper()):
                continue
            m = re.match(rf'^{re.escape(key)}\s*=\s*"(.*)"\s*$', line, re.I) or re.match(
                rf"^{re.escape(key)}\s*=\s*'(.*)'\s*$", line, re.I
            )
            if not m:
                m = re.match(rf"^{re.escape(key)}\s*=\s*(\S+)\s*$", line, re.I)
            if m:
                return m.group(1).strip()
    return ""


def resolve_openai_api_key() -> str:
    k = os.environ.get("OPENAI_API_KEY", "").strip()
    if k:
        return k
    k = _read_secret_from_toml("OPENAI_API_KEY")
    if k:
        return k
    try:
        st = __import__("streamlit")
        sec = getattr(st, "secrets", None)
        if sec is not None and "OPENAI_API_KEY" in sec:
            return str(sec["OPENAI_API_KEY"]).strip()
    except Exception:
        pass
    return ""


def resolve_openai_model() -> str:
    return (
        os.environ.get("OPENAI_MODEL", "").strip()
        or _read_secret_from_toml("OPENAI_MODEL")
        or "gpt-4o"
    )


def row_needs_llm_advisory(rf_label: str, xgb_label: str) -> bool:
    """True when at least one Attack or models disagree"""
    return not (str(rf_label) == "Normal" and str(xgb_label) == "Normal")


def format_raw_csv_row(df: pd.DataFrame, row_idx: int, max_fields: int = 50) -> str:
    if row_idx < 0 or row_idx >= len(df):
        return "(invalid row index)"
    row = df.iloc[row_idx]
    parts: List[str] = []
    for i, (col, val) in enumerate(row.items()):
        if i >= max_fields:
            parts.append(f"... ({len(row) - max_fields} more columns omitted)")
            break
        try:
            if pd.isna(val):
                s = ""
            else:
                s = str(val)
                if len(s) > 200:
                    s = s[:200] + "…"
        except Exception:
            s = str(val)
        parts.append(f"- **{col}**: {s}")
    return "\n".join(parts)


def _shap_top_lines(
    explainer: Any,
    x_row: np.ndarray,
    feature_names: List[str],
    pred_idx: int,
    predicted_class: str,
    top_n: int = 12,
) -> str:
    if explainer is None:
        return "(SHAP not available)"
    try:
        x_row = np.asarray(x_row, dtype=np.float64)
        sv = explainer.shap_values(x_row)
        if isinstance(sv, list):
            safe_idx = min(int(pred_idx), len(sv) - 1)
            safe_idx = max(0, safe_idx)
            inner = sv[safe_idx]
            shap_vals = (
                np.array(inner[0]).flatten()
                if hasattr(inner, "__len__") and len(inner)
                else np.array(sv[0][0]).flatten()
            )
        else:
            sv = np.asarray(sv)
            if len(sv.shape) == 3:
                c = min(int(pred_idx), sv.shape[2] - 1)
                shap_vals = sv[0, :, c].flatten()
            else:
                shap_vals = sv[0].flatten()
        n_f = len(feature_names)
        if len(shap_vals) < n_f:
            shap_vals = np.pad(shap_vals, (0, n_f - len(shap_vals)))
        elif len(shap_vals) > n_f:
            shap_vals = shap_vals[:n_f]
        pairs = sorted(
            [(feature_names[i], float(shap_vals[i])) for i in range(n_f)],
            key=lambda t: abs(t[1]),
            reverse=True,
        )[:top_n]
        return "\n".join(
            format_shap_line_for_llm(name, val, predicted_class) for name, val in pairs
        )
    except Exception as e:
        return f"(SHAP extraction failed: {e})"


def _lime_top_lines(
    explainer: Any,
    x_row: np.ndarray,
    predict_proba: Callable,
    pred_idx: int,
    predicted_class: str,
    top_n: int = 12,
) -> str:
    if explainer is None:
        return "(LIME not available)"
    try:
        safe_label = max(0, min(int(pred_idx), 1))
        exp = explainer.explain_instance(
            x_row[0], predict_proba, num_features=top_n, top_labels=2
        )
        lst = exp.as_list(label=safe_label)
        return "\n".join(
            format_lime_line_for_llm(cond, float(w), predicted_class)
            for cond, w in lst[:top_n]
        )
    except Exception as e:
        return f"(LIME extraction failed: {e})"


def _eli5_global_top_lines(
    perm_importance: Any,
    feature_names: List[str],
    top_n: int = 15,
) -> str:
    """Top global features from ELI5 permutation importance """
    if perm_importance is None or not hasattr(perm_importance, "feature_importances_"):
        return "(ELI5 global importance not available)"
    try:
        imp = perm_importance.feature_importances_
        pairs = sorted(
            [(feature_names[i], float(imp[i])) for i in range(min(len(feature_names), len(imp)))],
            key=lambda t: abs(t[1]),
            reverse=True,
        )[:top_n]
        return "\n".join(
            format_eli5_line_for_llm(name, val) for name, val in pairs
        )
    except Exception as e:
        return f"(ELI5 extraction failed: {e})"


def build_user_prompt(
    csv_name: str,
    row_1based: int,
    rf_pred: str,
    rf_conf: float,
    xgb_pred: str,
    xgb_conf: float,
    models_agree: bool,
    raw_row_markdown: str,
    rf_xai_summary_block: str,
    xgb_xai_summary_block: str,
    rf_shap_block: str,
    rf_lime_block: str,
    xgb_shap_block: str,
    xgb_lime_block: str,
    rf_eli5_block: str,
    xgb_eli5_block: str,
) -> str:
    if models_agree:
        agree_txt = f"Yes — both predict **{rf_pred}**."
        driver_note = (
            "Models agree on the label; compare RF vs XGB dashboard summaries below — "
            "they may emphasize different drivers."
        )
    else:
        agree_txt = (
            f"No — Random Forest says **{rf_pred}**, XGBoost says **{xgb_pred}**. "
            "Treat with extra caution."
        )
        driver_note = "Models disagree — prioritise verification; do not auto-block."
    return f"""## Flow under review

**File:** {csv_name}
**Row:** {row_1based}

### Model outputs (Attack/Normal only)
- **Random Forest:** {rf_pred} ({rf_conf:.2f}% confidence)
- **XGBoost:** {xgb_pred} ({xgb_conf:.2f}% confidence)
- **Agreement:** {agree_txt}
- **Note:** {driver_note}

### Dashboard XAI summary — Random Forest 
{rf_xai_summary_block}

### Dashboard XAI summary — XGBoost 
{xgb_xai_summary_block}

### Raw CSV values for the row
{raw_row_markdown}

### Explainability detail — Random Forest (for predicted class: {rf_pred})
**SHAP (this row):**
{rf_shap_block}

**LIME (this row):**
{rf_lime_block}

**ELI5 (global model importance):**
{rf_eli5_block}

### Explainability detail — XGBoost (for predicted class: {xgb_pred})
**SHAP (this row):**
{xgb_shap_block}

**LIME (this row):**
{xgb_lime_block}

**ELI5 (global model importance):**
{xgb_eli5_block}

---

Write for an engineer reviewing this one flow. Use the section headings from your instructions.
Treat the dashboard XAI summaries as the primary story for each model; use raw CSV for measurements and SHAP/LIME only as supporting detail.
Remember: SHAP/LIME numbers are explanation scores, not raw traffic measurements.
"""


def call_openai_advisory(user_content: str) -> Tuple[Optional[str], str]:
    api_key = resolve_openai_api_key()
    if not api_key:
        return None, "OPENAI_API_KEY not set (environment or .streamlit/secrets.toml)"

    model = resolve_openai_model()
    try:
        import urllib.request
    except ImportError:
        return None, "urllib not available"

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.25,
        "max_tokens": 1100,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"]
        return text.strip(), "OK"
    except Exception as e:
        try:
            if hasattr(e, "read"):
                detail = e.read().decode("utf-8", errors="replace")[:500]
                return None, f"OpenAI API error: {e} — {detail}"
        except Exception:
            pass
        return None, f"OpenAI API error: {e}"
