"""
Teams webhook summary after batch prediction. 
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np


def _read_teams_webhook_from_secrets_toml() -> str:
   
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
            if not line or not line.upper().startswith("TEAMS_WEBHOOK_URL"):
                continue
            m = re.match(
                r'^TEAMS_WEBHOOK_URL\s*=\s*"(.*)"\s*$',
                line,
            ) or re.match(r"^TEAMS_WEBHOOK_URL\s*=\s*'(.*)'\s*$", line)
            if m:
                u = m.group(1).strip()
                if u:
                    return u
    return ""


def teams_upload_fingerprint(
    batch_name: str,
    rf_pred: np.ndarray,
    xgb_pred: np.ndarray,
) -> str:
    #Stable id for one upload+batch result
    h = hashlib.md5()
    h.update(batch_name.encode("utf-8", errors="replace"))
    h.update(np.asarray(rf_pred, dtype=np.int64).tobytes())
    h.update(np.asarray(xgb_pred, dtype=np.int64).tobytes())
    return h.hexdigest()


def resolve_teams_webhook_url() -> str:
    """
    Webhook URL from (in order):
    1) TEAMS_WEBHOOK_URL environment variable
    2) .streamlit/secrets.toml (works for CLI and Streamlit)
    3) streamlit.secrets when the Streamlit app is running
    """
    url = os.environ.get("TEAMS_WEBHOOK_URL", "").strip()
    if url:
        return url
    url = _read_teams_webhook_from_secrets_toml()
    if url:
        return url
    try:
        st = __import__("streamlit")
        sec = getattr(st, "secrets", None)
        if sec is not None and "TEAMS_WEBHOOK_URL" in sec:
            return str(sec["TEAMS_WEBHOOK_URL"]).strip()
    except Exception:
        pass
    return ""


def compute_batch_stats(
    class_names_rf: list,
    class_names_xgb: list,
    rf_pred: np.ndarray,
    xgb_pred: np.ndarray,
) -> dict:
 
    rf_labels = np.array([str(class_names_rf[int(p)]) for p in rf_pred])
    xgb_labels = np.array([str(class_names_xgb[int(p)]) for p in xgb_pred])
    both_normal = int(np.sum((rf_labels == "Normal") & (xgb_labels == "Normal")))
    both_attack = int(np.sum((rf_labels == "Attack") & (xgb_labels == "Attack")))
    disagree = int(np.sum(rf_labels != xgb_labels))
    return {
        "n_analyzed": len(rf_pred),
        "both_normal": both_normal,
        "both_attack": both_attack,
        "disagree": disagree,
    }


def _teams_status_markdown(
    both_attack: int,
    both_normal: int,
    disagree: int,
) -> str:
    """Short markdown block for MessageCard (facts above carry the numbers)."""
    lines: list[str] = []
    if both_attack == 0 and disagree == 0:
        lines.append("**Status: Normal** — no flows predicted as Attack; models agree everywhere.")
        return "\n\n".join(lines)
    lines.append("**Status: Needs review**")
    if both_attack > 0:
        lines.append(
            f"- **{both_attack}** flows: Random Forest and XGBoost both predict **Attack** (treat as higher confidence)."
        )
    if disagree > 0:
        lines.append(
            f"- **{disagree}** flows: models **disagree** — inspect these before bulk actions."
        )
    if both_normal > 0:
        lines.append(f"- **{both_normal}** flows: both models predict **Normal**.")
    lines.append("\nSuggested next step: review flagged rows in the dashboard or export for SIEM / analyst follow-up.")
    return "\n\n".join(lines)


def build_teams_payload(
    batch_name: str,
    n_analyzed: int,
    elapsed_s: float,
    both_normal: int,
    both_attack: int,
    disagree: int,
    completed_iso: Optional[str] = None,
) -> dict[str, Any]:
    """
    Office 365 Connector MessageCard — renders as labeled rows (facts) in Teams, not one dense line.
    """
    if completed_iso is None:
        completed_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    needs_review = both_attack > 0 or disagree > 0
    # Green = OK, orange = attention 
    theme_color = "D83B01" if needs_review else "107C10"
    summary = (
        f"IDS: {'Needs review' if needs_review else 'Normal'} — {batch_name} ({n_analyzed} flows)"
    )
    section_text = _teams_status_markdown(both_attack, both_normal, disagree)

    return {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": theme_color,
        "summary": summary,
        "sections": [
            {
                "activityTitle": "Intrusion Detection Analysis Summary Report",
                "activitySubtitle": batch_name,
                "markdown": True,
                "facts": [
                    {"name": "Records analyzed", "value": str(n_analyzed)},
                    {"name": "Completed (UTC)", "value": completed_iso},
                    {"name": "Run time", "value": f"{elapsed_s:.1f} s"},
                    {"name": "Both Normal (RF + XGB)", "value": str(both_normal)},
                    {"name": "Both Attack (RF + XGB)", "value": str(both_attack)},
                    {"name": "Models Disagree", "value": str(disagree)},
                ],
                "text": section_text,
            }
        ],
    }


def send_teams_payload(
    payload: dict[str, Any],
    webhook_url: Optional[str] = None,
) -> Tuple[bool, str]:
    url = (webhook_url or resolve_teams_webhook_url()).strip()
    if not url:
        return False, "TEAMS_WEBHOOK_URL missing (env or .streamlit/secrets.toml)"
    try:
        import requests
    except ImportError:
        return False, "Install requests: pip install requests"
    try:
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code == 200:
            return True, "OK"
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, str(e)


def send_executive_summary(
    batch_name: str,
    class_names_rf: list,
    class_names_xgb: list,
    rf_pred: np.ndarray,
    xgb_pred: np.ndarray,
    elapsed_s: float,
    webhook_url: Optional[str] = None,
) -> Tuple[bool, str]:
    stats = compute_batch_stats(class_names_rf, class_names_xgb, rf_pred, xgb_pred)
    payload = build_teams_payload(
        batch_name=batch_name,
        n_analyzed=stats["n_analyzed"],
        elapsed_s=elapsed_s,
        both_normal=stats["both_normal"],
        both_attack=stats["both_attack"],
        disagree=stats["disagree"],
    )
    return send_teams_payload(payload, webhook_url=webhook_url)
