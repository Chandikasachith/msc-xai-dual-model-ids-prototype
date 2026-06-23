"""
Readable XAI text for the dashboard and LLM prompts.
.
"""

from __future__ import annotations

from html import escape
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

AI_DISCLAIMER_MARKDOWN = (
    "**AI-generated advisory** — for guidance only. "
    "Confirm with logs, SIEM, and human review before blocking, quarantining, or escalation."
)

AI_DISCLAIMER_SHORT = (
    "AI-generated — use after investigation; do not treat as automatic enforcement."
)

FEATURE_GLOSSARY: Dict[str, Dict[str, str]] = {
    "total_bytes": {
        "label": "Total data volume",
        "meaning": "Combined forward and backward bytes; unusually high volume can indicate flooding or bulk transfer.",
    },
    "bytes_ratio": {
        "label": "Forward/backward byte ratio",
        "meaning": "Skew in traffic direction; asymmetric flows can reflect one-way abuse or scanning.",
    },
    "total_packets": {
        "label": "Total packet count",
        "meaning": "How many packets were exchanged; spikes may reflect bursts or automated traffic.",
    },
    "packets_ratio": {
        "label": "Forward/backward packet ratio",
        "meaning": "Directional packet imbalance; relevant for probe-like or one-sided traffic.",
    },
    "data_transfer_rate": {
        "label": "Data transfer rate",
        "meaning": "Bytes per unit time; very high rates are often seen in volume-based attacks.",
    },
    "packets_per_second": {
        "label": "Packet rate",
        "meaning": "Packets per unit time; rapid rates can indicate automated or flood behaviour.",
    },
    "fwd_packets_rate": {
        "label": "Forward packet rate",
        "meaning": "Outbound packet frequency from the source side of the flow.",
    },
    "bwd_packets_rate": {
        "label": "Backward packet rate",
        "meaning": "Return-path packet frequency.",
    },
    "Flow Duration": {
        "label": "Flow duration",
        "meaning": "How long the connection lasted; very short or very long flows can be atypical.",
    },
    "Flow Bytes/s": {
        "label": "Flow byte rate",
        "meaning": "Overall throughput for the flow.",
    },
    "Flow Packets/s": {
        "label": "Flow packet rate",
        "meaning": "Overall packet throughput for the flow.",
    },
    "Destination Port": {
        "label": "Destination port",
        "meaning": "Service targeted; unusual ports can indicate probing or service-specific attacks.",
    },
    "Total Fwd Packets": {
        "label": "Forward packets",
        "meaning": "Packets sent in the forward direction.",
    },
    "Total Backward Packets": {
        "label": "Backward packets",
        "meaning": "Packets sent in the return direction.",
    },
    "Total Bwd Packets": {
        "label": "Backward packets",
        "meaning": "Packets sent in the return direction.",
    },
    "Total Length of Fwd Packets": {
        "label": "Forward data size",
        "meaning": "Volume of data sent forward.",
    },
    "Total Length of Bwd Packets": {
        "label": "Backward data size",
        "meaning": "Volume of data sent back.",
    },
    "Fwd Packets Length Total": {
        "label": "Forward data size",
        "meaning": "Volume of data sent forward.",
    },
    "Bwd Packets Length Total": {
        "label": "Backward data size",
        "meaning": "Volume of data sent back.",
    },
    "Bwd Packet Length Std": {
        "label": "Backward packet size variability",
        "meaning": "How much backward packet sizes vary; high variability can indicate irregular or bursty return traffic.",
    },
    "Bwd Packet Length Mean": {
        "label": "Average backward packet size",
        "meaning": "Typical size of packets on the return path; unusually large values can be suspicious.",
    },
    "Bwd Packet Length Max": {
        "label": "Maximum backward packet size",
        "meaning": "Largest backward packet observed in the flow.",
    },
    "Fwd Packet Length Std": {
        "label": "Forward packet size variability",
        "meaning": "Variation in forward packet sizes.",
    },
    "Fwd Packet Length Mean": {
        "label": "Average forward packet size",
        "meaning": "Typical forward packet size.",
    },
    "Packet Length Std": {
        "label": "Packet size variability",
        "meaning": "Overall variation in packet sizes across the flow.",
    },
    "Packet Length Mean": {
        "label": "Average packet size",
        "meaning": "Typical packet size across the flow.",
    },
    "Packet Length Variance": {
        "label": "Packet size variance",
        "meaning": "Spread of packet sizes; high variance means uneven packet sizes.",
    },
    "Avg Packet Size": {
        "label": "Average packet size",
        "meaning": "Typical packet size in the flow.",
    },
    "Avg Bwd Segment Size": {
        "label": "Average backward segment size",
        "meaning": "Typical size of backward TCP segments; large values can reflect bulky return traffic.",
    },
    "Avg Fwd Segment Size": {
        "label": "Average forward segment size",
        "meaning": "Typical size of forward TCP segments.",
    },
    "Fwd IAT Std": {
        "label": "Forward timing variability",
        "meaning": "Variation in time gaps between forward packets; irregular timing can reflect automated or bursty sending.",
    },
    "Fwd IAT Max": {
        "label": "Longest forward packet gap",
        "meaning": "Maximum time between forward packets; very long gaps can mean idle or intermittent sending.",
    },
    "Fwd IAT Mean": {
        "label": "Average forward packet gap",
        "meaning": "Typical time between forward packets.",
    },
    "Bwd IAT Std": {
        "label": "Backward timing variability",
        "meaning": "Variation in time gaps between backward packets.",
    },
    "SYN Flag Count": {
        "label": "SYN flag count",
        "meaning": "TCP SYN usage; elevated counts can relate to connection attempts or SYN floods.",
    },
    "ACK Flag Count": {
        "label": "ACK flag count",
        "meaning": "TCP ACK usage; helps characterise established vs half-open behaviour.",
    },
    "FIN Flag Count": {
        "label": "FIN flag count",
        "meaning": "Connection teardown signals in the flow.",
    },
    "RST Flag Count": {
        "label": "RST flag count",
        "meaning": "Abrupt connection resets; can appear in scans or failed handshakes.",
    },
    "Init Fwd Win Bytes": {
        "label": "Forward TCP window",
        "meaning": "Initial receive window on the forward path.",
    },
    "Init Bwd Win Bytes": {
        "label": "Backward TCP window",
        "meaning": "Initial receive window on the return path.",
    },
    "Subflow Fwd Packets": {
        "label": "Subflow forward packets",
        "meaning": "Packets in forward subflows.",
    },
    "Subflow Bwd Packets": {
        "label": "Subflow backward packets",
        "meaning": "Packets in backward subflows.",
    },
}


def feature_label(feature_name: str) -> str:
    entry = FEATURE_GLOSSARY.get(feature_name)
    if entry:
        return entry["label"]
    return feature_name.replace("_", " ")


def feature_meaning(feature_name: str) -> str:
    entry = FEATURE_GLOSSARY.get(feature_name)
    if entry:
        return entry["meaning"]
    return "Network flow statistic used by the model for this prediction."


def _direction_phrase(impact: float, predicted_class: str) -> str:
    opposite = "Normal" if predicted_class == "Attack" else "Attack"
    if impact > 0:
        return f"pushed this prediction toward **{predicted_class}**"
    if impact < 0:
        return f"pulled this prediction toward **{opposite}** (away from **{predicted_class}**)"
    return "had little effect on the prediction"


def _strength_phrase(pct_share: float) -> str:
    if pct_share >= 35:
        return "main reason"
    if pct_share >= 20:
        return "strong factor"
    if pct_share >= 10:
        return "contributing factor"
    return "minor factor"


def _methods_note(local_methods: Sequence[str], has_eli5: bool) -> str:
    parts: List[str] = []
    if local_methods:
        if len(local_methods) > 1:
            parts.append("SHAP and LIME agree on this row")
        else:
            parts.append(f"seen in {local_methods[0]} for this row")
    if has_eli5:
        parts.append("also important across the model generally")
    return f" _{'; '.join(parts)}._" if parts else ""


def format_xai_reason_bullet(
    index: int,
    reason: Dict[str, Any],
    predicted_class: str,
    pct_share: float,
) -> str:
    feat = reason["feature"]
    label = feature_label(feat)
    direction = _direction_phrase(float(reason.get("impact", 0)), predicted_class)
    strength = _strength_phrase(pct_share)
    local_methods = [m for m in reason.get("methods", []) if m in ("SHAP", "LIME")]
    has_eli5 = "ELI5" in reason.get("methods", [])
    methods = _methods_note(local_methods, has_eli5)
    meaning = feature_meaning(feat)
    return (
        f"{index}. **{label}** — {meaning} "
        f"This {direction} and is a **{strength}** here (~{pct_share:.0f}% of influence from the top drivers)."
        f"{methods}"
    )


def _strength_badge_class(pct_share: float) -> str:
    if pct_share >= 35:
        return "xai-badge-strong"
    if pct_share >= 20:
        return "xai-badge-medium"
    if pct_share >= 10:
        return "xai-badge-mild"
    return "xai-badge-minor"


def _method_chips_html(local_methods: Sequence[str], has_eli5: bool) -> str:
    chips: List[str] = []
    if len(local_methods) > 1:
        chips.append('<span class="xai-chip xai-chip-agree">SHAP + LIME agree</span>')
    elif len(local_methods) == 1:
        chips.append(f'<span class="xai-chip">{local_methods[0]}</span>')
    if has_eli5:
        chips.append('<span class="xai-chip xai-chip-global">Model-wide</span>')
    if not chips:
        return ""
    return '<div class="xai-driver-chips">' + "".join(chips) + "</div>"


def _popup_direction_html(impact: float, predicted_class: str) -> str:
    opposite = "Normal" if predicted_class == "Attack" else "Attack"
    if impact > 0:
        return f'Pushed toward <strong>{escape(predicted_class)}</strong>'
    if impact < 0:
        return f'Pulled toward <strong>{escape(opposite)}</strong>'
    return "Had little effect on this prediction"


def _format_driver_popup_html(
    reason: Dict[str, Any],
    predicted_class: str,
    pct_share: float,
) -> str:
    """Rich hover panel with full driver detail."""
    feat = reason["feature"]
    label = escape(feature_label(feat))
    raw_name = escape(feat)
    meaning = escape(feature_meaning(feat))
    strength = escape(_strength_phrase(pct_share))
    impact = float(reason.get("impact", 0))
    direction = _popup_direction_html(impact, predicted_class)
    methods = ", ".join(reason.get("methods", [])) or "—"

    rows = [
        f'<div class="xai-popup-row"><span>Influence share</span><strong>{pct_share:.0f}%</strong></div>',
        f'<div class="xai-popup-row"><span>Strength</span><strong>{strength}</strong></div>',
        f'<div class="xai-popup-row"><span>Direction</span><span>{direction}</span></div>',
        f'<div class="xai-popup-row"><span>XAI methods</span><strong>{escape(methods)}</strong></div>',
    ]
    if "value" in reason:
        rows.insert(
            0,
            f'<div class="xai-popup-row"><span>Value in this row</span>'
            f'<strong>{float(reason["value"]):.4g}</strong></div>',
        )
    if reason.get("shap_impact") is not None:
        rows.append(
            f'<div class="xai-popup-row"><span>SHAP impact</span>'
            f'<strong>{float(reason["shap_impact"]):+.4f}</strong></div>'
        )
    if reason.get("lime_impact") is not None:
        rows.append(
            f'<div class="xai-popup-row"><span>LIME weight</span>'
            f'<strong>{float(reason["lime_impact"]):+.4f}</strong></div>'
        )

    return (
        f'<div class="xai-driver-popup">'
        f'<div class="xai-popup-title">{label}</div>'
        f'<div class="xai-popup-field">{raw_name}</div>'
        f'<div class="xai-popup-body">{meaning}</div>'
        f'<div class="xai-popup-grid">{"".join(rows)}</div>'
        f'</div>'
    )

def format_xai_driver_card_html(
    index: int,
    reason: Dict[str, Any],
    predicted_class: str,
    pct_share: float,
) -> str:
    """HTML card for dashboard — highlights feature, strength, % and XAI methods."""
    feat = reason["feature"]
    label = feature_label(feat)
    strength = _strength_phrase(pct_share)
    badge_class = _strength_badge_class(pct_share)
    meaning = feature_meaning(feat)
    local_methods = [m for m in reason.get("methods", []) if m in ("SHAP", "LIME")]
    has_eli5 = "ELI5" in reason.get("methods", [])
    verdict_class = "xai-toward-attack" if predicted_class == "Attack" else "xai-toward-normal"
    chips = _method_chips_html(local_methods, has_eli5)
    popup = _format_driver_popup_html(reason, predicted_class, pct_share)
    return (
        f'<div class="xai-driver-card-wrap">'
        f'<div class="xai-driver-card">'
        f'<div class="xai-driver-top">'
        f'<span class="xai-driver-index">{index}</span>'
        f'<span class="xai-driver-name">{escape(label)}</span>'
        f'<span class="xai-driver-pct">{pct_share:.0f}%</span>'
        f'</div>'
        f'<div class="xai-driver-meta">'
        f'<span class="xai-badge {badge_class}">{strength.replace(" ", "&nbsp;")}</span>'
        f'<span class="xai-driver-toward {verdict_class}">toward {predicted_class}</span>'
        f'<span class="xai-driver-hint">Hover for details</span>'
        f'</div>'
        f'<div class="xai-driver-meaning">{escape(meaning)}</div>'
        f'{chips}'
        f'</div>'
        f'{popup}'
        f'</div>'
    )


def build_xai_driver_cards_html(
    feature_reasons: Sequence[Dict[str, Any]],
    predicted_class: str,
    top_n: int = 5,
) -> str:
    """Stack of driver cards for the dashboard UI."""
    top = feature_reasons[:top_n]
    if not top:
        return ""
    top_impacts = [abs(float(r.get("impact", 0))) for r in top]
    total_impact = sum(top_impacts) if top_impacts else 1.0
    cards = []
    for i, reason in enumerate(top, 1):
        pct = (abs(float(reason.get("impact", 0))) / total_impact * 100) if total_impact > 0 else 0.0
        cards.append(format_xai_driver_card_html(i, reason, predicted_class, pct))
    return '<div class="xai-driver-list">' + "".join(cards) + "</div>"

def _has_opposing_drivers(feature_reasons: Sequence[Dict[str, Any]], predicted_class: str) -> bool:
    toward = 0
    away = 0
    for r in feature_reasons[:5]:
        impact = float(r.get("impact", 0))
        if impact > 0:
            toward += 1
        elif impact < 0:
            away += 1
    return toward > 0 and away > 0


def synthesize_xai_summary(
    feature_reasons: Sequence[Dict[str, Any]],
    predicted_class: str,
    model_name: str,
) -> str:
    if not feature_reasons:
        return f"No clear drivers were found for this {model_name} **{predicted_class}** prediction."

    top = feature_reasons[0]
    top_label = feature_label(top["feature"])
    second_labels = [feature_label(r["feature"]) for r in feature_reasons[1:3]]
    extra = ""
    if second_labels:
        extra = f", with **{'** and **'.join(second_labels)}** also influential"

    base = (
        f"**{model_name}** predicts **{predicted_class}** mainly because of **{top_label}**"
        f"{extra}."
    )
    if _has_opposing_drivers(feature_reasons, predicted_class):
        base += (
            " Some features point the other way, but the model still lands on "
            f"**{predicted_class}** based on the combined effect of all drivers."
        )
    return base


def chart_title(predicted_class: str, model_name: str) -> str:
    return f"What pushed this flow toward {predicted_class} ({model_name})"


def chart_xlabel() -> str:
    return "Influence (+ toward predicted class, − toward the other class)"


def chart_xlabel_toward_prediction() -> str:
    return "Influence toward the predicted class"


def filter_toward_prediction_drivers(
    feature_reasons: Sequence[Dict[str, Any]],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Key drivers for the predicted label: impact > 0 (toward predicted class)."""
    supporting = [r for r in feature_reasons if float(r.get("impact", 0)) > 0]
    return supporting[:top_n]


def chart_feature_labels(feature_reasons: Sequence[Dict[str, Any]]) -> List[str]:
    return [feature_label(r["feature"])[:28] for r in feature_reasons]


def format_shap_line_for_llm(feature_name: str, shap_val: float, predicted_class: str) -> str:
    direction = "supports " + predicted_class if shap_val > 0 else "opposes " + predicted_class
    return (
        f"- {feature_label(feature_name)} (`{feature_name}`): {direction} the {predicted_class} label. "
        f"{feature_meaning(feature_name)} "
        f"[SHAP contribution for this row: {shap_val:+.4f} — this is NOT the raw field value.]"
    )


def format_lime_line_for_llm(condition: str, weight: float, predicted_class: str) -> str:
    direction = "supports " + predicted_class if weight > 0 else "opposes " + predicted_class
    return f"- {condition}: {direction} the {predicted_class} label [LIME weight: {weight:+.4f}]"


def format_eli5_line_for_llm(feature_name: str, importance: float) -> str:
    return (
        f"- {feature_label(feature_name)} (`{feature_name}`): "
        f"{feature_meaning(feature_name)} "
        f"[model-wide importance score: {importance:.4f}]"
    )


def build_xai_insight_lines(
    feature_reasons: Sequence[Dict[str, Any]],
    predicted_class: str,
    model_name: str,
    top_n: int = 5,
) -> tuple:
    """One-line summary plus bullet list for the dashboard."""
    top = feature_reasons[:top_n]
    if not top:
        return (
            f"No clear drivers were found for this {model_name} **{predicted_class}** prediction.",
            [],
        )
    top_impacts = [abs(float(r.get("impact", 0))) for r in top]
    total_impact = sum(top_impacts) if top_impacts else 1.0
    summary = synthesize_xai_summary(top, predicted_class, model_name)
    bullets = []
    for i, reason in enumerate(top, 1):
        pct = (abs(float(reason.get("impact", 0))) / total_impact * 100) if total_impact > 0 else 0.0
        bullets.append(format_xai_reason_bullet(i, reason, predicted_class, pct))
    return summary, bullets


def format_xai_insight_block_for_llm(
    feature_reasons: Sequence[Dict[str, Any]],
    predicted_class: str,
    model_name: str,
    top_n: int = 5,
) -> str:
    """Dashboard-style summary block passed into the LLM prompt."""
    summary, bullets = build_xai_insight_lines(
        feature_reasons, predicted_class, model_name, top_n=top_n
    )
    if not bullets:
        return summary
    return summary + "\n\nKey drivers:\n" + "\n".join(bullets)


def _parse_lime_feature_name(condition: str) -> str:
    for sep in (" <= ", " > ", " < ", " = ", " >= "):
        if sep in condition:
            return condition.split(sep)[0].strip()
    return condition.strip()


def _match_feature_name(partial: str, feature_names: List[str]) -> Optional[str]:
    partial_l = partial.lower()
    for feat in feature_names:
        fl = feat.lower()
        if partial_l in fl or fl in partial_l:
            return feat
    return None


def _extract_shap_vector(
    shap_explainer: Any,
    x_row: np.ndarray,
    pred_idx: int,
    n_features: int,
) -> Optional[np.ndarray]:
    if shap_explainer is None:
        return None
    try:
        x_row = np.asarray(x_row, dtype=np.float64)
        try:
            import shap

            is_tree = isinstance(shap_explainer, shap.explainers._tree.TreeExplainer)
        except Exception:
            is_tree = True
        if is_tree:
            shap_values = shap_explainer.shap_values(x_row)
        else:
            sample = x_row[0] if x_row.ndim > 1 else x_row
            shap_values = shap_explainer.shap_values(sample)

        safe_idx = max(0, min(int(pred_idx), 1))
        if isinstance(shap_values, list):
            if len(shap_values) == 1:
                class_vals = shap_values[0]
            else:
                class_vals = shap_values[min(safe_idx, len(shap_values) - 1)]
            if isinstance(class_vals, np.ndarray):
                shap_vals = (
                    np.array(class_vals[0]).flatten()
                    if len(class_vals.shape) == 2
                    else np.array(class_vals).flatten()
                )
            else:
                shap_vals = np.array(class_vals).flatten()
        else:
            shap_values = np.asarray(shap_values)
            if len(shap_values.shape) == 3:
                c = min(safe_idx, shap_values.shape[2] - 1)
                shap_vals = shap_values[0, :, c]
            elif len(shap_values.shape) == 2 and shap_values.shape[0] == 1:
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values.flatten()

        shap_vals = np.asarray(shap_vals).flatten()
        if len(shap_vals) < n_features:
            shap_vals = np.pad(shap_vals, (0, n_features - len(shap_vals)))
        elif len(shap_vals) > n_features:
            shap_vals = shap_vals[:n_features]
        return shap_vals
    except Exception:
        return None


def collect_feature_reasons(
    x_row: np.ndarray,
    feature_names: List[str],
    pred_idx: int,
    shap_explainer: Any,
    lime_explainer: Any,
    eli5_importance: Any,
    predict_proba: Callable,
) -> List[Dict[str, Any]]:
    """Merge SHAP/LIME ElI5 features"""
    all_reasons: Dict[str, Dict[str, Any]] = {}
    x_row = np.asarray(x_row, dtype=np.float64)
    if x_row.ndim == 1:
        x_row = x_row.reshape(1, -1)

    shap_vals = _extract_shap_vector(shap_explainer, x_row, pred_idx, len(feature_names))
    if shap_vals is not None:
        for i, feat_name in enumerate(feature_names):
            shap_impact = float(shap_vals[i])
            if abs(shap_impact) <= 0.001:
                continue
            all_reasons[feat_name] = {
                "feature": feat_name,
                "value": float(x_row[0][i]),
                "shap_score": abs(shap_impact),
                "shap_impact": shap_impact,
                "methods": ["SHAP"],
            }

    if lime_explainer is not None:
        try:
            safe_label = max(0, min(int(pred_idx), 1))
            lime_exp = lime_explainer.explain_instance(
                x_row[0], predict_proba, num_features=20, top_labels=1
            )
            for condition, weight in lime_exp.as_list(label=safe_label):
                matched = _match_feature_name(_parse_lime_feature_name(condition), feature_names)
                if not matched:
                    continue
                lime_impact = float(weight)
                if abs(lime_impact) <= 0.001:
                    continue
                if matched not in all_reasons:
                    all_reasons[matched] = {
                        "feature": matched,
                        "value": float(x_row[0][feature_names.index(matched)]),
                        "lime_score": abs(lime_impact),
                        "lime_impact": lime_impact,
                        "methods": ["LIME"],
                    }
                else:
                    all_reasons[matched]["lime_score"] = abs(lime_impact)
                    all_reasons[matched]["lime_impact"] = lime_impact
                    if "LIME" not in all_reasons[matched]["methods"]:
                        all_reasons[matched]["methods"].append("LIME")
        except Exception:
            pass

    if eli5_importance is not None and hasattr(eli5_importance, "feature_importances_"):
        importances = eli5_importance.feature_importances_
        for i, feat_name in enumerate(feature_names):
            eli5_score = float(importances[i])
            if eli5_score > 0.001 and feat_name in all_reasons:
                all_reasons[feat_name]["eli5_score"] = eli5_score
                if "ELI5" not in all_reasons[feat_name]["methods"]:
                    all_reasons[feat_name]["methods"].append("ELI5")

    for reason in all_reasons.values():
        primary_score = reason.get("shap_score", 0) or reason.get("lime_score", 0)
        local_methods = [m for m in reason["methods"] if m in ("SHAP", "LIME")]
        reason["combined_score"] = primary_score * (1 + len(local_methods) * 0.1)
        reason["impact"] = reason.get("shap_impact") or reason.get("lime_impact", 0)

    return [
        r
        for r in sorted(all_reasons.values(), key=lambda x: x["combined_score"], reverse=True)
        if r.get("shap_score", 0) > 0 or r.get("lime_score", 0) > 0
    ]
