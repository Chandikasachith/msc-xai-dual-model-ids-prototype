"""
CICIDS-2017 dual-model dashboard: RF + XGB predictions, per-row XAI.
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
import sys
import io
import time
from pathlib import Path

warnings.filterwarnings('ignore')

BASE = Path('SecondModel')


def load_secondmodel_npy(stem: str, dtype=np.float64) -> np.ndarray:
    """Load SecondModel array; prefer *_deploy.npy slices for cloud hosting."""
    deploy_path = BASE / f'{stem}_deploy.npy'
    full_path = BASE / f'{stem}.npy'
    path = deploy_path if deploy_path.exists() else full_path
    if not path.exists():
        raise FileNotFoundError(
            f'Missing {full_path.name} (run prepare_streamlit_deploy.py for cloud-sized slices)'
        )
    return np.asarray(np.load(path), dtype=dtype)


from cicids2017_preprocess import normalize_upload_columns, preprocess_for_rf, preprocess_for_xgb
from xai_insight_language import (
    AI_DISCLAIMER_MARKDOWN,
    AI_DISCLAIMER_SHORT,
    build_xai_insight_lines,
    chart_feature_labels,
    chart_title,
    chart_xlabel_toward_prediction,
    collect_feature_reasons,
    filter_toward_prediction_drivers,
    format_xai_insight_block_for_llm,
)

st.set_page_config(
    page_title="Dual-Model Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        margin: 1rem 0;
    }
    .attack {
        background-color: #ffcccc;
        color: #cc0000;
        border: 2px solid #cc0000;
    }
    .normal {
        background-color: #ccffcc;
        color: #006600;
        border: 2px solid #006600;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

if 'xai_package' not in st.session_state:
    st.session_state.xai_package = None
if 'rf_explainers' not in st.session_state:
    st.session_state.rf_explainers = {}
if 'xgb_explainers' not in st.session_state:
    st.session_state.xgb_explainers = {}

@st.cache_resource
def load_models():
    """Load RF/XGB models and preprocessors from SecondModel/."""
    package = {'rf': {'model': None}, 'xgb': {'model': None}}
    try:
        package['rf']['model'] = joblib.load(BASE / 'rf_model_binary_cicids2017.pkl')
        package['rf']['scaler'] = joblib.load(BASE / 'scaler_rf_cicids2017.pkl')
        package['rf']['selector'] = joblib.load(BASE / 'feature_selector_rf_cicids2017.pkl')
        package['rf']['label_encoders'] = joblib.load(BASE / 'label_encoders_rf_cicids2017.pkl')
        package['rf']['target_encoder'] = joblib.load(BASE / 'target_encoder_rf_cicids2017.pkl')
        package['rf']['feature_names'] = joblib.load(BASE / 'feature_names_rf_cicids2017.pkl')
        X_rf = load_secondmodel_npy('X_test_rf_cicids2017')
        y_rf = load_secondmodel_npy('y_test_rf_cicids2017', dtype=np.int64)
        package['rf']['test_data_sample'] = np.asarray(X_rf[:300], dtype=np.float64)
        package['rf']['test_labels'] = np.asarray(y_rf[:300], dtype=np.int64)
    except Exception as e:
        st.error(f"Error loading RF (CICIDS-2017): {e}")
        return None
    try:
        package['xgb']['model'] = joblib.load(BASE / 'xgb_model_binary_cicids2017_95percent.pkl')
        package['xgb']['scaler'] = joblib.load(BASE / 'scaler_xgb_cicids2017_95percent.pkl')
        package['xgb']['selector'] = joblib.load(BASE / 'feature_selector_xgb_cicids2017_95percent.pkl')
        package['xgb']['label_encoders'] = joblib.load(BASE / 'label_encoders_xgb_cicids2017_95percent.pkl')
        package['xgb']['target_encoder'] = joblib.load(BASE / 'target_encoder_xgb_cicids2017_95percent.pkl')
        package['xgb']['feature_names'] = joblib.load(BASE / 'feature_names_xgb_cicids2017_95percent.pkl')
        X_xgb = load_secondmodel_npy('X_test_xgb_cicids2017_95percent')
        y_xgb = load_secondmodel_npy('y_test_xgb_cicids2017_95percent', dtype=np.int64)
        package['xgb']['test_data_sample'] = np.asarray(X_xgb[:300], dtype=np.float64)
        package['xgb']['test_labels'] = np.asarray(y_xgb[:300], dtype=np.int64)
    except Exception as e:
        st.error(f"Error loading XGBoost (CICIDS-2017): {e}")
        return None
    return package

def initialize_xai_explainers(package):
    """Set up SHAP, LIME, and permutation importance for each model."""
    import shap
    import lime.lime_tabular
    from eli5.sklearn import PermutationImportance

    explainers = {'rf': {}, 'xgb': {}}

    if package['rf']['model'] is not None:
        try:
            explainers['rf']['shap'] = shap.TreeExplainer(package['rf']['model'])

            rf_test_data = package['rf']['test_data_sample']  # LIME background sample
            explainers['rf']['lime'] = lime.lime_tabular.LimeTabularExplainer(
                rf_test_data,
                feature_names=package['rf']['feature_names'],
                class_names=package['rf']['target_encoder'].classes_.tolist(),
                mode='classification',
                random_state=42
            )
            
            explainers['rf']['eli5'] = PermutationImportance(
                package['rf']['model'],
                random_state=42,
                n_iter=5,
                scoring='accuracy'
            )
            sample_size = min(200, len(rf_test_data))
            real_labels = package['rf'].get('test_labels', package['rf']['model'].predict(rf_test_data[:sample_size]))
            real_labels = real_labels[:sample_size] if len(real_labels) >= sample_size else real_labels
            explainers['rf']['eli5'].fit(rf_test_data[:sample_size], real_labels)
        except Exception as e:
            st.warning(f"RF XAI initialization warning: {e}")
    
    if package['xgb']['model'] is not None:
        try:
            try:
                explainers['xgb']['shap'] = shap.TreeExplainer(package['xgb']['model'])
            except:
                explainers['xgb']['shap'] = shap.KernelExplainer(
                    package['xgb']['model'].predict_proba,
                    package['xgb']['test_data_sample'][:50]
                )
            
            xgb_test_data = package['xgb']['test_data_sample']  # LIME background sample
            explainers['xgb']['lime'] = lime.lime_tabular.LimeTabularExplainer(
                xgb_test_data,
                feature_names=package['xgb']['feature_names'],
                class_names=package['xgb']['target_encoder'].classes_.tolist(),
                mode='classification',
                random_state=42
            )
            
            explainers['xgb']['eli5'] = PermutationImportance(
                package['xgb']['model'],
                random_state=42,
                n_iter=5,
                scoring='accuracy'
            )
            sample_size = min(200, len(xgb_test_data))
            real_labels = package['xgb'].get('test_labels', package['xgb']['model'].predict(xgb_test_data[:sample_size]))
            real_labels = real_labels[:sample_size] if len(real_labels) >= sample_size else real_labels
            explainers['xgb']['eli5'].fit(xgb_test_data[:sample_size], real_labels)
        except Exception as e:
            st.warning(f"XGBoost XAI initialization warning: {e}")
    
    return explainers

def get_prediction_rf(data, package):
    """RF predict + confidence for one row."""
    pred = package['rf']['model'].predict(data)[0]
    pred_proba = package['rf']['model'].predict_proba(data)[0]
    n_classes = len(package['rf']['target_encoder'].classes_)
    pred = min(pred, n_classes - 1)
    pred_label = package['rf']['target_encoder'].classes_[pred]
    confidence = pred_proba[pred] * 100
    return pred_label, confidence, pred_proba, pred

def get_prediction_xgb(data, package):
    """XGB predict + confidence for one row."""
    pred = package['xgb']['model'].predict(data)[0]
    pred_proba = package['xgb']['model'].predict_proba(data)[0]
    n_classes = len(package['xgb']['target_encoder'].classes_)
    pred = min(pred, n_classes - 1)
    pred_label = package['xgb']['target_encoder'].classes_[pred]
    confidence = pred_proba[pred] * 100
    return pred_label, confidence, pred_proba, pred

st.markdown('<div class="main-header">🛡️ Dual-Model Intrusion Detection System</div>', unsafe_allow_html=True)
st.markdown("**Random Forest (CICIDS-2017) + XGBoost (CICIDS-2017) with XAI Explanations**")
st.markdown("---")

# --- sidebar ---
with st.sidebar:
    st.header("📊 Control Panel")

    if st.button("🔄 Load Models", type="primary"):
        with st.spinner("Loading models and initializing XAI..."):
            package = load_models()
            if package:
                st.session_state.xai_package = package
                explainers = initialize_xai_explainers(package)
                st.session_state.rf_explainers = explainers.get('rf', {})
                st.session_state.xgb_explainers = explainers.get('xgb', {})
                st.success("✅ Models loaded successfully!")
                st.balloons()
    
    st.markdown("---")
    
    if st.session_state.xai_package is None:
        st.warning("⚠️ Please load models first!")
        st.info("Click 'Load Models' button above")
        st.stop()
    
    st.header("📥 Input Method")
    input_method = st.radio(
        "Choose input method:",
        ["Use Sample Data", "Upload File"],
        index=0
    )
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.info("""
    **Dual-Model System:**
    - Random Forest (CICIDS-2017)
    - XGBoost (CICIDS-2017)
    
    **XAI Methods:**
    - SHAP: Feature contributions
    - LIME: Local explanations
    - ELI5: Global importance
    """)

# --- main ---
package = st.session_state.xai_package

# --- input: built-in test rows or CSV upload ---
if input_method == "Use Sample Data":
    st.subheader("📋 Sample Data Selection")
    
    col1, col2 = st.columns(2)
    with col1:
        try:
            rf_samples = load_secondmodel_npy('X_test_rf_cicids2017')
            sample_options = list(range(1, min(len(rf_samples), 100) + 1))
            rf_sample_display = st.selectbox(
                "Select Random Forest Sample:",
                sample_options,
                key='rf_sample'
            )
            rf_sample_idx = rf_sample_display - 1
            rf_input = rf_samples[rf_sample_idx:rf_sample_idx+1]
            st.info(f"Sample {rf_sample_display} selected")
        except Exception as e:
            st.error(f"Error loading RF samples: {e}")
            st.stop()
    
    with col2:
        try:
            xgb_samples = load_secondmodel_npy('X_test_xgb_cicids2017_95percent')
            sample_options = list(range(1, min(len(xgb_samples), 100) + 1))
            xgb_sample_display = st.selectbox(
                "Select XGBoost Sample:",
                sample_options,
                key='xgb_sample'
            )
            xgb_sample_idx = xgb_sample_display - 1
            xgb_input = xgb_samples[xgb_sample_idx:xgb_sample_idx+1]
            st.info(f"Sample {xgb_sample_display} selected")
        except Exception as e:
            st.error(f"Error loading XGBoost samples: {e}")
            st.stop()

elif input_method == "Upload File":
    st.subheader("📤 File Upload")
    uploaded_file = st.file_uploader("Upload CSV (with header). Column names are auto-mapped.", type=['csv'])
    if not uploaded_file:
        st.info("Upload a CSV with header. Column names are auto-mapped. Optional: **Label** column.")
        st.stop()
    t_upload = time.perf_counter()
    try:
        raw = pd.read_csv(uploaded_file, sep=None, engine='python')
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
    if raw.empty:
        st.warning("CSV is empty.")
        st.stop()
    raw = normalize_upload_columns(raw)
    df_for_preprocess = raw.drop(columns=["Label"], errors="ignore")
    try:
        X_rf_upload = preprocess_for_rf(
            df_for_preprocess, package['rf']['scaler'], package['rf']['selector'], package['rf']['label_encoders']
        )
        X_xgb_upload = preprocess_for_xgb(
            df_for_preprocess, package['xgb']['scaler'], package['xgb']['selector'], package['xgb']['label_encoders']
        )
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()
    n_rows = len(X_rf_upload)
    # batch predict whole upload before single-row drill-down
    X_rf_arr = np.asarray(X_rf_upload, dtype=np.float64)
    X_xgb_arr = np.asarray(X_xgb_upload, dtype=np.float64)
    rf_pred_all = package['rf']['model'].predict(X_rf_arr)
    xgb_pred_all = package['xgb']['model'].predict(X_xgb_arr)
    upload_elapsed_s = time.perf_counter() - t_upload
    rf_classes = np.array(package['rf']['target_encoder'].classes_)
    xgb_classes = np.array(package['xgb']['target_encoder'].classes_)
    rf_labels = rf_classes[rf_pred_all]
    xgb_labels = xgb_classes[xgb_pred_all]
    rf_attack = int((rf_labels == 'Attack').sum())
    rf_normal = int((rf_labels == 'Normal').sum())
    xgb_attack = int((xgb_labels == 'Attack').sum())
    xgb_normal = int((xgb_labels == 'Normal').sum())
    st.markdown("### 📋 Summary for decision-making")
    sum_col1, sum_col2, sum_col3 = st.columns(3)
    with sum_col1:
        st.metric("Total rows", n_rows)
    with sum_col2:
        st.markdown("**Random Forest**")
        st.caption(f"Attack: **{rf_attack}** · Normal: **{rf_normal}**")
    with sum_col3:
        st.markdown("**XGBoost**")
        st.caption(f"Attack: **{xgb_attack}** · Normal: **{xgb_normal}**")
    if 'Label' in raw.columns:
        true_labels = raw['Label'].astype(str).str.strip()
        true_attack = int((true_labels.str.lower() == 'attack').sum())
        true_normal = int((true_labels.str.lower() == 'normal').sum())
        st.caption(f"Ground truth (if present): Attack {true_attack}, Normal {true_normal}")
    attack_idx_rf = np.where(rf_labels == 'Attack')[0]
    attack_idx_xgb = np.where(xgb_labels == 'Attack')[0]
    attack_idx_union = np.unique(np.concatenate([attack_idx_rf, attack_idx_xgb]))
    both_agree_attack = np.intersect1d(attack_idx_rf, attack_idx_xgb)
    st.markdown("**Suggested actions**")
    if len(attack_idx_union) == 0:
        st.success("No flows were predicted as Attack. Traffic appears normal; continue monitoring.")
    else:
        n_both = len(both_agree_attack)
        n_either = len(attack_idx_union)
        st.warning(f"**{n_either}** flow(s) predicted as Attack (both models agree on **{n_both}** — higher confidence).")
        st.markdown("**Recommendation:** Quarantine or deep-inspect these flows; export for SIEM/log review. If the same endpoints recur, consider blocking at perimeter or isolating the host.")
        ip_col = None
        for c in ['Source IP', 'Source IP ', 'SourceIP']:
            if c in raw.columns:
                ip_col = c
                break
        if ip_col:
            ips = raw.iloc[attack_idx_union][ip_col].dropna().astype(str).unique().tolist()
            ips = [x.strip() for x in ips if x.strip() and x.strip().lower() != 'nan']
            if ips:
                st.markdown("**Consider blocking or investigating these source IPs:**")
                st.code(', '.join(ips[:50]) + (' ...' if len(ips) > 50 else ''))
        st.caption(f"Row indices for export/review: {', '.join(map(str, attack_idx_union[:30].tolist()))}{' ...' if len(attack_idx_union) > 30 else ''}")
    from teams_executive_summary import (
        resolve_teams_webhook_url,
        send_executive_summary,
        teams_upload_fingerprint,
    )

    class_names_rf = list(package["rf"]["target_encoder"].classes_)
    class_names_xgb = list(package["xgb"]["target_encoder"].classes_)
    if resolve_teams_webhook_url():
        _fp = teams_upload_fingerprint(uploaded_file.name, rf_pred_all, xgb_pred_all)
        _attempted = st.session_state.get("teams_auto_attempted_fp_main")
        if _attempted != _fp:
            st.session_state.teams_auto_attempted_fp_main = _fp
            _ok, _msg = send_executive_summary(
                uploaded_file.name,
                class_names_rf,
                class_names_xgb,
                rf_pred_all,
                xgb_pred_all,
                float(upload_elapsed_s),
            )
            if _ok:
                try:
                    st.toast("Executive summary sent to Microsoft Teams.", icon="✅")
                except Exception:
                    st.success("Executive summary sent to Microsoft Teams.")
            else:
                st.warning(f"Could not send Teams summary: {_msg}")
        if st.button("Resend summary to Teams", key="teams_resend_main_dashboard"):
            ok2, msg2 = send_executive_summary(
                uploaded_file.name,
                class_names_rf,
                class_names_xgb,
                rf_pred_all,
                xgb_pred_all,
                float(upload_elapsed_s),
            )
            if ok2:
                st.success("Sent to Teams.")
            else:
                st.error(msg2)
    else:
        st.warning(
            "**Teams summary was not sent** — no webhook URL configured. "
            "Use `.streamlit/secrets.toml` (see `secrets.toml.example`) or set `TEAMS_WEBHOOK_URL` before `streamlit run`. "
           
        )
    st.markdown("---")
    st.success(f"Processed {n_rows} row(s). Select a row below to view prediction and XAI.")
    row_options = list(range(1, n_rows + 1))
    upload_row = st.selectbox("Row to predict and explain:", row_options, key="upload_row") - 1
    rf_input = np.asarray(X_rf_upload[upload_row:upload_row+1], dtype=np.float64)
    xgb_input = np.asarray(X_xgb_upload[upload_row:upload_row+1], dtype=np.float64)

# --- single-row prediction + explanations ---
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌲 Random Forest Model ")
    rf_pred, rf_conf, rf_proba, rf_pred_idx = get_prediction_rf(rf_input, package)

    pred_class = "attack" if rf_pred == "Attack" else "normal"
    st.markdown(
        f'<div class="prediction-box {pred_class}">{rf_pred}<br>Confidence: {rf_conf:.2f}%</div>',
        unsafe_allow_html=True
    )
    
    prob_col1, prob_col2 = st.columns(2)
    with prob_col1:
        st.metric("Attack Probability", f"{rf_proba[0]*100:.2f}%")
    with prob_col2:
        st.metric("Normal Probability", f"{rf_proba[1]*100:.2f}%")
    
    # per-row drivers: SHAP + LIME + ELI5 
    st.markdown("---")
    st.markdown("#### 💡 Why this prediction?")
    st.caption(
        "Key drivers for this prediction only (factors that support the predicted class). "
        "Opposing signals appear in Detailed XAI Explanations below."
    )
    
    try:
        pred_label = "Attack" if rf_pred == "Attack" else "Normal"
        st.markdown(f"**Prediction: {pred_label}** ({rf_conf:.2f}% confidence)")
        st.markdown("")
        
        all_reasons = {}
        
        shap_explainer = st.session_state.rf_explainers.get('shap')
        if shap_explainer:
            shap_values = shap_explainer.shap_values(rf_input)
            if isinstance(shap_values, list):
                safe_idx = min(rf_pred_idx, len(shap_values) - 1)
                if safe_idx < len(shap_values) and len(shap_values[safe_idx]) > 0:
                    shap_vals = np.array(shap_values[safe_idx][0]).flatten()
                else:
                    shap_vals = np.array(shap_values[0][0]).flatten()
            else:
                if len(shap_values.shape) == 3:
                    n_classes = shap_values.shape[2]
                    safe_idx = min(rf_pred_idx, n_classes - 1)
                    shap_vals = shap_values[0, :, safe_idx]
                else:
                    shap_vals = shap_values[0]
            
            for i, feat_name in enumerate(package['rf']['feature_names']):
                shap_impact = float(shap_vals[i])
                if abs(shap_impact) > 0.001:
                    if feat_name not in all_reasons:
                        all_reasons[feat_name] = {
                            'feature': feat_name,
                            'value': float(rf_input[0][i]),
                            'shap_score': abs(shap_impact),
                            'shap_impact': shap_impact,
                            'methods': ['SHAP']
                        }
                    else:
                        all_reasons[feat_name]['shap_score'] = abs(shap_impact)
                        all_reasons[feat_name]['shap_impact'] = shap_impact
                        all_reasons[feat_name]['methods'].append('SHAP')
        
        lime_explainer = st.session_state.rf_explainers.get('lime')
        if lime_explainer:
            try:
                safe_rf_pred_idx = max(0, min(rf_pred_idx, 1))
                lime_exp = lime_explainer.explain_instance(
                    rf_input[0],
                    package['rf']['model'].predict_proba,
                    num_features=20,
                    top_labels=1
                )
                lime_list = lime_exp.as_list(label=safe_rf_pred_idx)
                for condition, weight in lime_list:
                    feat_name = condition.split(' <= ')[0].split(' > ')[0].split(' < ')[0].split(' = ')[0].split(' >= ')[0].strip()
                    matched_feat = None
                    for feat in package['rf']['feature_names']:
                        if feat_name.lower() in feat.lower() or feat.lower() in feat_name.lower():
                            matched_feat = feat
                            break
                    if matched_feat:
                        lime_impact = float(weight)
                        if abs(lime_impact) > 0.001:
                            if matched_feat not in all_reasons:
                                all_reasons[matched_feat] = {
                                    'feature': matched_feat,
                                    'value': float(rf_input[0][package['rf']['feature_names'].index(matched_feat)]),
                                    'lime_score': abs(lime_impact),
                                    'lime_impact': lime_impact,
                                    'methods': ['LIME']
                                }
                            else:
                                all_reasons[matched_feat]['lime_score'] = abs(lime_impact)
                                all_reasons[matched_feat]['lime_impact'] = lime_impact
                                if 'LIME' not in all_reasons[matched_feat]['methods']:
                                    all_reasons[matched_feat]['methods'].append('LIME')
            except Exception as e:
                st.warning(f"LIME explanation error (RF): {str(e)}")
        
        eli5_importance = st.session_state.rf_explainers.get('eli5')
        if eli5_importance:
            importances = eli5_importance.feature_importances_
            for i, feat_name in enumerate(package['rf']['feature_names']):
                eli5_score = float(importances[i])
                if eli5_score > 0.001 and feat_name in all_reasons:
                    all_reasons[feat_name]['eli5_score'] = eli5_score
                    if 'ELI5' not in all_reasons[feat_name]['methods']:
                        all_reasons[feat_name]['methods'].append('ELI5')
        
        for feat_name, reason in all_reasons.items():
            primary_score = reason.get('shap_score', 0) or reason.get('lime_score', 0)
            local_methods = [m for m in reason['methods'] if m in ['SHAP', 'LIME']]
            method_bonus = len(local_methods) * 0.1
            reason['combined_score'] = primary_score * (1 + method_bonus)
            reason['impact'] = reason.get('shap_impact') or reason.get('lime_impact', 0)

        feature_reasons = [
            r for r in sorted(all_reasons.values(), key=lambda x: x['combined_score'], reverse=True)
            if r.get('shap_score', 0) > 0 or r.get('lime_score', 0) > 0
        ]

        if feature_reasons:
            supporting_reasons = filter_toward_prediction_drivers(feature_reasons, top_n=5)
            if supporting_reasons:
                summary, bullets = build_xai_insight_lines(
                    supporting_reasons, pred_label, "Random Forest"
                )
                st.markdown(summary)
                st.markdown("")
                st.markdown("**Key drivers (toward " + pred_label + ")**")
                st.markdown("\n\n".join(bullets))
                fig, ax = plt.subplots(figsize=(10, 4))
                features = chart_feature_labels(supporting_reasons)
                impacts = [r['impact'] for r in supporting_reasons]
                ax.barh(range(len(features)), impacts, color='green', alpha=0.7)
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features)
                ax.set_xlabel(chart_xlabel_toward_prediction())
                ax.set_title(chart_title(pred_label, "Random Forest"))
                ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info(
                    "No strong toward-prediction drivers found for this row. "
                    "See Detailed XAI Explanations below for full SHAP/LIME/ELI5 (+ and −)."
                )
            if any(float(r.get("impact", 0)) < 0 for r in feature_reasons):
                st.caption(
                    "Some features pulled toward the other class — see red bars in "
                    "**Detailed XAI Explanations** below."
                )
        else:
            st.info("XAI explainers not available. Loading detailed explanations below...")
    except Exception as e:
        st.warning(f"Could not generate reason summary: {e}")
        st.error(str(e))

    st.markdown("---")
    st.markdown("#### 🔍 Detailed XAI Explanations")
    with st.expander("📊 SHAP Explanation (Feature Contributions)", expanded=True):
        try:
            shap_explainer = st.session_state.rf_explainers.get('shap')
            if shap_explainer:
                shap_values = shap_explainer.shap_values(rf_input)
                safe_rf_pred_idx = max(0, min(rf_pred_idx, 1))
                if isinstance(shap_values, list):
                    safe_idx = min(safe_rf_pred_idx, len(shap_values) - 1)
                    safe_idx = max(0, safe_idx)
                    if safe_idx < len(shap_values) and len(shap_values[safe_idx]) > 0:
                        class_vals = shap_values[safe_idx]
                        if isinstance(class_vals, np.ndarray):
                            shap_vals = np.array(class_vals[0]).flatten() if len(class_vals.shape) == 2 else np.array(class_vals).flatten()
                        else:
                            class_vals = np.array(class_vals)
                            shap_vals = class_vals[0].flatten() if len(class_vals.shape) > 1 else class_vals.flatten()
                    else:
                        if len(shap_values) > 0 and len(shap_values[0]) > 0:
                            class_vals = shap_values[0]
                            if isinstance(class_vals, np.ndarray):
                                shap_vals = np.array(class_vals[0]).flatten() if len(class_vals.shape) == 2 else np.array(class_vals).flatten()
                            else:
                                class_vals = np.array(class_vals)
                                shap_vals = class_vals[0].flatten() if len(class_vals.shape) > 1 else class_vals.flatten()
                        else:
                            shap_vals = np.zeros(len(package['rf']['feature_names']))
                else:
                    if len(shap_values.shape) == 3:
                        n_classes = shap_values.shape[2]
                        safe_idx = min(safe_rf_pred_idx, n_classes - 1)
                        safe_idx = max(0, safe_idx)
                        shap_vals = shap_values[0, :, safe_idx]
                    else:
                        shap_vals = shap_values[0]
                n_features = len(package['rf']['feature_names'])
                if len(shap_vals) != n_features:
                    if len(shap_vals) < n_features:
                        shap_vals = np.pad(shap_vals, (0, n_features - len(shap_vals)), mode='constant')
                    else:
                        shap_vals = shap_vals[:n_features]
                feature_impacts = sorted(
                    [(package['rf']['feature_names'][i], float(shap_vals[i]))
                     for i in range(len(package['rf']['feature_names']))],
                    key=lambda x: abs(x[1]), reverse=True
                )[:15]
                df_shap = pd.DataFrame(feature_impacts, columns=['Feature', 'SHAP Value'])
                df_shap['Impact'] = df_shap['SHAP Value'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['red' if x < 0 else 'green' for x in df_shap['SHAP Value']]
                ax.barh(range(len(df_shap)), df_shap['SHAP Value'], color=colors)
                ax.set_yticks(range(len(df_shap)))
                ax.set_yticklabels(df_shap['Feature'])
                ax.set_xlabel('SHAP Value (Impact on Prediction)')
                ax.set_title('Top 15 Features - SHAP Values')
                ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig)
                st.dataframe(df_shap, use_container_width=True)
            else:
                st.warning("SHAP explainer not initialized")
        except Exception as e:
            st.error(f"SHAP error: {e}")

    with st.expander("🍋 LIME Explanation (Local Interpretable Model)"):
        try:
            lime_explainer = st.session_state.rf_explainers.get('lime')
            if lime_explainer:
                lime_exp = lime_explainer.explain_instance(
                    rf_input[0],
                    package['rf']['model'].predict_proba,
                    num_features=15,
                    top_labels=2
                )
                safe_rf_pred_idx = max(0, min(rf_pred_idx, 1))
                lime_list = lime_exp.as_list(label=safe_rf_pred_idx)
                df_lime = pd.DataFrame(lime_list, columns=['Feature Condition', 'Weight'])
                df_lime['Impact'] = df_lime['Weight'].apply(lambda x: 'Supports' if x > 0 else 'Opposes')
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['red' if x < 0 else 'green' for x in df_lime['Weight']]
                ax.barh(range(len(df_lime)), df_lime['Weight'], color=colors)
                ax.set_yticks(range(len(df_lime)))
                ax.set_yticklabels(df_lime['Feature Condition'], fontsize=8)
                ax.set_xlabel('LIME Weight')
                ax.set_title('Top 15 Features - LIME Weights')
                ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig)
                st.dataframe(df_lime, use_container_width=True)
            else:
                st.warning("LIME explainer not initialized")
        except Exception as e:
            st.error(f"LIME error: {e}")

    with st.expander("📈 ELI5 Global Feature Importance"):
        try:
            eli5_importance = st.session_state.rf_explainers.get('eli5')
            if eli5_importance:
                importances = eli5_importance.feature_importances_
                top_features = sorted(
                    [(package['rf']['feature_names'][i], float(importances[i]))
                     for i in range(len(package['rf']['feature_names']))],
                    key=lambda x: abs(x[1]), reverse=True
                )[:15]
                df_eli5 = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(df_eli5)), df_eli5['Importance'], color='steelblue')
                ax.set_yticks(range(len(df_eli5)))
                ax.set_yticklabels(df_eli5['Feature'])
                ax.set_xlabel('Permutation Importance')
                ax.set_title('Top 15 Features - ELI5 Global Importance')
                plt.tight_layout()
                st.pyplot(fig)
                st.dataframe(df_eli5, use_container_width=True)
            else:
                st.warning("ELI5 importance not computed")
        except Exception as e:
            st.error(f"ELI5 error: {e}")

with col2:
        st.markdown("### ⚡ XGBoost Model ")
        xgb_pred, xgb_conf, xgb_proba, xgb_pred_idx = get_prediction_xgb(xgb_input, package)

        pred_class = "attack" if xgb_pred == "Attack" else "normal"
        st.markdown(
            f'<div class="prediction-box {pred_class}">{xgb_pred}<br>Confidence: {xgb_conf:.2f}%</div>', 
            unsafe_allow_html=True
        )
        
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric("Attack Probability", f"{xgb_proba[0]*100:.2f}%")
        with prob_col2:
            st.metric("Normal Probability", f"{xgb_proba[1]*100:.2f}%")
        
        # per-row drivers: SHAP + LIME + ELI5 
        st.markdown("---")
        st.markdown("#### 💡 Why this prediction?")
        st.caption(
            "Key drivers for this prediction only (factors that support the predicted class). "
            "Opposing signals appear in Detailed XAI Explanations below."
        )
        
        try:
            pred_label = "Attack" if xgb_pred == "Attack" else "Normal"
            st.markdown(f"**Prediction: {pred_label}** ({xgb_conf:.2f}% confidence)")
            st.markdown("")
            
            all_reasons = {}
            
            shap_explainer = st.session_state.xgb_explainers.get('shap')
            if shap_explainer:
                try:
                    if type(shap_explainer).__name__ == 'TreeExplainer':
                        shap_values = shap_explainer.shap_values(xgb_input)
                    else:
                        shap_values = shap_explainer.shap_values(xgb_input[0])
                    
                    safe_pred_idx = max(0, min(xgb_pred_idx, 1))
                    # normalize SHAP output (list vs ndarray; XGB can return either)
                    if isinstance(shap_values, list):
                        if len(shap_values) == 1:
                            class_vals = shap_values[0]
                        else:
                            safe_idx = min(safe_pred_idx, len(shap_values) - 1)
                            safe_idx = max(0, safe_idx)
                            class_vals = shap_values[safe_idx]
                        if isinstance(class_vals, np.ndarray):
                            if len(class_vals.shape) == 2:
                                shap_vals = np.array(class_vals[0]).flatten()
                            elif len(class_vals.shape) == 1:
                                shap_vals = np.array(class_vals).flatten()
                            else:
                                class_vals_flat = class_vals.flatten()
                                if len(class_vals_flat) > 0:
                                    shap_vals = class_vals_flat
                                else:
                                    shap_vals = np.array(class_vals).flatten()
                        else:
                            class_vals = np.array(class_vals)
                            if len(class_vals.shape) > 1:
                                shap_vals = class_vals[0].flatten()
                            else:
                                shap_vals = class_vals.flatten()
                    else:
                        shap_values = np.array(shap_values)
                        if len(shap_values.shape) == 3:
                            n_classes = shap_values.shape[2]
                            safe_idx = min(safe_pred_idx, n_classes - 1)
                            safe_idx = max(0, safe_idx)
                            shap_vals = shap_values[0, :, safe_idx]
                        elif len(shap_values.shape) == 2:
                            if shap_values.shape[0] == 1:
                                shap_vals = shap_values[0]
                            else:
                                shap_vals = shap_values.flatten()
                        else:
                            shap_vals = np.array(shap_values).flatten()

                    n_features = len(package['xgb']['feature_names'])
                    if len(shap_vals) != n_features:
                        if len(shap_vals) < n_features:
                            shap_vals = np.pad(shap_vals, (0, n_features - len(shap_vals)), mode='constant')
                        else:
                            shap_vals = shap_vals[:n_features]

                    if np.allclose(shap_vals, 0):
                        if isinstance(shap_values, list) and len(shap_values) > 1:
                            other_idx = 1 - safe_pred_idx
                            if other_idx < len(shap_values):
                                class_vals = shap_values[other_idx]
                                if isinstance(class_vals, np.ndarray):
                                    if len(class_vals.shape) == 2:
                                        shap_vals = np.array(class_vals[0]).flatten()
                                    else:
                                        shap_vals = np.array(class_vals).flatten()
                                    if len(shap_vals) != n_features:
                                        if len(shap_vals) < n_features:
                                            shap_vals = np.pad(shap_vals, (0, n_features - len(shap_vals)), mode='constant')
                                        else:
                                            shap_vals = shap_vals[:n_features]
                except Exception:
                    shap_vals = np.zeros(len(package['xgb']['feature_names']))
                
                for i, feat_name in enumerate(package['xgb']['feature_names']):
                    if i >= len(shap_vals):
                        break
                    shap_impact = float(shap_vals[i])
                    if abs(shap_impact) > 0.001:
                        if feat_name not in all_reasons:
                            all_reasons[feat_name] = {
                                'feature': feat_name,
                                'value': float(xgb_input[0][i]),
                                'shap_score': abs(shap_impact),
                                'shap_impact': shap_impact,
                                'methods': ['SHAP']
                            }
                        else:
                            all_reasons[feat_name]['shap_score'] = abs(shap_impact)
                            all_reasons[feat_name]['shap_impact'] = shap_impact
                            all_reasons[feat_name]['methods'].append('SHAP')
            
            lime_explainer = st.session_state.xgb_explainers.get('lime')
            if lime_explainer:
                try:
                    safe_xgb_pred_idx = max(0, min(xgb_pred_idx, 1))
                    
                    lime_exp = lime_explainer.explain_instance(
                        xgb_input[0],
                        package['xgb']['model'].predict_proba,
                        num_features=20,
                        top_labels=1
                    )
                    lime_list = lime_exp.as_list(label=safe_xgb_pred_idx)
                    
                    for condition, weight in lime_list:
                        feat_name = condition.split(' <= ')[0].split(' > ')[0].split(' < ')[0].split(' = ')[0].split(' >= ')[0].strip()
                        
                        matched_feat = None
                        for feat in package['xgb']['feature_names']:
                            if feat_name.lower() in feat.lower() or feat.lower() in feat_name.lower():
                                matched_feat = feat
                                break
                        
                        if matched_feat:
                            lime_impact = float(weight)
                            if abs(lime_impact) > 0.001:
                                if matched_feat not in all_reasons:
                                    all_reasons[matched_feat] = {
                                        'feature': matched_feat,
                                        'value': float(xgb_input[0][package['xgb']['feature_names'].index(matched_feat)]),
                                        'lime_score': abs(lime_impact),
                                        'lime_impact': lime_impact,
                                        'methods': ['LIME']
                                    }
                                else:
                                    all_reasons[matched_feat]['lime_score'] = abs(lime_impact)
                                    all_reasons[matched_feat]['lime_impact'] = lime_impact
                                    if 'LIME' not in all_reasons[matched_feat]['methods']:
                                        all_reasons[matched_feat]['methods'].append('LIME')
                except Exception as e:
                    st.warning(f"LIME explanation error (XGB): {str(e)}")
            
            eli5_importance = st.session_state.xgb_explainers.get('eli5')
            if eli5_importance:
                importances = eli5_importance.feature_importances_
                for i, feat_name in enumerate(package['xgb']['feature_names']):
                    eli5_score = float(importances[i])
                    if eli5_score > 0.001:
                        if feat_name in all_reasons:
                            all_reasons[feat_name]['eli5_score'] = eli5_score
                            if 'ELI5' not in all_reasons[feat_name]['methods']:
                                all_reasons[feat_name]['methods'].append('ELI5')
            
            for feat_name, reason in all_reasons.items():
                primary_score = reason.get('shap_score', 0) or reason.get('lime_score', 0)
                local_methods = [m for m in reason['methods'] if m in ['SHAP', 'LIME']]
                method_bonus = len(local_methods) * 0.1
                reason['combined_score'] = primary_score * (1 + method_bonus)
                reason['impact'] = reason.get('shap_impact') or reason.get('lime_impact', 0)
            
            feature_reasons = [
                r for r in sorted(all_reasons.values(), key=lambda x: x['combined_score'], reverse=True)
                if r.get('shap_score', 0) > 0 or r.get('lime_score', 0) > 0
            ]
            
            if feature_reasons:
                supporting_reasons = filter_toward_prediction_drivers(feature_reasons, top_n=5)
                if supporting_reasons:
                    summary, bullets = build_xai_insight_lines(
                        supporting_reasons, pred_label, "XGBoost"
                    )
                    st.markdown(summary)
                    st.markdown("")
                    st.markdown("**Key drivers (toward " + pred_label + ")**")
                    st.markdown("\n\n".join(bullets))
                    fig, ax = plt.subplots(figsize=(10, 4))
                    features = chart_feature_labels(supporting_reasons)
                    impacts = [r['impact'] for r in supporting_reasons]
                    ax.barh(range(len(features)), impacts, color='green', alpha=0.7)
                    ax.set_yticks(range(len(features)))
                    ax.set_yticklabels(features)
                    ax.set_xlabel(chart_xlabel_toward_prediction())
                    ax.set_title(chart_title(pred_label, "XGBoost"))
                    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
                    ax.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info(
                        "No strong toward-prediction drivers found for this row. "
                        "See Detailed XAI Explanations below for full SHAP/LIME (+ and −)."
                    )
                if any(float(r.get("impact", 0)) < 0 for r in feature_reasons):
                    st.caption(
                        "Some features pulled toward the other class — see red bars in "
                        "**Detailed XAI Explanations** below."
                    )
            else:
                st.info("XAI explainers not available. Loading detailed explanations below...")
        except Exception as e:
            st.warning(f"Could not generate reason summary: {e}")
            st.error(str(e))
        
        st.markdown("---")
        st.markdown("#### 🔍 Detailed XAI Explanations")
        
        with st.expander("📊 SHAP Explanation (Feature Contributions)", expanded=True):
            try:
                shap_explainer = st.session_state.xgb_explainers.get('shap')
                if shap_explainer:
                    try:
                        if type(shap_explainer).__name__ == 'TreeExplainer':
                            shap_values = shap_explainer.shap_values(xgb_input)
                        else:
                            shap_values = shap_explainer.shap_values(xgb_input[0])
                        
                        safe_pred_idx = max(0, min(xgb_pred_idx, 1))
                        if isinstance(shap_values, list):
                            if len(shap_values) == 1:
                                class_vals = shap_values[0]
                            else:
                                safe_idx = min(safe_pred_idx, len(shap_values) - 1)
                                safe_idx = max(0, safe_idx)
                                class_vals = shap_values[safe_idx]
                            if isinstance(class_vals, np.ndarray):
                                if len(class_vals.shape) == 2:
                                    shap_vals = np.array(class_vals[0]).flatten()
                                elif len(class_vals.shape) == 1:
                                    shap_vals = np.array(class_vals).flatten()
                                else:
                                    class_vals_flat = class_vals.flatten()
                                    shap_vals = class_vals_flat if len(class_vals_flat) > 0 else np.array(class_vals).flatten()
                            else:
                                class_vals = np.array(class_vals)
                                shap_vals = class_vals[0].flatten() if len(class_vals.shape) > 1 else class_vals.flatten()
                        else:
                            shap_values = np.array(shap_values)
                            if len(shap_values.shape) == 3:
                                n_classes = shap_values.shape[2]
                                safe_idx = min(safe_pred_idx, n_classes - 1)
                                safe_idx = max(0, safe_idx)
                                shap_vals = shap_values[0, :, safe_idx]
                            elif len(shap_values.shape) == 2:
                                shap_vals = shap_values[0] if shap_values.shape[0] == 1 else shap_values.flatten()
                            else:
                                shap_vals = np.array(shap_values).flatten()

                        n_features = len(package['xgb']['feature_names'])
                        if len(shap_vals) != n_features:
                            if len(shap_vals) < n_features:
                                shap_vals = np.pad(shap_vals, (0, n_features - len(shap_vals)), mode='constant')
                            else:
                                shap_vals = shap_vals[:n_features]

                        if np.allclose(shap_vals, 0):
                            if isinstance(shap_values, list) and len(shap_values) > 1:
                                other_idx = 1 - safe_pred_idx
                                if other_idx < len(shap_values):
                                    class_vals = shap_values[other_idx]
                                    if isinstance(class_vals, np.ndarray):
                                        shap_vals = np.array(class_vals[0]).flatten() if len(class_vals.shape) == 2 else np.array(class_vals).flatten()
                                        if len(shap_vals) != n_features:
                                            if len(shap_vals) < n_features:
                                                shap_vals = np.pad(shap_vals, (0, n_features - len(shap_vals)), mode='constant')
                                            else:
                                                shap_vals = shap_vals[:n_features]
                    except Exception as e:
                        shap_vals = np.zeros(len(package['xgb']['feature_names']))
                        st.warning(f"SHAP extraction error: {e}")
                    
                    feature_impacts = sorted(
                        [(package['xgb']['feature_names'][i], float(shap_vals[i])) 
                         for i in range(len(package['xgb']['feature_names']))],
                        key=lambda x: abs(x[1]), reverse=True
                    )[:15]
                    
                    df_shap = pd.DataFrame(feature_impacts, columns=['Feature', 'SHAP Value'])
                    df_shap['Impact'] = df_shap['SHAP Value'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['red' if x < 0 else 'green' for x in df_shap['SHAP Value']]
                    ax.barh(range(len(df_shap)), df_shap['SHAP Value'], color=colors)
                    ax.set_yticks(range(len(df_shap)))
                    ax.set_yticklabels(df_shap['Feature'])
                    ax.set_xlabel('SHAP Value (Impact on Prediction)')
                    ax.set_title('Top 15 Features - SHAP Values')
                    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.dataframe(df_shap, use_container_width=True)
                else:
                    st.warning("SHAP explainer not initialized")
            except Exception as e:
                st.error(f"SHAP error: {e}")
        
        with st.expander("🍋 LIME Explanation (Local Interpretable Model)"):
            try:
                lime_explainer = st.session_state.xgb_explainers.get('lime')
                if lime_explainer:
                    lime_exp = lime_explainer.explain_instance(
                        xgb_input[0],
                        package['xgb']['model'].predict_proba,
                        num_features=15,
                        top_labels=2
                    )
                    safe_xgb_pred_idx = max(0, min(xgb_pred_idx, 1))
                    lime_list = lime_exp.as_list(label=safe_xgb_pred_idx)
                    df_lime = pd.DataFrame(lime_list, columns=['Feature Condition', 'Weight'])
                    df_lime['Impact'] = df_lime['Weight'].apply(lambda x: 'Supports' if x > 0 else 'Opposes')
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['red' if x < 0 else 'green' for x in df_lime['Weight']]
                    ax.barh(range(len(df_lime)), df_lime['Weight'], color=colors)
                    ax.set_yticks(range(len(df_lime)))
                    ax.set_yticklabels(df_lime['Feature Condition'], fontsize=8)
                    ax.set_xlabel('LIME Weight')
                    ax.set_title('Top 15 Features - LIME Weights')
                    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.dataframe(df_lime, use_container_width=True)
                else:
                    st.warning("LIME explainer not initialized")
            except Exception as e:
                st.error(f"LIME error: {e}")
        
        with st.expander("📈 ELI5 Global Feature Importance"):
            try:
                eli5_importance = st.session_state.xgb_explainers.get('eli5')
                if eli5_importance:
                    importances = eli5_importance.feature_importances_
                    top_features = sorted(
                        [(package['xgb']['feature_names'][i], float(importances[i])) 
                         for i in range(len(package['xgb']['feature_names']))],
                        key=lambda x: abs(x[1]), reverse=True
                    )[:15]
                    df_eli5 = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(range(len(df_eli5)), df_eli5['Importance'], color='steelblue')
                    ax.set_yticks(range(len(df_eli5)))
                    ax.set_yticklabels(df_eli5['Feature'])
                    ax.set_xlabel('Permutation Importance')
                    ax.set_title('Top 15 Features - ELI5 Global Importance')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.dataframe(df_eli5, use_container_width=True)
                else:
                    st.warning("ELI5 importance not computed")
            except Exception as e:
                st.error(f"ELI5 error: {e}")

# --- model agreement ---
st.markdown("---")
st.subheader("📊 Model Comparison")
comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
with comp_col1:
    agreement = "✅ YES" if rf_pred == xgb_pred else "❌ NO"
    st.metric("Agreement", agreement)
with comp_col2:
    st.metric("RF Prediction", rf_pred)
with comp_col3:
    st.metric("XGBoost Prediction", xgb_pred)
with comp_col4:
    conf_diff = abs(rf_conf - xgb_conf)
    st.metric("Confidence Difference", f"{conf_diff:.2f}%")
if rf_pred != xgb_pred:
    st.warning("⚠️ **Models Disagree!** The two models have different predictions. Review XAI explanations to understand why.")
else:
    st.success("✅ **Models Agree!** Both models predict the same class.")

# --- OpenAI analyst notes ---
if input_method == "Upload File":
    from llm_openai_advisory import (
        build_user_prompt,
        call_openai_advisory,
        format_raw_csv_row,
        row_needs_llm_advisory,
        resolve_openai_api_key,
        _eli5_global_top_lines,
        _lime_top_lines,
        _shap_top_lines,
    )
    from teams_executive_summary import teams_upload_fingerprint

    st.markdown("---")
    st.subheader("🤖 AI analyst notes")
    st.warning(AI_DISCLAIMER_MARKDOWN)
    st.caption(
        "Uses this row’s **raw CSV values**, dashboard XAI summaries (RF + XGB), and SHAP/LIME/ELI5 detail. "
        + AI_DISCLAIMER_SHORT
    )
    if not row_needs_llm_advisory(rf_pred, xgb_pred):
        st.info(
            "OpenAI suggestions are offered when **at least one model predicts Attack** or **models disagree**. "
            "This row is **both Normal** — no LLM call."
        )
    elif not resolve_openai_api_key():
        st.warning(
            "Add **OPENAI_API_KEY** to `.streamlit/secrets.toml`. "
            "See `.streamlit/secrets.toml.example`."
        )
    else:
        _bfp = teams_upload_fingerprint(uploaded_file.name, rf_pred_all, xgb_pred_all)
        _llm_cache = f"llm_openai_v8_{_bfp}_{upload_row}"
        if st.button("Generate analyst notes", key="btn_llm_openai_main"):
            _raw_md = format_raw_csv_row(raw, upload_row)
            _rf_e = st.session_state.rf_explainers
            _xgb_e = st.session_state.xgb_explainers
            _rf_reasons = collect_feature_reasons(
                rf_input,
                list(package["rf"]["feature_names"]),
                rf_pred_idx,
                _rf_e.get("shap"),
                _rf_e.get("lime"),
                _rf_e.get("eli5"),
                package["rf"]["model"].predict_proba,
            )
            _xgb_reasons = collect_feature_reasons(
                xgb_input,
                list(package["xgb"]["feature_names"]),
                xgb_pred_idx,
                _xgb_e.get("shap"),
                _xgb_e.get("lime"),
                _xgb_e.get("eli5"),
                package["xgb"]["model"].predict_proba,
            )
            _rf_xai_block = format_xai_insight_block_for_llm(
                _rf_reasons, rf_pred, "Random Forest"
            )
            _xgb_xai_block = format_xai_insight_block_for_llm(
                _xgb_reasons, xgb_pred, "XGBoost"
            )
            _rf_sh = _shap_top_lines(
                _rf_e.get("shap"),
                rf_input,
                list(package["rf"]["feature_names"]),
                rf_pred_idx,
                rf_pred,
            )
            _rf_li = _lime_top_lines(
                _rf_e.get("lime"),
                rf_input,
                package["rf"]["model"].predict_proba,
                rf_pred_idx,
                rf_pred,
            )
            _rf_e5 = _eli5_global_top_lines(
                _rf_e.get("eli5"),
                list(package["rf"]["feature_names"]),
            )
            _xgb_sh = _shap_top_lines(
                _xgb_e.get("shap"),
                xgb_input,
                list(package["xgb"]["feature_names"]),
                xgb_pred_idx,
                xgb_pred,
            )
            _xgb_li = _lime_top_lines(
                _xgb_e.get("lime"),
                xgb_input,
                package["xgb"]["model"].predict_proba,
                xgb_pred_idx,
                xgb_pred,
            )
            _xgb_e5 = _eli5_global_top_lines(
                _xgb_e.get("eli5"),
                list(package["xgb"]["feature_names"]),
            )
            _user = build_user_prompt(
                uploaded_file.name,
                upload_row + 1,
                rf_pred,
                rf_conf,
                xgb_pred,
                xgb_conf,
                rf_pred == xgb_pred,
                _raw_md,
                _rf_xai_block,
                _xgb_xai_block,
                _rf_sh,
                _rf_li,
                _xgb_sh,
                _xgb_li,
                _rf_e5,
                _xgb_e5,
            )
            _out, _err = call_openai_advisory(_user)
            st.session_state[_llm_cache] = (_out, _err)
        if _llm_cache in st.session_state:
            _o, _e = st.session_state[_llm_cache]
            if _o:
                st.caption(AI_DISCLAIMER_SHORT)
                st.markdown(_o)
            else:
                st.error(_e)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <strong>Dual-Model Intrusion Detection System</strong><br>
    Random Forest (CICIDS-2017) + XGBoost (CICIDS-2017)<br>
    Powered by SHAP, LIME, and ELI5
</div>
""", unsafe_allow_html=True)

