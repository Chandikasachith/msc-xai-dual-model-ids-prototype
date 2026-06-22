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

CHART_TEAL = '#14B8A6'
CHART_RED = '#F87171'
CHART_BG = '#131F2E'
CHART_TEXT = '#94A3B8'
CHART_GRID = '#1E3044'


def style_chart(ax, fig):
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    ax.tick_params(colors=CHART_TEXT, labelsize=9)
    ax.xaxis.label.set_color(CHART_TEXT)
    ax.yaxis.label.set_color(CHART_TEXT)
    ax.title.set_color('#F9FAFB')
    ax.title.set_fontsize(11)
    ax.title.set_fontweight('600')
    for spine in ax.spines.values():
        spine.set_color(CHART_GRID)
    ax.grid(axis='x', color=CHART_GRID, alpha=0.45, linestyle='--')


def polish_chart(ax):
    """Re-apply dark-theme colors after set_title / set_xlabel calls."""
    ax.title.set_color('#F9FAFB')
    ax.xaxis.label.set_color(CHART_TEXT)
    ax.yaxis.label.set_color(CHART_TEXT)
    ax.tick_params(colors=CHART_TEXT, labelsize=9)


def create_chart(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    style_chart(ax, fig)
    return fig, ax


def bar_colors(values):
    return [CHART_RED if x < 0 else CHART_TEAL for x in values]


def render_kpi_row(cards):
    parts = ['<div class="kpi-row">']
    for card in cards:
        parts.append(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{card["label"]}</div>'
            f'<div class="kpi-value">{card["value"]}</div>'
            f'<div class="kpi-sub">{card.get("sub", "")}</div>'
            f'</div>'
        )
    parts.append('</div>')
    st.markdown(''.join(parts), unsafe_allow_html=True)


def render_prediction_panel(model_name, pred_label, confidence, proba, accent='rf'):
    attack_pct = float(proba[0]) * 100
    normal_pct = float(proba[1]) * 100
    badge = 'badge-attack' if pred_label == 'Attack' else 'badge-normal'
    st.markdown(
        f'<div class="pred-panel pred-{accent}">'
        f'<div class="pred-panel-header">{model_name}</div>'
        f'<div class="pred-verdict {badge}">{pred_label}</div>'
        f'<div class="pred-confidence">{confidence:.1f}% confidence</div>'
        f'<div class="prob-bar-row"><span class="prob-label">Attack</span>'
        f'<div class="prob-track"><div class="prob-fill attack-fill" style="width:{attack_pct:.1f}%"></div></div>'
        f'<span class="prob-pct">{attack_pct:.1f}%</span></div>'
        f'<div class="prob-bar-row"><span class="prob-label">Normal</span>'
        f'<div class="prob-track"><div class="prob-fill normal-fill" style="width:{normal_pct:.1f}%"></div></div>'
        f'<span class="prob-pct">{normal_pct:.1f}%</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_section_title(title, subtitle=''):
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ''
    st.markdown(
        f'<div class="section-title"><div class="section-bar"></div>'
        f'<div><div class="section-heading">{title}</div>{sub}</div></div>',
        unsafe_allow_html=True,
    )


def render_alert(message, level='info'):
    st.markdown(f'<div class="alert-banner alert-{level}">{message}</div>', unsafe_allow_html=True)


DASHBOARD_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
    .stApp { background-color: #0B1622; }
    [data-testid="stSidebar"] {
        background-color: #0F1B2A !important;
        border-right: 1px solid #1E3044;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #F9FAFB !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
    }
    [data-testid="stMetric"] {
        background: #131F2E;
        border: 1px solid #1E3044;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    [data-testid="stMetric"] label { color: #94A3B8 !important; font-size: 0.75rem !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #F9FAFB !important; }
    div[data-testid="stExpander"] {
        background: #131F2E;
        border: 1px solid #1E3044;
        border-radius: 10px;
    }
    .stButton > button[kind="primary"] {
        background: #14B8A6 !important;
        color: #0B1622 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        width: 100%;
    }
    .stButton > button[kind="primary"]:hover {
        background: #2DD4BF !important;
        color: #0B1622 !important;
    }
    .page-hero {
        background: linear-gradient(135deg, #0F1B2A 0%, #131F2E 100%);
        border: 1px solid #1E3044;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
    }
    .page-hero h1 {
        color: #F9FAFB;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
    }
    .page-hero p {
        color: #94A3B8;
        font-size: 0.95rem;
        margin: 0;
    }
    .status-chips { margin-top: 0.85rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
    .status-chip {
        background: #14B8A620;
        color: #2DD4BF;
        border: 1px solid #14B8A640;
        border-radius: 999px;
        padding: 0.2rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .status-chip.pending { background: #1E3044; color: #94A3B8; border-color: #374151; }
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.75rem;
        margin: 0 0 1.25rem 0;
    }
    @media (max-width: 1100px) { .kpi-row { grid-template-columns: repeat(3, 1fr); } }
    @media (max-width: 700px) { .kpi-row { grid-template-columns: repeat(2, 1fr); } }
    .kpi-card {
        background: #131F2E;
        border: 1px solid #1E3044;
        border-radius: 10px;
        padding: 1rem 1.1rem;
    }
    .kpi-label {
        color: #94A3B8;
        font-size: 0.72rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        color: #F9FAFB;
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .kpi-sub { color: #64748B; font-size: 0.72rem; margin-top: 0.25rem; }
    .pred-panel {
        background: #131F2E;
        border: 1px solid #1E3044;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .pred-panel.pred-rf { border-left: 3px solid #60A5FA; }
    .pred-panel.pred-xgb { border-left: 3px solid #A78BFA; }
    .pred-panel-header {
        color: #94A3B8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }
    .pred-verdict {
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .badge-attack { color: #F87171; }
    .badge-normal { color: #2DD4BF; }
    .pred-confidence { color: #64748B; font-size: 0.85rem; margin-bottom: 1rem; }
    .prob-bar-row {
        display: grid;
        grid-template-columns: 3.5rem 1fr 3rem;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.45rem;
    }
    .prob-label { color: #94A3B8; font-size: 0.78rem; }
    .prob-track {
        background: #0B1622;
        border-radius: 999px;
        height: 8px;
        overflow: hidden;
    }
    .prob-fill { height: 100%; border-radius: 999px; }
    .attack-fill { background: linear-gradient(90deg, #F87171, #EF4444); }
    .normal-fill { background: linear-gradient(90deg, #14B8A6, #2DD4BF); }
    .prob-pct { color: #F9FAFB; font-size: 0.78rem; text-align: right; }
    .section-title {
        display: flex;
        align-items: flex-start;
        gap: 0.65rem;
        margin: 1.25rem 0 0.85rem 0;
    }
    .section-bar {
        width: 3px;
        height: 2rem;
        background: #14B8A6;
        border-radius: 2px;
        flex-shrink: 0;
        margin-top: 0.1rem;
    }
    .section-heading {
        color: #F9FAFB;
        font-size: 1.05rem;
        font-weight: 600;
    }
    .section-sub { color: #64748B; font-size: 0.82rem; margin-top: 0.15rem; }
    .alert-banner {
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        margin: 0.75rem 0 1rem 0;
        font-size: 0.9rem;
        line-height: 1.45;
    }
    .alert-success { background: #14B8A615; border: 1px solid #14B8A640; color: #2DD4BF; }
    .alert-warning { background: #FBBF2415; border: 1px solid #FBBF2440; color: #FCD34D; }
    .alert-info { background: #131F2E; border: 1px solid #1E3044; color: #94A3B8; }
    .sidebar-card {
        background: #131F2E;
        border: 1px solid #1E3044;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.82rem;
        color: #94A3B8;
        line-height: 1.5;
    }
    .sidebar-card strong { color: #F9FAFB; }
    .page-footer {
        text-align: center;
        color: #64748B;
        font-size: 0.8rem;
        padding: 1.5rem 0 0.5rem;
        border-top: 1px solid #1E3044;
        margin-top: 2rem;
    }
    .page-footer strong { color: #94A3B8; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
"""

st.set_page_config(
    page_title="Dual-Model Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

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

_models_loaded = st.session_state.xai_package is not None
_chip_ready = '<span class="status-chip">Models ready</span>' if _models_loaded else '<span class="status-chip pending">Load models to begin</span>'
_chip_xai = '<span class="status-chip">XAI ready</span>' if _models_loaded else '<span class="status-chip pending">XAI pending</span>'
st.markdown(
    f'<div class="page-hero">'
    f'<h1>🛡️ Dual-Model Intrusion Detection System</h1>'
    f'<p>Random Forest + XGBoost with SHAP, LIME &amp; ELI5 explanations</p>'
    f'<div class="status-chips">{_chip_ready}{_chip_xai}<span class="status-chip">Dual model</span></div>'
    f'</div>',
    unsafe_allow_html=True,
)

# --- sidebar ---
with st.sidebar:
    st.markdown("### Control Panel")

    if st.button("Load Models", type="primary"):
        with st.spinner("Loading models and initializing XAI..."):
            package = load_models()
            if package:
                st.session_state.xai_package = package
                explainers = initialize_xai_explainers(package)
                st.session_state.rf_explainers = explainers.get('rf', {})
                st.session_state.xgb_explainers = explainers.get('xgb', {})
                try:
                    st.toast("Models loaded successfully.", icon="✅")
                except Exception:
                    st.success("Models loaded successfully.")

    if st.session_state.xai_package is None:
        render_alert("Load models from the button above to start analysis.", "info")
        st.stop()

    render_section_title("Input Method", "Sample data or CSV upload")
    input_method = st.radio(
        "Choose input method:",
        ["Use Sample Data", "Upload File"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="sidebar-card">'
        '<strong>Random Forest</strong><br>'
        '<strong>XGBoost</strong><br><br>'
        '<strong>XAI:</strong> SHAP · LIME · ELI5'
        '</div>',
        unsafe_allow_html=True,
    )

# --- main ---
package = st.session_state.xai_package

# --- input: built-in test rows or CSV upload ---
if input_method == "Use Sample Data":
    render_section_title("Sample Data", "Pick a test row for each model")

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
            st.caption(f"Sample {rf_sample_display} selected")
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
            st.caption(f"Sample {xgb_sample_display} selected")
        except Exception as e:
            st.error(f"Error loading XGBoost samples: {e}")
            st.stop()

    render_kpi_row([
        {"label": "RF sample", "value": str(rf_sample_display), "sub": "test row index"},
        {"label": "XGB sample", "value": str(xgb_sample_display), "sub": "test row index"},
        {"label": "Input mode", "value": "Sample", "sub": "built-in data"},
        {"label": "Models", "value": "2", "sub": "RF + XGBoost"},
        {"label": "XAI", "value": "3", "sub": "SHAP · LIME · ELI5"},
    ])

elif input_method == "Upload File":
    render_section_title("File Upload", "CSV with header — columns auto-mapped")
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
    attack_idx_rf = np.where(rf_labels == 'Attack')[0]
    attack_idx_xgb = np.where(xgb_labels == 'Attack')[0]
    attack_idx_union = np.unique(np.concatenate([attack_idx_rf, attack_idx_xgb]))
    both_agree_attack = np.intersect1d(attack_idx_rf, attack_idx_xgb)
    n_both = len(both_agree_attack)
    n_either = len(attack_idx_union)
    render_section_title("Batch Summary", "Overview for decision-making")
    render_kpi_row([
        {"label": "Total flows", "value": f"{n_rows:,}", "sub": "rows processed"},
        {"label": "RF flagged", "value": str(rf_attack), "sub": f"{rf_normal} normal"},
        {"label": "XGB flagged", "value": str(xgb_attack), "sub": f"{xgb_normal} normal"},
        {"label": "Both agree", "value": str(n_both), "sub": "attack consensus"},
        {"label": "Either flagged", "value": str(n_either), "sub": "needs review"},
    ])
    if 'Label' in raw.columns:
        true_labels = raw['Label'].astype(str).str.strip()
        true_attack = int((true_labels.str.lower() == 'attack').sum())
        true_normal = int((true_labels.str.lower() == 'normal').sum())
        st.caption(f"Ground truth (if present): Attack {true_attack}, Normal {true_normal}")
    render_section_title("Suggested Actions")
    if len(attack_idx_union) == 0:
        render_alert("No flows were predicted as Attack. Traffic appears normal; continue monitoring.", "success")
    else:
        render_alert(
            f"{n_either} flow(s) predicted as Attack (both models agree on {n_both} — higher confidence). "
            "Quarantine or deep-inspect these flows; export for SIEM/log review.",
            "warning",
        )
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
    st.caption(f"Processed {n_rows} row(s). Select a row below to view prediction and XAI.")
    row_options = list(range(1, n_rows + 1))
    upload_row = st.selectbox("Row to predict and explain:", row_options, key="upload_row") - 1
    rf_input = np.asarray(X_rf_upload[upload_row:upload_row+1], dtype=np.float64)
    xgb_input = np.asarray(X_xgb_upload[upload_row:upload_row+1], dtype=np.float64)

# --- single-row prediction + explanations ---
render_section_title("Model Predictions", "Side-by-side RF and XGBoost with XAI")

col1, col2 = st.columns(2)

with col1:
    rf_pred, rf_conf, rf_proba, rf_pred_idx = get_prediction_rf(rf_input, package)
    render_prediction_panel("Random Forest", rf_pred, rf_conf, rf_proba, accent='rf')

    render_section_title("Why this prediction?", "Key drivers toward the predicted class")
    st.caption(
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
                fig, ax = create_chart(figsize=(10, 4))
                features = chart_feature_labels(supporting_reasons)
                impacts = [r['impact'] for r in supporting_reasons]
                ax.barh(range(len(features)), impacts, color=CHART_TEAL, alpha=0.85)
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features, color=CHART_TEXT)
                ax.set_xlabel(chart_xlabel_toward_prediction())
                ax.set_title(chart_title(pred_label, "Random Forest"))
                ax.axvline(x=0, color='#64748B', linestyle='--', linewidth=0.5)
                polish_chart(ax)
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

    render_section_title("Detailed XAI", "SHAP, LIME and ELI5 for Random Forest")
    with st.expander("SHAP — Feature Contributions", expanded=True):
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
                fig, ax = create_chart(figsize=(10, 6))
                colors = bar_colors(df_shap['SHAP Value'])
                ax.barh(range(len(df_shap)), df_shap['SHAP Value'], color=colors)
                ax.set_yticks(range(len(df_shap)))
                ax.set_yticklabels(df_shap['Feature'], color=CHART_TEXT)
                ax.set_xlabel('SHAP Value (Impact on Prediction)')
                ax.set_title('Top 15 Features — SHAP Values')
                ax.axvline(x=0, color='#64748B', linestyle='--', linewidth=0.5)
                polish_chart(ax)
                plt.tight_layout()
                st.pyplot(fig)
                st.dataframe(df_shap, use_container_width=True)
            else:
                st.warning("SHAP explainer not initialized")
        except Exception as e:
            st.error(f"SHAP error: {e}")

    with st.expander("LIME — Local Interpretable Model"):
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
                fig, ax = create_chart(figsize=(10, 6))
                colors = bar_colors(df_lime['Weight'])
                ax.barh(range(len(df_lime)), df_lime['Weight'], color=colors)
                ax.set_yticks(range(len(df_lime)))
                ax.set_yticklabels(df_lime['Feature Condition'], fontsize=8, color=CHART_TEXT)
                ax.set_xlabel('LIME Weight')
                ax.set_title('Top 15 Features — LIME Weights')
                ax.axvline(x=0, color='#64748B', linestyle='--', linewidth=0.5)
                polish_chart(ax)
                plt.tight_layout()
                st.pyplot(fig)
                st.dataframe(df_lime, use_container_width=True)
            else:
                st.warning("LIME explainer not initialized")
        except Exception as e:
            st.error(f"LIME error: {e}")

    with st.expander("ELI5 — Global Feature Importance"):
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
                fig, ax = create_chart(figsize=(10, 6))
                ax.barh(range(len(df_eli5)), df_eli5['Importance'], color=CHART_TEAL, alpha=0.85)
                ax.set_yticks(range(len(df_eli5)))
                ax.set_yticklabels(df_eli5['Feature'], color=CHART_TEXT)
                ax.set_xlabel('Permutation Importance')
                ax.set_title('Top 15 Features — ELI5 Global Importance')
                polish_chart(ax)
                plt.tight_layout()
                st.pyplot(fig)
                st.dataframe(df_eli5, use_container_width=True)
            else:
                st.warning("ELI5 importance not computed")
        except Exception as e:
            st.error(f"ELI5 error: {e}")

with col2:
        xgb_pred, xgb_conf, xgb_proba, xgb_pred_idx = get_prediction_xgb(xgb_input, package)
        render_prediction_panel("XGBoost", xgb_pred, xgb_conf, xgb_proba, accent='xgb')

        render_section_title("Why this prediction?", "Key drivers toward the predicted class")
        st.caption(
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
                    fig, ax = create_chart(figsize=(10, 4))
                    features = chart_feature_labels(supporting_reasons)
                    impacts = [r['impact'] for r in supporting_reasons]
                    ax.barh(range(len(features)), impacts, color=CHART_TEAL, alpha=0.85)
                    ax.set_yticks(range(len(features)))
                    ax.set_yticklabels(features, color=CHART_TEXT)
                    ax.set_xlabel(chart_xlabel_toward_prediction())
                    ax.set_title(chart_title(pred_label, "XGBoost"))
                    ax.axvline(x=0, color='#64748B', linestyle='--', linewidth=0.5)
                    polish_chart(ax)
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
        
        render_section_title("Detailed XAI", "SHAP, LIME and ELI5 for XGBoost")
        with st.expander("SHAP — Feature Contributions", expanded=True):
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
                    
                    fig, ax = create_chart(figsize=(10, 6))
                    colors = bar_colors(df_shap['SHAP Value'])
                    ax.barh(range(len(df_shap)), df_shap['SHAP Value'], color=colors)
                    ax.set_yticks(range(len(df_shap)))
                    ax.set_yticklabels(df_shap['Feature'], color=CHART_TEXT)
                    ax.set_xlabel('SHAP Value (Impact on Prediction)')
                    ax.set_title('Top 15 Features — SHAP Values')
                    ax.axvline(x=0, color='#64748B', linestyle='--', linewidth=0.5)
                    polish_chart(ax)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.dataframe(df_shap, use_container_width=True)
                else:
                    st.warning("SHAP explainer not initialized")
            except Exception as e:
                st.error(f"SHAP error: {e}")
        
        with st.expander("LIME — Local Interpretable Model"):
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
                    
                    fig, ax = create_chart(figsize=(10, 6))
                    colors = bar_colors(df_lime['Weight'])
                    ax.barh(range(len(df_lime)), df_lime['Weight'], color=colors)
                    ax.set_yticks(range(len(df_lime)))
                    ax.set_yticklabels(df_lime['Feature Condition'], fontsize=8, color=CHART_TEXT)
                    ax.set_xlabel('LIME Weight')
                    ax.set_title('Top 15 Features — LIME Weights')
                    ax.axvline(x=0, color='#64748B', linestyle='--', linewidth=0.5)
                    polish_chart(ax)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.dataframe(df_lime, use_container_width=True)
                else:
                    st.warning("LIME explainer not initialized")
            except Exception as e:
                st.error(f"LIME error: {e}")
        
        with st.expander("ELI5 — Global Feature Importance"):
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
                    
                    fig, ax = create_chart(figsize=(10, 6))
                    ax.barh(range(len(df_eli5)), df_eli5['Importance'], color=CHART_TEAL, alpha=0.85)
                    ax.set_yticks(range(len(df_eli5)))
                    ax.set_yticklabels(df_eli5['Feature'], color=CHART_TEXT)
                    ax.set_xlabel('Permutation Importance')
                    ax.set_title('Top 15 Features — ELI5 Global Importance')
                    polish_chart(ax)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.dataframe(df_eli5, use_container_width=True)
                else:
                    st.warning("ELI5 importance not computed")
            except Exception as e:
                st.error(f"ELI5 error: {e}")

# --- model agreement ---
conf_diff = abs(rf_conf - xgb_conf)
agreement_yes = rf_pred == xgb_pred
render_section_title("Model Comparison", "Agreement and confidence across both models")
render_kpi_row([
    {"label": "Agreement", "value": "Yes" if agreement_yes else "No", "sub": "same prediction"},
    {"label": "RF prediction", "value": rf_pred, "sub": f"{rf_conf:.1f}% confidence"},
    {"label": "XGB prediction", "value": xgb_pred, "sub": f"{xgb_conf:.1f}% confidence"},
    {"label": "Confidence Difference", "value": f"{conf_diff:.1f}%", "sub": "RF vs XGB"},
    {"label": "Status", "value": "Aligned" if agreement_yes else "Review", "sub": "action hint"},
])
if rf_pred != xgb_pred:
    render_alert("Models disagree — review XAI explanations to understand why.", "warning")
else:
    render_alert("Both models predict the same class.", "success")

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

    render_section_title("AI Analyst Notes", "Optional LLM advisory for flagged rows")
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

st.markdown(
    '<div class="page-footer">'
    '<strong>Dual-Model Intrusion Detection</strong><br>'
    'Random Forest + XGBoost · SHAP · LIME · ELI5'
    '</div>',
    unsafe_allow_html=True,
)

