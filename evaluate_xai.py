"""
XAI evaluation on a fixed stratified test sample (stability, fidelity, agreement).

  python evaluate_xai.py
  python evaluate_xai.py --n100
"""

import argparse
import sys
import time
import warnings
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent
SECOND = BASE / 'SecondModel'
OUT_DIR = BASE / 'Presentation_Evidence'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed constants
SAMPLE_SEED = 42
N_SAMPLE = 500
N_ATTACK = 250
N_NORMAL = 250
LIME_RUNS = 5
LIME_SEEDS = list(range(42, 42 + LIME_RUNS))
TOP_K_STABILITY = 15
TOP_K_FIDELITY = [3, 5]
TOP_K_RANKING = 15
TOP_K_JACCARD = 10

# Indices output path (for reproducibility)
INDICES_PATH = OUT_DIR / 'xai_thesis_sample_500_indices.json'
RESULTS_TXT_PATH = OUT_DIR / 'xai_thesis_evaluation_results.txt'
TABLE_A_PATH = OUT_DIR / 'xai_thesis_table_a_metrics.txt'
TABLE_B_PATH = OUT_DIR / 'xai_thesis_table_b_agreement.txt'
PPT_SNIPPET_PATH = OUT_DIR / 'XAI_VALIDATION_PROGRESS_PPT.txt'


def load_models_and_test():
    """Load RF and XGB models and test arrays. Returns dict with rf_* and xgb_* keys."""
    out = {}
    try:
        out['rf_model'] = joblib.load(SECOND / 'rf_model_binary_cicids2017.pkl')
        out['rf_X'] = np.load(SECOND / 'X_test_rf_cicids2017.npy')
        out['rf_y'] = np.load(SECOND / 'y_test_rf_cicids2017.npy')
        out['rf_feat'] = joblib.load(SECOND / 'feature_names_rf_cicids2017.pkl')
    except Exception as e:
        out['rf_error'] = str(e)
    try:
        out['xgb_model'] = joblib.load(SECOND / 'xgb_model_binary_cicids2017_95percent.pkl')
        out['xgb_X'] = np.load(SECOND / 'X_test_xgb_cicids2017_95percent.npy')
        out['xgb_y'] = np.load(SECOND / 'y_test_xgb_cicids2017_95percent.npy')
        out['xgb_feat'] = joblib.load(SECOND / 'feature_names_xgb_cicids2017_95percent.pkl')
    except Exception as e:
        out['xgb_error'] = str(e)
    return out


def get_stratified_sample_indices(y, n_attack=250, n_normal=250, random_state=42):
    """Return indices for n_attack Attack + n_normal Normal. y: 0=Attack, 1=Normal."""
    rng = np.random.default_rng(random_state)
    idx_attack = np.where(y == 0)[0]
    idx_normal = np.where(y == 1)[0]
    if len(idx_attack) < n_attack or len(idx_normal) < n_normal:
        raise ValueError(f"Not enough samples: Attack={len(idx_attack)}, Normal={len(idx_normal)}")
    sel_attack = rng.choice(idx_attack, size=n_attack, replace=False)
    sel_normal = rng.choice(idx_normal, size=n_normal, replace=False)
    indices = np.concatenate([sel_attack, sel_normal])
    rng.shuffle(indices)
    return indices.tolist()


def _get_shap_explainer(model, X_background, use_kernel_fallback=True):
    import shap
    try:
        return shap.TreeExplainer(model)
    except (ValueError, TypeError) as e:
        if not use_kernel_fallback:
            raise
        err_msg = str(e) if e else ''
        if 'base_score' in err_msg or 'convert string to float' in err_msg:
            bg = np.asarray(X_background, dtype=np.float64)
            if len(bg) > 50:
                bg = bg[:50]
            return shap.KernelExplainer(model.predict_proba, bg)
        raise e


def _shap_values_safe(explainer, X, n_feat=None):
    raw = explainer.shap_values(X)
    if raw is None:
        return None
    if isinstance(raw, list):
        raw = raw[0] if len(raw) else None
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64)
    while arr.ndim > 2:
        arr = arr.mean(axis=-1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if n_feat is not None and arr.shape[1] > n_feat:
        arr = arr[:, :n_feat]
    return arr


def run_lime_stability(X, y, model, feat_names, indices):
    """LIME stability: R=5 runs per instance, top-k=15, Spearman across runs. Report stats + config."""
    import lime.lime_tabular
    X_s = X[indices].astype(np.float64)
    y_s = y[indices]
    n_feat = len(feat_names)
    k = min(TOP_K_STABILITY, n_feat)
    class_names = ['Attack', 'Normal']

    # LIME config (defaults used in explain_instance)
    # LimeTabularExplainer: discretize_continuous=True, discretizer='quartile', kernel_width=None -> sqrt(n_feat)
    num_samples = 5000  # default in explain_instance
    discretizer = 'quartile'
    kernel_width = np.sqrt(n_feat)
    feature_selection = 'auto'
    lime_config = {
        'num_samples': num_samples,
        'discretizer': discretizer,
        'kernel_width': float(kernel_width),
        'feature_selection': feature_selection,
    }

    spearman_per_instance = []
    for i in range(len(X_s)):
        ranks_per_run = []
        for seed in LIME_SEEDS:
            exp = lime.lime_tabular.LimeTabularExplainer(
                X_s, feature_names=feat_names, class_names=class_names,
                mode='classification', random_state=seed
            )
            exp_result = exp.explain_instance(
                X_s[i], model.predict_proba, num_features=k,
                num_samples=num_samples
            )
            # Ranking by |weight| for top-k features (by index)
            lime_list = exp_result.as_list()
            # Map feature name -> rank (0 = most important)
            name_to_rank = {}
            for r, (name, w) in enumerate(sorted(lime_list, key=lambda x: -abs(x[1]))):
                for fi, f in enumerate(feat_names):
                    if f == name or (isinstance(name, str) and f in name):
                        name_to_rank[fi] = r
                        break
            rank_vec = np.array([name_to_rank.get(j, n_feat) for j in range(n_feat)])
            ranks_per_run.append(rank_vec)
        # Pairwise Spearman between run0 and run1, run0 and run2, ... then average per instance
        rhos = []
        for r in range(1, len(ranks_per_run)):
            rho, _ = spearmanr(ranks_per_run[0], ranks_per_run[r])
            if not np.isnan(rho):
                rhos.append(rho)
        spearman_per_instance.append(np.mean(rhos) if rhos else np.nan)

    spearman_per_instance = np.array([x for x in spearman_per_instance if not np.isnan(x)])
    if len(spearman_per_instance) == 0:
        return {'done': False, 'lime_config': lime_config}

    def stats(v):
        v = np.asarray(v)
        return {
            'mean': float(np.mean(v)),
            'std': float(np.std(v)),
            'median': float(np.median(v)),
            'q25': float(np.percentile(v, 25)),
            'q75': float(np.percentile(v, 75)),
            'min': float(np.min(v)),
            'max': float(np.max(v)),
            'pct_below_0_5': float(100 * np.mean(v < 0.5)),
        }

    return {
        'done': True,
        'lime_config': lime_config,
        'spearman_stats': stats(spearman_per_instance),
        'n_instances': len(spearman_per_instance),
    }


def run_fidelity(X, y, model, feat_names, indices, explainer_shap, lime_explainer):
    """Fidelity: SHAP, LIME, ELI5; k=3 and k=5; ablate to 0 (standardized space); flips, acc drop, Δp_attack."""
    from eli5.sklearn import PermutationImportance
    X_s = X[indices].astype(np.float64)
    y_s = y[indices]
    n_feat = len(feat_names)
    n_sample = len(X_s)
    # Class index for Attack (predict_proba column)
    classes = model.classes_
    attack_idx = int(np.argmin(classes)) if hasattr(classes, '__len__') else 0  # 0=Attack typically

    results = {}
    for method in ['SHAP', 'LIME', 'ELI5']:
        results[method] = {}

    # ELI5 one-off
    eli5_obj = PermutationImportance(model, random_state=42, n_iter=5, scoring='accuracy')
    eli5_obj.fit(X_s, y_s)
    eli5_importances = eli5_obj.feature_importances_

    for k in TOP_K_FIDELITY:
        k = min(k, n_feat)
        for method in ['SHAP', 'LIME', 'ELI5']:
            X_abl = X_s.copy()
            if method == 'SHAP':
                for i in range(n_sample):
                    sv = _shap_values_safe(explainer_shap, X_s[i:i+1], n_feat)
                    if sv is not None and sv.size > 0:
                        top_idx = np.argsort(np.abs(sv).flatten())[-k:]
                        X_abl[i, top_idx] = 0
                    else:
                        top_idx = np.argsort(-eli5_importances)[:k]
                        X_abl[i, top_idx] = 0
            elif method == 'LIME':
                for i in range(n_sample):
                    exp = lime_explainer.explain_instance(
                        X_s[i], model.predict_proba,
                        num_features=min(TOP_K_STABILITY, n_feat)
                    )
                    lime_list = exp.as_list()
                    top_names = [x[0] for x in sorted(lime_list, key=lambda x: -abs(x[1]))[:k]]
                    for fi, f in enumerate(feat_names):
                        if any(f == n or (isinstance(n, str) and f in n) for n in top_names):
                            X_abl[i, fi] = 0
            else:
                top_indices = np.argsort(-eli5_importances)[:k]
                X_abl[:, top_indices] = 0

            pred_orig = model.predict(X_s)
            pred_abl = model.predict(X_abl)
            flips = int(np.sum(pred_orig != pred_abl))
            acc_orig = accuracy_score(y_s, pred_orig)
            acc_abl = accuracy_score(y_s, pred_abl)
            acc_drop = (acc_orig - acc_abl) * 100
            flips_rate = 100.0 * flips / n_sample

            # Δp_attack for Attack samples only
            attack_mask = (y_s == 0)
            if attack_mask.any():
                p_orig = model.predict_proba(X_s)[:, attack_idx]
                p_abl = model.predict_proba(X_abl)[:, attack_idx]
                dp = p_orig - p_abl
                dp_attack = dp[attack_mask]
                mean_dp = float(np.mean(dp_attack))
                median_dp = float(np.median(dp_attack))
            else:
                mean_dp = median_dp = None

            results[method][k] = {
                'flips': flips,
                'flips_rate_pct': round(flips_rate, 2),
                'acc_drop_pct': round(acc_drop, 2),
                'mean_dp_attack': mean_dp,
                'median_dp_attack': median_dp,
            }

    return {'done': True, 'fidelity': results}


def run_runtime(X, y, model, feat_names, indices, explainer_shap, lime_explainer):
    """SHAP and LIME per-instance runtime (mean, median, p95). ELI5 one-off time."""
    X_s = X[indices].astype(np.float64)
    y_s = y[indices]
    n_sample = len(X_s)
    n_feat = len(feat_names)

    times_shap = []
    for i in range(n_sample):
        t0 = time.perf_counter()
        _shap_values_safe(explainer_shap, X_s[i:i+1], n_feat)
        times_shap.append(time.perf_counter() - t0)

    times_lime = []
    for i in range(n_sample):
        t0 = time.perf_counter()
        lime_explainer.explain_instance(
            X_s[i], model.predict_proba,
            num_features=min(TOP_K_STABILITY, n_feat)
        )
        times_lime.append(time.perf_counter() - t0)

    from eli5.sklearn import PermutationImportance
    t0 = time.perf_counter()
    PermutationImportance(model, random_state=42, n_iter=5, scoring='accuracy').fit(X_s, y_s)
    eli5_sec = time.perf_counter() - t0

    def stats_sec(times):
        t = np.array(times)
        return {
            'mean_s': round(float(np.mean(t)), 4),
            'median_s': round(float(np.median(t)), 4),
            'p95_s': round(float(np.percentile(t, 95)), 4),
        }

    return {
        'done': True,
        'SHAP': stats_sec(times_shap),
        'LIME': stats_sec(times_lime),
        'ELI5_one_off_s': round(eli5_sec, 2),
    }


def run_ranking_agreement(X, y, model, feat_names, indices, explainer_shap, lime_explainer):
    """Spearman SHAP vs LIME, SHAP vs ELI5, LIME vs ELI5 (top-k=15). Distribution + Jaccard top-10."""
    from eli5.sklearn import PermutationImportance
    X_s = X[indices].astype(np.float64)
    y_s = y[indices]
    n_feat = len(feat_names)
    k_rank = min(TOP_K_RANKING, n_feat)
    k_jaccard = min(TOP_K_JACCARD, n_feat)

    eli5_obj = PermutationImportance(model, random_state=42, n_iter=5, scoring='accuracy')
    eli5_obj.fit(X_s, y_s)
    eli5_global_rank = np.argsort(-eli5_obj.feature_importances_)

    corr_sl = []
    corr_se = []
    corr_le = []
    jaccard_sl = []
    jaccard_se = []
    jaccard_le = []

    for i in range(len(X_s)):
        # SHAP rank
        sv = _shap_values_safe(explainer_shap, X_s[i:i+1], n_feat)
        if sv is None or sv.size == 0:
            continue
        shap_rank = np.argsort(np.argsort(-np.abs(sv.flatten())))
        # LIME rank
        exp = lime_explainer.explain_instance(
            X_s[i], model.predict_proba, num_features=k_rank
        )
        lime_list = exp.as_list()
        lime_rank = np.full(n_feat, n_feat)
        for r, (name, w) in enumerate(sorted(lime_list, key=lambda x: -abs(x[1]))):
            for fi, f in enumerate(feat_names):
                if f == name or (isinstance(name, str) and f in name):
                    lime_rank[fi] = r
                    break
        # ELI5 is same for all instances
        eli5_rank = np.argsort(np.argsort(-eli5_obj.feature_importances_))

        rho_sl, _ = spearmanr(shap_rank, lime_rank)
        rho_se, _ = spearmanr(shap_rank, eli5_rank)
        rho_le, _ = spearmanr(lime_rank, eli5_rank)
        if not np.isnan(rho_sl):
            corr_sl.append(rho_sl)
        if not np.isnan(rho_se):
            corr_se.append(rho_se)
        if not np.isnan(rho_le):
            corr_le.append(rho_le)

        # Jaccard top-10
        def top_k_set(rank_vec, k):
            return set(np.argsort(rank_vec)[:k])
        set_shap = top_k_set(shap_rank, k_jaccard)
        set_lime = top_k_set(lime_rank, k_jaccard)
        set_eli5 = top_k_set(eli5_rank, k_jaccard)
        jaccard_sl.append(len(set_shap & set_lime) / len(set_shap | set_lime) if (set_shap | set_lime) else 0)
        jaccard_se.append(len(set_shap & set_eli5) / len(set_shap | set_eli5) if (set_shap | set_eli5) else 0)
        jaccard_le.append(len(set_lime & set_eli5) / len(set_lime | set_eli5) if (set_lime | set_eli5) else 0)

    def dist_stats(v, name):
        v = np.array(v)
        return {
            'mean': round(float(np.mean(v)), 4),
            'median': round(float(np.median(v)), 4),
            'q25': round(float(np.percentile(v, 25)), 4),
            'q75': round(float(np.percentile(v, 75)), 4),
            'min': round(float(np.min(v)), 4),
            'max': round(float(np.max(v)), 4),
            'pct_negative': round(100 * np.mean(v < 0), 2),
        }

    def jaccard_stats(v):
        v = np.array(v)
        return {'mean': round(float(np.mean(v)), 4), 'median': round(float(np.median(v)), 4)}

    return {
        'done': True,
        'Spearman': {
            'SHAP_vs_LIME': dist_stats(corr_sl, 'SL'),
            'SHAP_vs_ELI5': dist_stats(corr_se, 'SE'),
            'LIME_vs_ELI5': dist_stats(corr_le, 'LE'),
        },
        'Jaccard_top10': {
            'SHAP_vs_LIME': jaccard_stats(jaccard_sl),
            'SHAP_vs_ELI5': jaccard_stats(jaccard_se),
            'LIME_vs_ELI5': jaccard_stats(jaccard_le),
        },
    }


def run_all_for_model(data, model_key, indices):
    """Run stability, fidelity, runtime, ranking for one model. Build explainers once."""
    import lime.lime_tabular
    X = data[f'{model_key}_X'].astype(np.float64)
    y = data[f'{model_key}_y']
    model = data[f'{model_key}_model']
    feat = data[f'{model_key}_feat']
    X_s = X[indices]
    y_s = y[indices]
    class_names = ['Attack', 'Normal']

    explainer_shap = _get_shap_explainer(model, X_s)
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_s, feature_names=feat, class_names=class_names,
        mode='classification', random_state=42
    )

    out = {}
    out['stability'] = run_lime_stability(X, y, model, feat, indices)
    out['fidelity'] = run_fidelity(X, y, model, feat, indices, explainer_shap, lime_explainer)
    out['runtime'] = run_runtime(X, y, model, feat, indices, explainer_shap, lime_explainer)
    # runtime used y_s in ELI5 fit; pass y again for ranking
    out['ranking'] = run_ranking_agreement(X, y, model, feat, indices, explainer_shap, lime_explainer)
    return out


def write_results_txt(all_results, indices, path, meta=None):
    """meta: optional dict with n_sample, n_attack, n_normal, indices_line (str)."""
    m = {
        'n_sample': N_SAMPLE,
        'n_attack': N_ATTACK,
        'n_normal': N_NORMAL,
        'indices_line': f"Indices saved to: {INDICES_PATH}",
    }
    if meta:
        m.update({k: v for k, v in meta.items() if v is not None})

    lines = [
        "=" * 70,
        f"XAI THESIS EVALUATION — Reproducible Results (N={m['n_sample']} stratified)",
        "=" * 70,
        "",
        f"Sample: N={m['n_sample']} (Attack={m['n_attack']}, Normal={m['n_normal']}), random_seed={SAMPLE_SEED}.",
        m['indices_line'],
        "",
        "--- Indices used (first 20 and last 5) ---",
        str(indices[:20]) + " ... " + str(indices[-5:]),
        "",
        "=" * 70,
    ]

    for model_label, model_key in [('RF', 'rf'), ('XGB', 'xgb')]:
        if model_key not in all_results:
            continue
        r = all_results[model_key]
        lines.append("")
        lines.append(f"--- {model_label} ---")
        # Stability
        s = r.get('stability', {})
        if s.get('done'):
            lines.append("  LIME Stability (R=5 runs, top-k=15):")
            lines.append(f"    Spearman: mean={s['spearman_stats']['mean']:.4f}, std={s['spearman_stats']['std']:.4f}, "
                        f"median={s['spearman_stats']['median']:.4f}, IQR=[{s['spearman_stats']['q25']:.4f}, {s['spearman_stats']['q75']:.4f}], "
                        f"min={s['spearman_stats']['min']:.4f}, max={s['spearman_stats']['max']:.4f}, "
                        f"% Spearman<0.5 = {s['spearman_stats']['pct_below_0_5']:.2f}%")
            lines.append(f"    LIME config: num_samples={s['lime_config']['num_samples']}, discretizer={s['lime_config']['discretizer']}, "
                        f"kernel_width={s['lime_config']['kernel_width']:.2f}, feature_selection={s['lime_config']['feature_selection']}")
        # Fidelity
        f = r.get('fidelity', {})
        if f.get('done'):
            lines.append("  Fidelity (ablate top-k to 0 in standardized space):")
            for method in ['SHAP', 'LIME', 'ELI5']:
                for k in TOP_K_FIDELITY:
                    if k in f.get('fidelity', {}).get(method, {}):
                        v = f['fidelity'][method][k]
                        dp_mean = v.get('mean_dp_attack')
                        dp_med = v.get('median_dp_attack')
                        lines.append(f"    {method} k={k}: flips={v['flips']}, flips_rate%={v['flips_rate_pct']}, acc_drop%={v['acc_drop_pct']}, "
                                    f"mean_Δp_attack={dp_mean}, median_Δp_attack={dp_med}")
        # Runtime
        rt = r.get('runtime', {})
        if rt.get('done'):
            lines.append("  Runtime: SHAP mean/median/p95 (s): "
                        f"{rt['SHAP']['mean_s']}/{rt['SHAP']['median_s']}/{rt['SHAP']['p95_s']}; "
                        f"LIME: {rt['LIME']['mean_s']}/{rt['LIME']['median_s']}/{rt['LIME']['p95_s']}; "
                        f"ELI5 one-off: {rt['ELI5_one_off_s']} s")
        # Ranking
        rank = r.get('ranking', {})
        if rank.get('done'):
            lines.append("  Feature ranking agreement (Spearman, top-k=15):")
            for pair, stats in rank.get('Spearman', {}).items():
                lines.append(f"    {pair}: mean={stats['mean']}, median={stats['median']}, IQR=[{stats['q25']},{stats['q75']}], "
                            f"min={stats['min']}, max={stats['max']}, % negative={stats['pct_negative']}%")
            lines.append("  Jaccard top-10:")
            for pair, j in rank.get('Jaccard_top10', {}).items():
                lines.append(f"    {pair}: mean={j['mean']}, median={j['median']}")

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def write_table_a(all_results, path):
    """Table A: XAI metrics summary (stability, fidelity, runtime)."""
    rows = []
    for model_label, model_key in [('RF', 'rf'), ('XGB', 'xgb')]:
        if model_key not in all_results:
            continue
        r = all_results[model_key]
        s = r.get('stability', {})
        f = r.get('fidelity', {})
        rt = r.get('runtime', {})
        row = {'Model': model_label}
        if s.get('done'):
            row['LIME_Spearman_mean'] = f"{s['spearman_stats']['mean']:.4f}"
            row['LIME_Spearman_%<0.5'] = f"{s['spearman_stats']['pct_below_0_5']:.2f}%"
        if f.get('done'):
            fid = f.get('fidelity', {})
            for k in TOP_K_FIDELITY:
                for method in ['SHAP', 'LIME', 'ELI5']:
                    v = fid.get(method, {}).get(k, {})
                    row[f'{method}_flips_k{k}'] = v.get('flips', '')
                    row[f'{method}_acc_drop%_k{k}'] = v.get('acc_drop_pct', '')
        if rt.get('done'):
            row['SHAP_mean_s'] = rt['SHAP']['mean_s']
            row['LIME_mean_s'] = rt['LIME']['mean_s']
            row['ELI5_one_off_s'] = rt['ELI5_one_off_s']
        rows.append(row)
    df = pd.DataFrame(rows)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("Table A: XAI metrics summary (stability, fidelity, runtime)\n")
        f.write("=" * 60 + "\n")
        f.write(df.to_string(index=False) + "\n")


def write_table_b(all_results, path):
    """Table B: Agreement metrics (Spearman distribution + Jaccard overlap)."""
    lines = ["Table B: Agreement metrics (Spearman distribution + Jaccard top-10 overlap)", "=" * 60]
    for model_label, model_key in [('RF', 'rf'), ('XGB', 'xgb')]:
        if model_key not in all_results:
            continue
        r = all_results[model_key].get('ranking', {})
        if not r.get('done'):
            continue
        lines.append(f"\n{model_label}:")
        for pair, stats in r.get('Spearman', {}).items():
            lines.append(f"  {pair} Spearman: mean={stats['mean']}, median={stats['median']}, IQR=[{stats['q25']},{stats['q75']}], % neg={stats['pct_negative']}%")
        for pair, j in r.get('Jaccard_top10', {}).items():
            lines.append(f"  {pair} Jaccard: mean={j['mean']}, median={j['median']}")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def write_ppt_snippet(all_results: dict, path: Path, meta=None) -> None:
    """Progress-report / PowerPoint copy: how we validate + latest numeric results."""
    m = {
        'n_sample': N_SAMPLE,
        'n_attack': N_ATTACK,
        'n_normal': N_NORMAL,
        'indices_path': INDICES_PATH,
        'results_name': 'xai_thesis_evaluation_results.txt',
        'table_a_name': 'xai_thesis_table_a_metrics.txt',
        'table_b_name': 'xai_thesis_table_b_agreement.txt',
        'indices_blurb': None,
        'results_section_note': 'xai_thesis_table_a_metrics.txt and table_b',
        'regenerate_cmd': 'python evaluate_xai.py',
    }
    if meta:
        m.update({k: v for k, v in meta.items() if v is not None})
    if m['indices_blurb'] is None:
        m['indices_blurb'] = (
            f"The same row indices are saved for reproducibility (see {m['indices_path'].name})."
        )

    lines = [
        "XAI VALIDATION — copy for progress report or PowerPoint",
        "=" * 60,
        "",
        "HOW WE VALIDATE (methods)",
        "-" * 40,
        "We evaluate post-hoc explanations on a fixed stratified sample from the held-out ",
        f"test set: N={m['n_sample']} instances ({m['n_attack']} Attack, {m['n_normal']} Normal), seed={SAMPLE_SEED}. ",
        m['indices_blurb'],
        "",
        "Dimensions:",
        "  (1) Stability — LIME is run R=5 times per instance; Spearman correlation of feature ",
        f"     rankings across runs (top-k={TOP_K_STABILITY}).",
        "  (2) Fidelity — For SHAP, LIME, and ELI5, we ablate the top-k important features ",
        f"     (k in {TOP_K_FIDELITY}) and report prediction flips and accuracy drop on the sample.",
        "  (3) Runtime — Mean time per instance for SHAP and LIME; ELI5 is one global fit.",
        "  (4) Agreement — Spearman correlation and Jaccard overlap (top-10) between explainer ",
        f"     rankings (top-k={TOP_K_RANKING} for Spearman).",
        "",
        f"RESULTS (latest run — also in {m['results_section_note']})",
        "-" * 40,
    ]

    for label, key in [('Random Forest (CICIDS-2017)', 'rf'), ('XGBoost (CICIDS-2017)', 'xgb')]:
        if key not in all_results:
            continue
        r = all_results[key]
        lines.append(f"{label}:")
        s = r.get('stability', {})
        if s.get('done'):
            st = s['spearman_stats']
            lines.append(
                f"  • LIME stability: Spearman mean={st['mean']:.4f}, "
                f"% instances with Spearman<0.5 = {st['pct_below_0_5']:.2f}%"
            )
        f = r.get('fidelity', {})
        if f.get('done'):
            fid = f.get('fidelity', {})
            for k in TOP_K_FIDELITY:
                parts = []
                for method in ['SHAP', 'LIME', 'ELI5']:
                    v = fid.get(method, {}).get(k, {})
                    if v:
                        parts.append(
                            f"{method}: {v.get('flips', '?')} flips, "
                            f"{v.get('acc_drop_pct', '?')}% acc drop"
                        )
                if parts:
                    lines.append(f"  • Fidelity k={k}: " + " | ".join(parts))
        rt = r.get('runtime', {})
        if rt.get('done'):
            lines.append(
                f"  • Runtime (mean s): SHAP={rt['SHAP']['mean_s']:.4f}, "
                f"LIME={rt['LIME']['mean_s']:.4f}, ELI5 one-off={rt['ELI5_one_off_s']:.2f} s"
            )
        lines.append("")

    lines.append("Cross-method agreement (feature ranking)")
    lines.append("-" * 40)
    for label, key in [('RF', 'rf'), ('XGB', 'xgb')]:
        if key not in all_results:
            continue
        rk = all_results[key].get('ranking', {})
        if not rk.get('done'):
            continue
        lines.append(f"{label}:")
        for pair, stats in rk.get('Spearman', {}).items():
            lines.append(
                f"  • {pair}: Spearman mean={stats['mean']}, median={stats['median']}, "
                f"% negative={stats['pct_negative']}%"
            )
        for pair, j in rk.get('Jaccard_top10', {}).items():
            lines.append(f"  • {pair}: Jaccard top-10 mean={j['mean']}, median={j['median']}")
        lines.append("")

    lines.extend([
        "SOURCE FILES (project Presentation_Evidence/)",
        "-" * 40,
        f"  {m['results_name']}  — full narrative",
        f"  {m['table_a_name']}    — Table A (compact)",
        f"  {m['table_b_name']}    — Table B (agreement)",
        f"  {m['indices_path'].name}  — reproducible sample indices",
        "",
        f"Regenerate: {m['regenerate_cmd']}",
        "",
    ])
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description='XAI thesis evaluation (CICIDS-2017 RF / XGB).')
    parser.add_argument(
        '--n100',
        action='store_true',
        help='Use fixed 100-sample indices from xai_thesis_sample_100_indices.json; write *_n100.txt outputs.',
    )
    args = parser.parse_args()

    results_path = RESULTS_TXT_PATH
    table_a_path = TABLE_A_PATH
    table_b_path = TABLE_B_PATH
    ppt_path = PPT_SNIPPET_PATH
    indices_path = INDICES_PATH
    n_sample, n_attack, n_normal = N_SAMPLE, N_ATTACK, N_NORMAL
    indices_loaded = False

    print("Loading models and test data...")
    data = load_models_and_test()
    for key in ['rf', 'xgb']:
        if data.get(f'{key}_error'):
            print(f"  {key}: {data[f'{key}_error']}")
    if data.get('rf_error') and data.get('xgb_error'):
        sys.exit(1)

    indices_by_model = {}

    if args.n100:
        n_sample, n_attack, n_normal = 100, 50, 50
        indices_path = OUT_DIR / 'xai_thesis_sample_100_indices.json'
        if not indices_path.is_file():
            print(f"Missing indices file: {indices_path}", file=sys.stderr)
            sys.exit(1)
        with open(indices_path, encoding='utf-8') as f:
            blob = json.load(f)
        indices_by_model['rf'] = blob['rf_indices']
        indices_by_model['xgb'] = blob['xgb_indices']
        indices_loaded = True
        results_path = OUT_DIR / 'xai_thesis_evaluation_results_n100.txt'
        table_a_path = OUT_DIR / 'xai_thesis_table_a_metrics_n100.txt'
        table_b_path = OUT_DIR / 'xai_thesis_table_b_agreement_n100.txt'
        ppt_path = OUT_DIR / 'XAI_VALIDATION_PROGRESS_PPT_n100.txt'
        print(f"Using predefined indices from {indices_path.name} (N={n_sample}).")
    else:
        for model_key in ['rf', 'xgb']:
            if data.get(f'{model_key}_error'):
                continue
            y = data[f'{model_key}_y']
            try:
                idx = get_stratified_sample_indices(y, N_ATTACK, N_NORMAL, SAMPLE_SEED)
                indices_by_model[model_key] = idx
            except ValueError as e:
                print(f"  {model_key}: {e}")
                indices_by_model[model_key] = list(range(min(N_SAMPLE, len(y))))

        with open(INDICES_PATH, 'w', encoding='utf-8') as f:
            json.dump({
                'seed': SAMPLE_SEED, 'n_attack': N_ATTACK, 'n_normal': N_NORMAL,
                'rf_indices': indices_by_model.get('rf'),
                'xgb_indices': indices_by_model.get('xgb'),
            }, f, indent=2)
        print(f"Saved sample indices to {INDICES_PATH}")

    for k, idx in indices_by_model.items():
        if idx:
            print(f"  {k}: N={len(idx)} indices (first 10): {idx[:10]}")

    all_results = {}
    for model_key in ['rf', 'xgb']:
        if data.get(f'{model_key}_error'):
            continue
        idx = indices_by_model[model_key]
        if len(idx) != n_sample:
            print(f"  {model_key}: using {len(idx)} samples (requested {n_sample})")
        print(f"\nRunning XAI evaluation for {model_key.upper()}...")
        all_results[model_key] = run_all_for_model(data, model_key, idx)

    results_meta = {
        'n_sample': n_sample,
        'n_attack': n_attack,
        'n_normal': n_normal,
        'indices_line': (
            f"Indices loaded from: {indices_path.name} (unchanged; predefined sample)."
            if indices_loaded
            else f"Indices saved to: {indices_path}"
        ),
    }
    write_results_txt(
        all_results,
        indices_by_model.get('rf') or indices_by_model.get('xgb') or [],
        results_path,
        meta=results_meta,
    )
    write_table_a(all_results, table_a_path)
    write_table_b(all_results, table_b_path)
    ppt_meta = {
        'n_sample': n_sample,
        'n_attack': n_attack,
        'n_normal': n_normal,
        'indices_path': indices_path,
        'results_name': results_path.name,
        'table_a_name': table_a_path.name,
        'table_b_name': table_b_path.name,
        'results_section_note': f"{table_a_path.name} and {table_b_path.name}",
        'regenerate_cmd': 'python evaluate_xai.py --n100' if args.n100 else 'python evaluate_xai.py',
    }
    if indices_loaded:
        ppt_meta['indices_blurb'] = (
            f"Row indices are fixed in {indices_path.name} (50 Attack + 50 Normal per model test set)."
        )
    write_ppt_snippet(all_results, ppt_path, meta=ppt_meta)
    print(f"\nWrote {results_path}")
    print(f"Wrote {table_a_path}")
    print(f"Wrote {table_b_path}")
    print(f"Wrote {ppt_path}")


if __name__ == '__main__':
    main()
