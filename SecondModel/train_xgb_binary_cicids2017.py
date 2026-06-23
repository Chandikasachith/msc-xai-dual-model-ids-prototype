"""
XGBoost Binary Classification for CICIDS-2017 (Normal vs Attack)
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import argparse
import joblib
import time
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import seaborn as sns
import warnings
import sys
import io
from pathlib import Path

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# Minimal Feature Engineering
def add_engineered_features(df):
    """Add only essential engineered features"""
    df = df.copy()
    
    if 'Total Length of Fwd Packets' in df.columns and 'Total Length of Bwd Packets' in df.columns:
        df['total_bytes'] = df['Total Length of Fwd Packets'] + df['Total Length of Bwd Packets']
    
    if 'Total Fwd Packets' in df.columns and 'Total Bwd Packets' in df.columns:
        df['total_packets'] = df['Total Fwd Packets'] + df['Total Bwd Packets']
    
    return df

print("="*70)
print("XGBOOST BINARY INTRUSION DETECTION: CICIDS-2017")
print("="*70)
print("\nStrategy:")
print("  ✓ Remove duplicates")
print("  ✓ Fixed 70/15/15 split (Train/Val/Test)")
print("  ✓ Early stopping on validation set")
print("="*70)
print()

parser = argparse.ArgumentParser(description='XGBoost binary CICIDS-2017 (~95%% target).')
parser.add_argument('--five-runs', action='store_true', help='Run 5 times with seeds 42-46, report mean and 95%% CI (like RF).')
args = parser.parse_args()
FIVE_RUNS = args.five_runs
if FIVE_RUNS:
    print("MODE: 5 runs (seeds 42-46) -> mean accuracy and 95%% CI")
    print()

# Step 1: Load data
print("Step 1: Loading CICIDS-2017 data...")
data_dir = Path('cicids-2017')

parquet_files = [
    'Benign-Monday-no-metadata.parquet',
    'Botnet-Friday-no-metadata.parquet',
    'Bruteforce-Tuesday-no-metadata.parquet',
    'DDoS-Friday-no-metadata.parquet',
    'DoS-Wednesday-no-metadata.parquet',
    'Infiltration-Thursday-no-metadata.parquet',
    'Portscan-Friday-no-metadata.parquet',
    'WebAttacks-Thursday-no-metadata.parquet'
]

dataframes = []
for file in parquet_files:
    file_path = data_dir / file
    if file_path.exists():
        print(f"  Loading {file}...")
        df = pd.read_parquet(file_path)
        if 'Label' not in df.columns:
            label = file.split('-')[0]
            df['Label'] = label
        dataframes.append(df)
        print(f"    ✓ {len(df):,} samples")
    else:
        print(f"  File not found: {file}")

if not dataframes:
    print("ERROR: No data files found!")
    exit(1)

# Combine and remove duplicates
print("\nCombining all datasets...")
combined_df = pd.concat(dataframes, ignore_index=True)
print(f"✓ Total samples before deduplication: {combined_df.shape[0]:,}")

print("\nStep 1b: Removing duplicate samples...")
initial_count = len(combined_df)
combined_df = combined_df.drop_duplicates()
duplicates_removed = initial_count - len(combined_df)
print(f"✓ Removed {duplicates_removed:,} duplicate rows ({duplicates_removed/initial_count*100:.2f}%)")
print(f"✓ Total samples after deduplication: {combined_df.shape[0]:,}")
print()

# Step 2: Binary mapping
print("Step 2: Converting to Binary Classification...")
if 'Label' in combined_df.columns:
    combined_df['Label'] = combined_df['Label'].apply(
        lambda x: 'Normal' if str(x).lower() in ['benign', 'normal'] else 'Attack'
    )

print("✓ Binary class distribution:")
for cls, count in combined_df['Label'].value_counts().items():
    print(f"  {cls:10} → {count:8,} ({count/len(combined_df)*100:.1f}%)")
print()

# Step 3: Minimal Feature Engineering
print("Step 3: Adding Minimal Engineered Features...")
combined_df = add_engineered_features(combined_df)
print(f"✓ Features: {combined_df.shape[1]}")
print()

# Step 4: Preprocessing 
print("Step 4: Preprocessing...")
columns_to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 
                   'SimillarHTTP', 'Inbound', 'Fwd Header Length.1']

for col in columns_to_drop:
    if col in combined_df.columns:
        combined_df = combined_df.drop(col, axis=1)

X = combined_df.drop('Label', axis=1)
y = combined_df['Label']

X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# Encode
categorical_features = []
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
        categorical_features.append(col)

label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature].astype(str))
    label_encoders[feature] = le

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
print(f"✓ Classes: {list(target_encoder.classes_)}")

# Fixed 70/15/15 split (Train/Val/Test)
print("\nStep 5: Creating fixed 70/15/15 split (Train/Val/Test)...")
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# First split: 70% train, 30% temp 
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, 
    test_size=(VAL_SIZE + TEST_SIZE), 
    random_state=RANDOM_STATE, 
    stratify=y_encoded
)

# Second split: Split temp into 15% val and 15% test 
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE), 
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print(f"✓ Training set:   {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Validation set: {X_val.shape[0]:,} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Test set:       {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print()

# Scale
print("Step 6: Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("✓ Scaling complete")
print()

# Step 7: Feature Selection 
print("Step 7: Feature Selection (Selecting top 20 features ...")
selector = SelectKBest(f_classif, k=min(20, X_train_scaled.shape[1]))
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)
print(f"✓ Selected top {selector.get_support().sum()} features from {X_train_scaled.shape[1]} features")
print()

# Step 8: Class imbalance with scale_pos_weight
print("Step 8: Handling class imbalance")
print(f"  Training samples: {X_train_selected.shape[0]:,}")
normal_count = (y_train == 1).sum()  # 1 = Normal
attack_count = (y_train == 0).sum()  # 0 = Attack
scale_pos_weight = normal_count / attack_count if attack_count > 0 else 1
print(f"  Class distribution: Normal={normal_count:,}, Attack={attack_count:,}")
print(f"  scale_pos_weight: {scale_pos_weight:.4f} (XGBoost imbalance handling)")
print("✓ Using imbalanced data with scale_pos_weight")
print()

# Step 9: Train XGBoost
MODEL_SEEDS = [42, 43, 44, 45, 46]
print("Step 9: Training XGBoost ...")
if FIVE_RUNS:
    print(f"  Running 5 times with seeds {MODEL_SEEDS} (same split, same test set)")
else:
    print("Parameters: 200 trees, max_depth=2, stronger regularization")
print("Early stopping: Using validation set")
print()

accuracies_5run = []
start_time = time.time()
for run_idx, model_seed in enumerate(MODEL_SEEDS if FIVE_RUNS else [RANDOM_STATE]):
    if FIVE_RUNS:
        print(f"\n  --- Run {run_idx + 1}/5 (seed={model_seed}) ---")
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=2,                 
        learning_rate=0.08,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=12,          # Stricter
        gamma=0.6,                    # Stronger regularization
        reg_alpha=0.8,                # Stronger L1
        reg_lambda=6,                 # Stronger L2
        scale_pos_weight=scale_pos_weight,
        random_state=model_seed,
        n_jobs=-1,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0 if FIVE_RUNS else 1,
        early_stopping_rounds=15
    )
    xgb_model.fit(
        X_train_selected, 
        y_train,
        eval_set=[(X_val_selected, y_val)],  # Use VALIDATION set for early stopping
        verbose=FIVE_RUNS is False
    )
    test_acc = accuracy_score(y_test, xgb_model.predict(X_test_selected))
    if FIVE_RUNS:
        accuracies_5run.append(test_acc)
        print(f"  Test accuracy: {test_acc*100:.2f}%")

training_time = time.time() - start_time
if FIVE_RUNS:
    n_runs = len(accuracies_5run)
    mean_acc = np.mean(accuracies_5run) * 100
    std_acc = np.std(accuracies_5run, ddof=1) * 100
    t_val = scipy_stats.t.ppf(0.975, n_runs - 1)
    se = std_acc / np.sqrt(n_runs)
    ci_half = t_val * se
    ci_low = mean_acc - ci_half
    ci_high = mean_acc + ci_half
    print()
    print("="*70)
    print("XGB 5-RUN SUMMARY (same split, seeds 42-46)")
    print("="*70)
    print(f"  Test accuracy:  {mean_acc:.2f}% mean  (std = {std_acc:.2f}%)")
    print(f"  95% CI:         [{ci_low:.2f}%, {ci_high:.2f}%]")
    print("="*70)
    print()
else:
    print(f"\n✓ Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    if hasattr(xgb_model, 'best_iteration') and xgb_model.best_iteration is not None:
        print(f"✓ Early stopping: Best iteration at {xgb_model.best_iteration + 1} trees")
print()

# Check for overfitting
print("Step 9b: Checking for overfitting (Train vs Validation)...")
train_pred = xgb_model.predict(X_train_selected)
val_pred = xgb_model.predict(X_val_selected)
train_acc = accuracy_score(y_train, train_pred)
val_acc = accuracy_score(y_val, val_pred)
train_f1 = f1_score(y_train, train_pred, average='weighted', zero_division=0)
val_f1 = f1_score(y_val, val_pred, average='weighted', zero_division=0)

print(f"  Training Accuracy:   {train_acc*100:.2f}%")
print(f"  Validation Accuracy: {val_acc*100:.2f}%")
print(f"  Difference:          {abs(train_acc - val_acc)*100:.2f}%")
print(f"  Training F1:        {train_f1:.4f}")
print(f"  Validation F1:       {val_f1:.4f}")
print(f"  F1 Difference:       {abs(train_f1 - val_f1):.4f}")

if train_acc - val_acc > 0.05:  # More than 5% gap
    print("   WARNING: Possible overfitting detected (train >> validation)")
elif train_acc - val_acc > 0.02:  # More than 2% gap
    print("  CAUTION: Moderate gap between train and validation")
else:
    print("  No significant overfitting detected")
print()

# Step 10: Evaluate on TEST set 
print("Step 10: Evaluating XGBoost Model on TEST set...")
y_pred = xgb_model.predict(X_test_selected)
y_pred_proba = xgb_model.predict_proba(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)

print("="*70)
print("XGBOOST BINARY MODEL PERFORMANCE - CICIDS-2017 ")
print("="*70)
print(f"Overall Accuracy: {accuracy*100:.2f}%")
print()
print("Classification Report:")
report = classification_report(y_test, y_pred, target_names=target_encoder.classes_, digits=4)
print(report)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
cm_df = pd.DataFrame(cm, index=target_encoder.classes_, columns=target_encoder.classes_)
print(cm_df)
print()

tn, fp, fn, tp = cm.ravel()
print("Detailed Metrics:")
print(f"  True Negatives (Attack correctly detected):  {tn:,}")
print(f"  False Positives (Normal marked as Attack):   {fp:,}")
print(f"  False Negatives (Attack marked as Normal):   {fn:,}")
print(f"  True Positives (Normal correctly detected):  {tp:,}")
print()

# Step 11: Visualizations
print("Step 11: Creating visualizations...")

feature_names_all = list(X.columns)
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = [feature_names_all[i] for i in selected_feature_indices]

importances = xgb_model.feature_importances_
n_top = min(20, len(importances))
indices = np.argsort(importances)[::-1][:n_top]

plt.figure(figsize=(12, 8))
plt.bar(range(n_top), importances[indices], edgecolor='black', color='darkorange')
plt.xticks(range(n_top), [selected_feature_names[i] for i in indices], rotation=45, ha='right')
plt.title('Top 20 Features - CICIDS-2017 XGBoost', 
          fontsize=14, fontweight='bold')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Importance Score', fontsize=12)
plt.tight_layout()
plt.savefig('xgb_feature_importance_cicids2017_95percent.png', dpi=300, bbox_inches='tight')
print("✓ Saved: xgb_feature_importance_cicids2017_95percent.png")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_,
            cbar_kws={'label': 'Count'})
plt.title('XGBoost Binary Classification Confusion Matrix - CICIDS-2017 ', 
          fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('xgb_confusion_matrix_cicids2017_95percent.png', dpi=300, bbox_inches='tight')
print("✓ Saved: xgb_confusion_matrix_cicids2017_95percent.png")
print()

# Step 12: Save model
print("Step 12: Saving XGBoost model and preprocessors...")
joblib.dump(xgb_model, 'xgb_model_binary_cicids2017_95percent.pkl')
joblib.dump(scaler, 'scaler_xgb_cicids2017_95percent.pkl')
joblib.dump(selector, 'feature_selector_xgb_cicids2017_95percent.pkl')
joblib.dump(label_encoders, 'label_encoders_xgb_cicids2017_95percent.pkl')
joblib.dump(target_encoder, 'target_encoder_xgb_cicids2017_95percent.pkl')
joblib.dump(selected_feature_names, 'feature_names_xgb_cicids2017_95percent.pkl')

# Full test tensors =
np.save('X_test_xgb_cicids2017_95percent.npy', X_test_selected)
np.save('y_test_xgb_cicids2017_95percent.npy', y_test)
np.save('y_pred_xgb_cicids2017_95percent.npy', y_pred)
np.save('y_pred_proba_xgb_cicids2017_95percent.npy', y_pred_proba)

print("✓ Saved: xgb_model_binary_cicids2017_95percent.pkl")
print("✓ Saved: scaler_xgb_cicids2017_95percent.pkl")
print("✓ Saved: feature_selector_xgb_cicids2017_95percent.pkl")
print("✓ Saved: label_encoders_xgb_cicids2017_95percent.pkl")
print("✓ Saved: target_encoder_xgb_cicids2017_95percent.pkl")
print("✓ Saved: feature_names_xgb_cicids2017_95percent.pkl")
print(f"✓ Saved: Full test set ({len(y_test):,} rows) — X_test_xgb_cicids2017_95percent.npy")
print()

# Top features
print("Top 10 Most Important Features:")
for i in range(min(10, len(indices))):
    idx = indices[i]
    feat_name = selected_feature_names[idx]
    print(f"  {i+1:2}. {feat_name:35} → {importances[idx]:.4f}")
print()

# Final summary
print("="*70)
print("XGBOOST BINARY CLASSIFICATION COMPLETE - CICIDS-2017")
print("="*70)
print(f"Final Accuracy:  {accuracy*100:.2f}%")
print(f"Training Time:   {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
print()
print("✓ Fixed 70/15/15 split (Train/Val/Test)")
print("✓ Top 20 features, max_depth=2, stronger regularization")
print("="*70)

