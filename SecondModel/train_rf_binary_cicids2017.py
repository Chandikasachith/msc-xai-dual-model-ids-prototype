"""
Random Forest Binary Classification for CICIDS-2017 (Normal vs Attack)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from collections import Counter
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import io
from pathlib import Path

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# Feature Engineering Function
def add_engineered_features(df):
    """Add domain-specific engineered features"""
    df = df.copy()
    
    # Data transfer patterns
    if 'Total Length of Fwd Packets' in df.columns and 'Total Length of Bwd Packets' in df.columns:
        df['total_bytes'] = df['Total Length of Fwd Packets'] + df['Total Length of Bwd Packets']
        df['bytes_ratio'] = df['Total Length of Fwd Packets'] / (df['Total Length of Bwd Packets'] + 1)
    
    if 'Total Fwd Packets' in df.columns and 'Total Bwd Packets' in df.columns:
        df['total_packets'] = df['Total Fwd Packets'] + df['Total Bwd Packets']
        df['packets_ratio'] = df['Total Fwd Packets'] / (df['Total Bwd Packets'] + 1)
    
    # Flow duration features
    if 'Flow Duration' in df.columns:
        if 'total_bytes' in df.columns:
            df['data_transfer_rate'] = df['total_bytes'] / (df['Flow Duration'] + 1)
        if 'total_packets' in df.columns:
            df['packets_per_second'] = df['total_packets'] / (df['Flow Duration'] + 1)
    
    # Connection patterns
    if 'Total Fwd Packets' in df.columns and 'Flow Duration' in df.columns:
        df['fwd_packets_rate'] = df['Total Fwd Packets'] / (df['Flow Duration'] + 1)
    
    if 'Total Bwd Packets' in df.columns and 'Flow Duration' in df.columns:
        df['bwd_packets_rate'] = df['Total Bwd Packets'] / (df['Flow Duration'] + 1)
    
    return df

print("="*70)
print("RANDOM FOREST BINARY INTRUSION DETECTION: CICIDS-2017")
print("Normal vs Attack Classification")
print("="*70)
print("\nImplementing:")
print("  ✓ Binary Classification")
print("  ✓ Feature Engineering (domain-specific features)")
print("  ✓ Class weights")
print("  ✓ Optimized Random Forest")
print("  ✓ Fixed 70/15/15 split (Train/Val/Test)")
print("="*70)
print()

# Step 1: Load all CICIDS-2017 parquet files
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
        print(f"    ⚠ File not found: {file}")

if not dataframes:
    print("ERROR: No data files found!")
    print("Please check that CICIDS-2017 data is in cicids-2017/ folder")
    exit(1)

# Combine all dataframes
print("\nCombining all datasets...")
combined_df = pd.concat(dataframes, ignore_index=True)
print(f"✓ Total samples before deduplication: {combined_df.shape[0]:,}")

# Remove duplicates
print("\nStep 1b: Removing duplicate samples...")
initial_count = len(combined_df)
combined_df = combined_df.drop_duplicates()
duplicates_removed = initial_count - len(combined_df)
print(f"✓ Removed {duplicates_removed:,} duplicate rows ({duplicates_removed/initial_count*100:.2f}%)")
print(f"✓ Total samples after deduplication: {combined_df.shape[0]:,}")
print()

# Step 2: Binary mapping (Normal vs Attack)
print("Step 2: Converting to Binary Classification...")
if 'Label' in combined_df.columns:
    combined_df['Label'] = combined_df['Label'].apply(
        lambda x: 'Normal' if str(x).lower() in ['benign', 'normal'] else 'Attack'
    )

print("✓ Binary class distribution:")
for cls, count in combined_df['Label'].value_counts().items():
    print(f"  {cls:10} → {count:8,} ({count/len(combined_df)*100:.1f}%)")
print()

# Step 3: Feature Engineering
print("Step 3: Adding Engineered Features...")
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

# Handle infinite and NaN values
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# Encode categorical features
categorical_features = []
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
        categorical_features.append(col)

label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature].astype(str))
    label_encoders[feature] = le

# Encode target (0 = Attack, 1 = Normal)
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
print(f"✓ Classes: {list(target_encoder.classes_)}")

# Fixed 70/15/15 split (Train/Val/Test)
print("\nStep 5: Creating fixed 70/15/15 split (Train/Val/Test)...")
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# First split: 70% train, 30% temp (which will become 15% val + 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, 
    test_size=(VAL_SIZE + TEST_SIZE),  # 30% for val+test
    random_state=RANDOM_STATE, 
    stratify=y_encoded
)

# Second split: Split temp into 15% val and 15% test (50/50 of the 30%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),  # 15% / 30% = 0.5
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print(f"✓ Training set:   {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Validation set: {X_val.shape[0]:,} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Test set:       {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print()

# Scale features
print("Step 6: Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("✓ Scaling complete (fit on train, transform on val/test)")
print()

# Step 7: Feature Selection (Optional - can use all features)
print("Step 7: Feature Selection...")
print("  Using SelectKBest to select top features...")
# Use top 50 features for Random Forest (can adjust)
selector = SelectKBest(f_classif, k=min(50, X_train_scaled.shape[1]))
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

feature_names_all = list(X.columns)
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = [feature_names_all[i] for i in selected_feature_indices]

print(f"✓ Selected top {len(selected_feature_names)} features from {X_train_scaled.shape[1]} features")
print()

# Step 8: Use class weights (NO SMOTE - consistent with XGBoost)
print("Step 8: Using class weights (NO SMOTE - consistent with XGBoost)...")
print(f"  Training samples: {X_train_selected.shape[0]:,}")
# Calculate class weights
class_counts = Counter(y_train)
total = sum(class_counts.values())
class_weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
print(f"  Class distribution: {dict(class_counts)}")
print(f"  Class weights: {class_weights}")
print("✓ Using imbalanced data with class weights (consistent with XGBoost)")
print()

# Use original training data (not balanced)
X_train_balanced = X_train_selected
y_train_balanced = y_train

# Step 9: Train Random Forest
print("Step 9: Training Random Forest...")
print("Parameters: 200 trees, max_depth=6 (EXACTLY matching XGBoost for fair comparison)")
print("Using class weights (consistent with XGBoost approach)")
print("NOTE: Matching XGBoost depth=6 to ensure fair comparison and avoid overfitting")
print()

rf_model = RandomForestClassifier(
    n_estimators=200,          # Match XGBoost tree count
    max_depth=6,               # Match XGBoost depth exactly for fair comparison
    min_samples_split=20,     # Increased for more regularization
    min_samples_leaf=10,       # Increased for more regularization
    max_features='sqrt',      # Use sqrt of features
    max_samples=0.7,          # More aggressive subsampling for regularization
    class_weight=class_weights,  # Handle class imbalance
    random_state=42,
    n_jobs=-1,
    verbose=0
)

start_time = time.time()
rf_model.fit(X_train_balanced, y_train_balanced)
training_time = time.time() - start_time
print(f"\n✓ Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
print()

# Check for overfitting: Compare training vs validation performance
print("Step 9b: Checking for overfitting (Train vs Validation)...")
train_pred = rf_model.predict(X_train_balanced)
val_pred = rf_model.predict(X_val_selected)
train_acc = accuracy_score(y_train_balanced, train_pred)
val_acc = accuracy_score(y_val, val_pred)
train_f1 = f1_score(y_train_balanced, train_pred, average='weighted', zero_division=0)
val_f1 = f1_score(y_val, val_pred, average='weighted', zero_division=0)

print(f"  Training Accuracy:   {train_acc*100:.2f}%")
print(f"  Validation Accuracy: {val_acc*100:.2f}%")
print(f"  Difference:          {abs(train_acc - val_acc)*100:.2f}%")
print(f"  Training F1:         {train_f1:.4f}")
print(f"  Validation F1:       {val_f1:.4f}")
print(f"  F1 Difference:       {abs(train_f1 - val_f1):.4f}")

if train_acc - val_acc > 0.05:  # More than 5% gap
    print("  ⚠ WARNING: Possible overfitting detected (train >> validation)")
elif train_acc - val_acc > 0.02:  # More than 2% gap
    print("  ⚠ CAUTION: Moderate gap between train and validation")
else:
    print("  ✓ No significant overfitting detected")
print()

# Step 10: Evaluate on TEST set
print("Step 10: Evaluating Random Forest Model on TEST set...")
print("(Test set was never used during training)")
y_pred = rf_model.predict(X_test_selected)
y_pred_proba = rf_model.predict_proba(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)

print("="*70)
print("RANDOM FOREST BINARY MODEL PERFORMANCE - CICIDS-2017")
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

# Calculate Attack-specific metrics
attack_precision = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
attack_recall = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
attack_f1 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

print("Attack Class Metrics (pos_label=0):")
print(f"  Precision: {attack_precision*100:.2f}%")
print(f"  Recall:    {attack_recall*100:.2f}%")
print(f"  F1-Score:  {attack_f1:.4f}")
print()

# Step 11: Visualizations
print("Step 11: Creating visualizations...")

# Feature importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:min(20, len(importances))]

plt.figure(figsize=(12, 8))
plt.bar(range(len(indices)), importances[indices], edgecolor='black', color='darkgreen')
plt.xticks(range(len(indices)), [selected_feature_names[i] for i in indices], rotation=45, ha='right')
plt.title('Top 20 Features - CICIDS-2017 Random Forest', 
          fontsize=14, fontweight='bold')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Importance Score', fontsize=12)
plt.tight_layout()
plt.savefig('rf_feature_importance_cicids2017.png', dpi=300, bbox_inches='tight')
print("✓ Saved: rf_feature_importance_cicids2017.png")

# Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_,
            cbar_kws={'label': 'Count'})
plt.title('Random Forest Binary Classification Confusion Matrix - CICIDS-2017', 
          fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('rf_confusion_matrix_cicids2017.png', dpi=300, bbox_inches='tight')
print("✓ Saved: rf_confusion_matrix_cicids2017.png")
print()

# Step 12: Save model
print("Step 12: Saving Random Forest model and preprocessors...")
joblib.dump(rf_model, 'rf_model_binary_cicids2017.pkl')
joblib.dump(scaler, 'scaler_rf_cicids2017.pkl')
joblib.dump(selector, 'feature_selector_rf_cicids2017.pkl')
joblib.dump(label_encoders, 'label_encoders_rf_cicids2017.pkl')
joblib.dump(target_encoder, 'target_encoder_rf_cicids2017.pkl')
joblib.dump(selected_feature_names, 'feature_names_rf_cicids2017.pkl')

# Full test tensors (for thesis metrics / generate_cicids2017_dual_model_evidence.py).
# Dashboard uses the first rows only for LIME/SHAP (see streamlit_dashboard.py).
np.save('X_test_rf_cicids2017.npy', X_test_selected)
np.save('y_test_rf_cicids2017.npy', y_test)
np.save('y_pred_rf_cicids2017.npy', y_pred)
np.save('y_pred_proba_rf_cicids2017.npy', y_pred_proba)

print("✓ Saved: rf_model_binary_cicids2017.pkl")
print("✓ Saved: scaler_rf_cicids2017.pkl")
print("✓ Saved: feature_selector_rf_cicids2017.pkl")
print("✓ Saved: label_encoders_rf_cicids2017.pkl")
print("✓ Saved: target_encoder_rf_cicids2017.pkl")
print("✓ Saved: feature_names_rf_cicids2017.pkl")
print(f"✓ Saved: Full test set ({len(y_test):,} rows) — X_test_rf_cicids2017.npy / y_test_rf_cicids2017.npy")
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
print("RANDOM FOREST BINARY CLASSIFICATION COMPLETE - CICIDS-2017!")
print("="*70)
print(f"Final Accuracy:  {accuracy*100:.2f}%")
print(f"Training Time:   {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
print()
print("✓ Fixed 70/15/15 split (Train/Val/Test)")
print("✓ Feature engineering applied")
print("✓ Class weights (NO SMOTE - consistent with XGBoost)")
print("✓ Top 50 features selected")
print("✓ Random Forest with 200 trees, max_depth=6 (matching XGBoost)")
print("✓ Test set untouched until final evaluation")
print("✓ Ready for comparison with XGBoost model")
print("="*70)

