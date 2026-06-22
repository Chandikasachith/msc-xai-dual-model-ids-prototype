# Fixed Splits Methodology

## Rule: One Model → One Dataset → One Split → One Metric Definition

This document explains the fixed split methodology implemented to ensure academic defensibility and consistency across all metrics and visualizations.

## Overview

The new script `generate_training_history_fixed_splits.py` implements a strict data splitting methodology:

- **NSL-KDD**: Fixed 70/15/15 split (Train/Val/Test)
- **CICIDS-2017**: Fixed 70/15/15 split (Train/Val/Test)

## Key Principles

### 1. Fixed Splits
- All splits use `random_state=42` for reproducibility
- Split ratios are exactly 70% train, 15% validation, 15% test
- Same split is used throughout the entire analysis

### 2. Test Set Usage
The **same TEST set** is used to compute:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

### 3. Validation Set Usage
The **same VALIDATION set** is used to produce:
- Tuning curves (n_estimators vs metrics)
- All validation metrics during hyperparameter tuning

## Implementation Details

### NSL-KDD (Random Forest)
1. Load full training data from `Dataset_NSL/KDDTrain+.txt`
2. Apply feature engineering (same as training script)
3. Apply preprocessing (using saved preprocessors)
4. Create fixed 70/15/15 split:
   - First split: 70% train, 30% temp
   - Second split: 30% temp → 15% val + 15% test
5. Generate tuning curves using validation set
6. Train final model and compute all metrics on test set

### CICIDS-2017 (XGBoost)
1. Load all parquet files and combine
2. Remove duplicates
3. Apply preprocessing (using saved preprocessors)
4. Create fixed 70/15/15 split (same method as NSL-KDD)
5. Generate tuning curves using validation set
6. Train final model and compute all metrics on test set

## Output Files

The script generates:
- `TRAINING_HISTORY_RandomForest_FIXED_SPLIT.png`: Random Forest tuning curves and test metrics
- `TRAINING_HISTORY_XGBoost_FIXED_SPLIT.png`: XGBoost tuning curves and test metrics

## Benefits

1. **Academic Defensibility**: One consistent split per dataset ensures all metrics are comparable
2. **Reproducibility**: Fixed random_state ensures same splits every time
3. **Consistency**: Same test set for all metrics, same validation set for all curves
4. **Clarity**: Slide titles and curves match because they use the same data splits

## Usage

```bash
python generate_training_history_fixed_splits.py
```

## Results Summary

### Random Forest (NSL-KDD)
- **Test Accuracy**: 99.90%
- **Test Precision**: 0.9990
- **Test Recall**: 0.9990
- **Test F1**: 0.9990
- **Test ROC-AUC**: 1.0000
- **Test PR-AUC**: 1.0000

### XGBoost (CICIDS-2017)
- **Test Accuracy**: 96.90%
- **Test Precision**: 0.9700
- **Test Recall**: 0.9690
- **Test F1**: 0.9675
- **Test ROC-AUC**: 0.9905
- **Test PR-AUC**: 0.9979

## Notes

- All splits are stratified to maintain class distribution
- Preprocessing uses saved preprocessors from original training scripts
- Feature engineering matches the original training scripts
- The validation set is used ONLY for tuning curves
- The test set is used ONLY for final metrics (never touched during training)



