# Research Project Overview: Dual-Model Intrusion Detection System

## 🎯 Project Goal
Build a **dual-model intrusion detection system** that uses two different machine learning models trained on different datasets to detect network attacks. The system includes **Explainable AI (XAI)** to help understand why predictions are made.

---

## 🏗️ System Architecture

### **Two-Model Approach**

Your system uses **two separate models** that work together:

1. **Random Forest Model** (NSL-KDD dataset)
   - Trained on NSL-KDD dataset
   - Binary classification: Normal vs Attack
   - Accuracy: ~75.44%
   - Location: Main directory

2. **XGBoost Model** (CICIDS-2017 dataset)
   - Trained on CICIDS-2017 dataset
   - Binary classification: Normal vs Attack
   - Location: `SecondModel/` folder

### **Why Two Models?**
- Different models trained on different datasets provide complementary perspectives
- If one model misses something, the other might catch it
- Better overall detection reliability

---

## 📊 Model 1: Random Forest (NSL-KDD)

### **Dataset: NSL-KDD**
- Classic intrusion detection dataset
- Contains network traffic features
- Multiple attack types: DoS, Probe, R2L, U2R, Normal

### **What Was Done:**
1. **Feature Engineering** - Added 10 domain-specific features:
   - `data_transfer_rate` - How fast data is transferred
   - `total_bytes` - Total data volume
   - `bytes_per_connection` - Data per connection
   - And 7 more...

2. **SMOTE Balancing** - Balanced the training data:
   - Original: 125,973 samples (imbalanced)
   - After SMOTE: 336,715 samples (balanced)

3. **Model Optimization**:
   - Trees: 400 (increased from 100)
   - Max Depth: 35 (increased from 20)
   - Result: **75.44% accuracy** (improved from 74.72%)

### **Key Files:**
- `train_nslkdd_rf_final.py` - Training script (USE THIS ONE)
- `rf_model_nslkdd_final.pkl` - Trained model
- `scaler_nslkdd_final.pkl` - Feature scaler
- `label_encoders_nslkdd_final.pkl` - Categorical encoders
- `target_encoder_nslkdd_final.pkl` - Class encoder

### **Performance:**
- **Accuracy:** 75.44%
- **Best at:** DoS attacks (95.94% precision), Normal traffic (97.12% recall)
- **Challenging:** R2L and U2R attacks (rare in training data)

---

## 📊 Model 2: XGBoost (CICIDS-2017)

### **Dataset: CICIDS-2017**
- Modern, comprehensive intrusion detection dataset
- Contains 8 different attack types:
  - Benign (Normal)
  - Botnet
  - Bruteforce
  - DDoS
  - DoS
  - Infiltration
  - Portscan
  - WebAttacks

### **What Was Done:**
1. **Data Loading** - Loads 8 parquet files from `cicids-2017/` folder
2. **Binary Classification** - Maps all attacks to "Attack", Benign to "Normal"
3. **Feature Engineering** - Similar to RF model:
   - Data transfer patterns
   - Connection patterns
   - Security indicators
4. **SMOTE Balancing** - Balances training data
5. **XGBoost Training**:
   - 400 estimators
   - Max depth: 6
   - Learning rate: 0.1

### **Key Files:**
- `SecondModel/train_xgb_binary_cicids2017.py` - Main training script
- `SecondModel/xgb_model_binary_cicids2017.pkl` - Trained model
- `SecondModel/scaler_xgb_cicids2017.pkl` - Feature scaler
- Multiple variants (95percent, conservative, realistic) for different scenarios

### **Performance:**
- Model trained and saved
- Ready for evaluation and integration

---

## 🔍 Explainable AI (XAI) Components

### **Three XAI Methods Implemented:**

1. **SHAP (SHapley Additive exPlanations)**
   - Shows how much each feature contributes to a prediction
   - Provides feature importance scores
   - Visualizations: bar charts, summary plots

2. **LIME (Local Interpretable Model-agnostic Explanations)**
   - Explains individual predictions locally
   - Shows which features pushed the prediction toward Attack or Normal
   - Creates interpretable explanations

3. **ELI5 (Permutation-based Feature Importance)**
   - Global feature importance rankings
   - Shows which features matter most overall
   - Permutation-based method

### **XAI Files:**
- `unified_xai_dual_models.py` - Creates XAI explanations for both models
- `unified_xai_package.pkl` - Saved package with models and explainers
- `binary_xai_analysis.py` - XAI analysis scripts
- Various visualization outputs (PNG files)

---

## 🖥️ Interactive Dashboard

### **Streamlit Web Dashboard**

A user-friendly web interface to:
- Load both models
- Make predictions on sample data
- View XAI explanations (SHAP, LIME, ELI5)
- Compare predictions from both models
- See when models agree/disagree

### **How to Run:**
```bash
streamlit run streamlit_dashboard.py
```

### **Features:**
- Side-by-side model comparison
- Real-time predictions
- Interactive XAI visualizations
- Sample data selection
- File upload capability (preprocessing needed)

### **Dashboard Files:**
- `streamlit_dashboard.py` - Main dashboard code
- `DASHBOARD_README.md` - Dashboard documentation

---

## 📁 Project Structure

```
Research Development/
│
├── Main Directory (Random Forest - NSL-KDD)
│   ├── train_nslkdd_rf_final.py          # ⭐ Main RF training script
│   ├── rf_model_nslkdd_final.pkl         # Trained RF model
│   ├── scaler_nslkdd_final.pkl           # Feature scaler
│   ├── label_encoders_nslkdd_final.pkl   # Encoders
│   ├── target_encoder_nslkdd_final.pkl   # Class encoder
│   │
│   ├── streamlit_dashboard.py            # Web dashboard
│   ├── unified_xai_dual_models.py        # XAI system
│   ├── unified_xai_package.pkl          # Saved XAI package
│   │
│   ├── COMPLETION_REPORT.txt            # RF improvement report
│   ├── IMPROVEMENT_SUMMARY.txt          # Detailed analysis
│   └── DASHBOARD_README.md               # Dashboard guide
│
└── SecondModel/ (XGBoost - CICIDS-2017)
    ├── train_xgb_binary_cicids2017.py   # ⭐ Main XGB training script
    ├── xgb_model_binary_cicids2017.pkl   # Trained XGB model
    ├── scaler_xgb_cicids2017.pkl        # Feature scaler
    ├── label_encoders_xgb_cicids2017.pkl # Encoders
    │
    ├── cicids-2017/                      # Dataset folder
    │   ├── Benign-Monday-no-metadata.parquet
    │   ├── Botnet-Friday-no-metadata.parquet
    │   ├── Bruteforce-Tuesday-no-metadata.parquet
    │   ├── DDoS-Friday-no-metadata.parquet
    │   ├── DoS-Wednesday-no-metadata.parquet
    │   ├── Infiltration-Thursday-no-metadata.parquet
    │   ├── Portscan-Friday-no-metadata.parquet
    │   └── WebAttacks-Thursday-no-metadata.parquet
    │
    └── [Multiple model variants and outputs]
```

---

## 🎓 Research Components

### **1. Machine Learning Models**
- ✅ Random Forest (NSL-KDD) - Optimized to 75.44%
- ✅ XGBoost (CICIDS-2017) - Trained and ready

### **2. Feature Engineering**
- ✅ Domain-specific features for both datasets
- ✅ 10 engineered features for NSL-KDD
- ✅ Similar engineering for CICIDS-2017

### **3. Data Balancing**
- ✅ SMOTE oversampling implemented
- ✅ Handles class imbalance

### **4. Explainable AI**
- ✅ SHAP explanations
- ✅ LIME explanations
- ✅ ELI5 feature importance

### **5. User Interface**
- ✅ Streamlit dashboard
- ✅ Interactive visualizations
- ✅ Model comparison

---

## 📈 Current Status

### **✅ Completed:**
1. Random Forest model trained and optimized (75.44% accuracy)
2. XGBoost model trained on CICIDS-2017
3. Feature engineering implemented for both
4. SMOTE balancing implemented
5. XAI system (SHAP, LIME, ELI5) integrated
6. Streamlit dashboard created
7. Model evaluation and visualization

### **🔄 In Progress / Future Work:**
1. Dashboard file upload preprocessing
2. Manual feature input form
3. Enhanced visualizations (waterfall plots, force plots)
4. Export functionality for results
5. Further model optimization if needed

---

## 🚀 How to Use

### **1. Train/Retrain Models:**

**Random Forest (NSL-KDD):**
```bash
python train_nslkdd_rf_final.py
```

**XGBoost (CICIDS-2017):**
```bash
cd SecondModel
python train_xgb_binary_cicids2017.py
```

### **2. Generate XAI Explanations:**
```bash
python unified_xai_dual_models.py
```

### **3. Run Dashboard:**
```bash
streamlit run streamlit_dashboard.py
```

### **4. Make Predictions (Python):**
```python
import joblib

# Load RF model
rf_model = joblib.load('rf_model_nslkdd_final.pkl')
scaler = joblib.load('scaler_nslkdd_final.pkl')

# Preprocess and predict
X_scaled = scaler.transform(X_new)
predictions = rf_model.predict(X_scaled)
```

---

## 📝 Key Findings

### **Random Forest (NSL-KDD):**
- Baseline: 74.72% → Final: 75.44% (+0.72%)
- 6 out of top 10 features are engineered
- Best performance on DoS and Normal traffic
- R2L and U2R remain challenging (rare attacks)

### **Feature Engineering Impact:**
- Top features include: `data_transfer_rate`, `total_bytes`, `bytes_per_connection`
- Engineered features significantly contribute to model performance

### **Dual-Model Benefits:**
- Two perspectives on the same problem
- Can catch different patterns
- More robust overall system

---

## 🔧 Technical Details

### **Technologies Used:**
- Python 3.x
- scikit-learn (Random Forest)
- XGBoost
- SHAP, LIME, ELI5 (XAI)
- Streamlit (Dashboard)
- pandas, numpy (Data processing)
- matplotlib, seaborn (Visualization)
- imbalanced-learn (SMOTE)

### **Model Parameters:**

**Random Forest:**
- n_estimators: 400
- max_depth: 35
- class_weight: balanced

**XGBoost:**
- n_estimators: 400
- max_depth: 6
- learning_rate: 0.1

---

## 📚 Documentation Files

- `COMPLETION_REPORT.txt` - Random Forest improvement summary
- `IMPROVEMENT_SUMMARY.txt` - Detailed RF analysis
- `DASHBOARD_README.md` - Dashboard usage guide
- `PROJECT_OVERVIEW.md` - This file

---

## ❓ Common Questions

**Q: Which model should I use?**
A: Use both! They complement each other. RF for NSL-KDD patterns, XGBoost for CICIDS-2017 patterns.

**Q: How do I improve accuracy further?**
A: Options include:
- Hyperparameter tuning (RandomizedSearchCV)
- Threshold optimization
- Ensemble methods
- More feature engineering

**Q: What if models disagree?**
A: This is valuable! It shows uncertainty. You can:
- Use voting/consensus
- Investigate with XAI
- Flag for manual review

**Q: Can I add more models?**
A: Yes! The system is designed to be extensible. Add models to `unified_xai_dual_models.py` and dashboard.

---

## 🎯 Next Steps for Research

1. **Evaluate XGBoost performance** - Get accuracy metrics
2. **Compare both models** - Which performs better on what?
3. **Ensemble approach** - Combine predictions from both models
4. **Real-world testing** - Test on new, unseen data
5. **Paper writing** - Document methodology and results

---

## 📞 Need Help?

If you need to understand or modify any component:
1. Check the relevant training script (comments explain each step)
2. Review the completion reports
3. Look at the dashboard code for usage examples
4. Check XAI scripts for explanation generation

---

**Last Updated:** Based on current project state
**Status:** ✅ Core components complete, ready for research and evaluation





