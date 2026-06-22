# Streamlit Dashboard - Dual-Model Intrusion Detection System

## Overview
Interactive web dashboard for dual-model intrusion detection with XAI explanations.

**Models:**
- Random Forest (trained on NSL-KDD dataset)
- XGBoost (trained on CICIDS-2017 dataset)

**XAI Methods:**
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- ELI5 (Permutation-based Feature Importance)

## Prerequisites

All required packages should already be installed:
- streamlit
- joblib
- numpy
- pandas
- matplotlib
- seaborn
- shap
- lime
- eli5
- scikit-learn
- xgboost

## Required Files

Before running the dashboard, ensure these files exist:
1. `unified_xai_package.pkl` - Models and preprocessors (created by `unified_xai_dual_models.py`)
2. `rf_test_samples.npy` - Random Forest test samples
3. `xgb_test_samples.npy` - XGBoost test samples

## Running the Dashboard

1. **Open terminal/command prompt**
2. **Navigate to project directory:**
   ```bash
   cd "C:\Users\UPEIRCH\Documents\MSC\Research\Research Development"
   ```

3. **Run Streamlit:**
   ```bash
   streamlit run streamlit_dashboard.py
   ```

4. **Access the dashboard:**
   - The dashboard will automatically open in your default web browser
   - URL: `http://localhost:8501`
   - If it doesn't open automatically, copy the URL from the terminal

## Dashboard Features

### 1. **Model Loading**
   - Click "🔄 Load Models" button in the sidebar
   - Models and XAI explainers will be initialized
   - This may take 30-60 seconds on first load

### 2. **Input Methods**
   - **Use Sample Data:** Select from pre-loaded test samples
   - **Upload File:** Upload CSV file (preprocessing to be implemented)

### 3. **Predictions Display**
   - Side-by-side comparison of both models
   - Shows prediction (Attack/Normal) and confidence percentage
   - Displays probability for both classes

### 4. **XAI Explanations**
   For each model, you can view:
   - **SHAP:** Feature-level contributions with bar charts
   - **LIME:** Local explanations with feature conditions
   - **ELI5:** Global feature importance rankings

### 5. **Model Comparison**
   - Shows if both models agree or disagree
   - Displays confidence difference
   - Highlights when models have conflicting predictions

## Dashboard Layout

```
┌─────────────────────────────────────────────────┐
│  🛡️ Dual-Model Intrusion Detection System      │
├──────────────┬──────────────────────────────────┤
│  Sidebar     │  Main Content                    │
│              │                                   │
│  Load Models │  [Sample Selection]              │
│  Input Method│                                   │
│              │  ┌──────────┬──────────┐        │
│              │  │ RF Model │ XGB Model │        │
│              │  │ Prediction + XAI    │        │
│              │  └──────────┴──────────┘        │
│              │                                   │
│              │  [Comparison View]              │
└──────────────┴──────────────────────────────────┘
```

## Troubleshooting

### Issue: "Error loading models"
**Solution:** 
- Ensure `unified_xai_package.pkl` exists
- Run `unified_xai_dual_models.py` first to create the package

### Issue: "Error loading test samples"
**Solution:**
- Ensure `rf_test_samples.npy` and `xgb_test_samples.npy` exist
- These are created by `unified_xai_dual_models.py`

### Issue: "SHAP/LIME/ELI5 not working"
**Solution:**
- Wait for models to fully load (check sidebar for success message)
- XAI explainers are initialized after model loading
- Try reloading models if issues persist

### Issue: Dashboard is slow
**Solution:**
- First load takes time (30-60 seconds) for XAI initialization
- Subsequent interactions are faster (cached)
- ELI5 computation uses reduced iterations (5) for speed

## Next Steps

1. **File Upload Preprocessing:**
   - Implement preprocessing functions for user-uploaded CSV files
   - Map user data to model feature formats

2. **Manual Input:**
   - Add form inputs for manual feature entry
   - Validate and preprocess user inputs

3. **Enhanced Visualizations:**
   - Add SHAP waterfall plots
   - Add SHAP force plots
   - Add interactive Plotly charts

4. **Export Results:**
   - Add download button for predictions
   - Export XAI explanations as PDF/CSV

## Notes

- The dashboard uses caching for faster performance
- Models are loaded once and reused
- XAI explainers are initialized on first load
- Sample data is pre-loaded for quick testing

## Support

For issues or questions:
1. Check that all required files exist
2. Verify models were trained successfully
3. Ensure all dependencies are installed
4. Review error messages in the terminal

---

**Created for Research: Dual-Model Intrusion Detection with XAI**

