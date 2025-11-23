# Model Report — Interpretable Time-Series Forecasting (LSTM + SHAP)

## 1. Project summary
This project trains an LSTM model to forecast **energy_consumption** using exogenous features (temperature, humidity, wind_speed, pressure) and lag features. The model predictions are interpreted using SHAP to identify the top drivers of forecast changes.

---

## 2. Data preparation
- **Source file:** `data/time_series_dataset.csv`
- **Datetime parsing & sorting:** `date` column converted to datetime and sorted ascending.
- **Missing values:** Rows with missing values for essential features/target were dropped (or imputed — mention if you imputed).
- **Feature engineering:**
  - Lag features: `lag1`, `lag2`, `lag3` (previous 1,2,3 time steps of target).
  - (Optional) Time features: `dayofweek`, `hour` if data is hourly.
- **Scaling:** Features scaled using `MinMaxScaler` prior to LSTM input (fit on train set, transform on test set).
- **Train/test split:** Time-based split (no shuffle). Typical split: 80% train, 20% test.

---

## 3. Baseline model (implemented)
A simple baseline using **Exponential Smoothing** (Holt-Winters without seasonality) was implemented in `src/baseline.py`. Baseline metrics:
- Baseline RMSE: `<baseline_rmse>`  
- Baseline MAE: `<baseline_mae>`

(Replace angle-bracket values after running `run.py`.)

Rationale: baseline provides a simple, interpretable reference to judge LSTM performance.

---

## 4. LSTM model design and training
- **Architecture:**
  - Input shape: (timesteps=1, features=7) — features = [temperature, humidity, wind_speed, pressure, lag1, lag2, lag3]
  - LSTM(64, activation='tanh')
  - Dense(32, activation='relu')
  - Dense(1) output
- **Loss / optimizer:** MSE loss, Adam optimizer.
- **Training:** `epochs=20`, `batch_size=32` (adjustable), `validation_split=0.1`.
- **Performance (example):**
  - LSTM RMSE: `<lstm_rmse>`  
  - LSTM MAE: `<lstm_mae>`

(Replace with your run values.)

---

## 5. Comparison: LSTM vs Baseline
- Present a small table comparing RMSE and MAE for baseline vs LSTM (copy values from run).
- Short interpretation: if LSTM RMSE < baseline RMSE → LSTM captures nonlinear/lag effects; else investigate overfitting or feature issues.

---

## 6. SHAP analysis (XAI)
- **Explainer used:** `shap.KernelExplainer` (or `DeepExplainer` where compatible).  
- **Background dataset:** sample from training data (e.g., 50 rows).
- **Global explanation:** SHAP summary plot & bar chart showing global feature importance.
- **Local explanation:** SHAP force plot for selected test instances to show positive/negative contributions.
- **Top 5 features (example):**
  1. `lag1` — largest positive influence when high (suggesting strong autocorrelation).
  2. `temperature` — moderate influence; spikes increase consumption.
  3. `lag2`
  4. `humidity`
  5. `wind_speed`
- **Concrete example:** For test instance on `<date>`, model predicted `<pred>`; SHAP shows `lag1 (+40)`, `temperature (+12)`, `humidity (-5)` contributions (numbers illustrative — replace with real SHAP values).

---

## 7. Stability and intrinsic importance check
- **Intrinsic check:** Compare SHAP ranks with LSTM dense-layer weights or permutation importance (if implemented).
- **Finding:** If SHAP top features align with intrinsic measures → model explanations stable. If they differ, consider:
  - Data leakage, scaling inconsistency, or model capturing complex interactions not reflected in intrinsic weights.

---

## 8. Limitations & next steps
- Baseline could be improved (ARIMA/seasonal ETS) for better comparison.
- Use longer background dataset for SHAP to reduce variance.
- Consider an attention-based LSTM or Transformer for better temporal interpretability.
- Provide a formal text summary of top-5 features with specific examples (see `reports/Feature_Importance_Summary.md`).

---

## 9. How to reproduce
1. Install requirements:
```bash
pip install -r requirements.txt
