# Time-Series Forecasting with LSTM and SHAP

## Project Overview
This project demonstrates **time-series forecasting** using an **LSTM (Long Short-Term Memory)** model and explains the model predictions using **SHAP (SHapley Additive exPlanations)**.  
The main goal is to predict energy consumption based on weather and historical data and provide interpretable explanations for the model’s predictions.

## Folder Structure

time-series-forecasting/
├── notebooks/
│   └── LSTM_SHAP_Forecasting.ipynb
├── src/
│   ├── data_prep.py
│   ├── model_lstm.py
│   ├── shap_explain.py
│   └── utils.py (optional)
├── data/
│   └── energy_data.csv
├── outputs/
├── requirements.txt
├── README.md
└── run.py

## How to Run
1. Install dependencies: `pip install -r requirements.txt`  
2. Open `notebooks/LSTM_SHAP_Forecasting.ipynb` in Colab or Jupyter  
3. Run all cells. Plots are saved automatically in `outputs/` folder.

## Outputs
- `outputs/actual_vs_predicted.png`  
- `outputs/shap_summary.png`  
- `outputs/shap_force_plot.html`

## Libraries
- numpy, pandas, matplotlib, seaborn  
- tensorflow, keras, shap
