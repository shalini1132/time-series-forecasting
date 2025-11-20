# ⏳ Time Series Forecasting with LSTM & SHAP (Explainable AI)

This project focuses on forecasting energy consumption using a multivariate time-series dataset.  
The model is built using **LSTM (Long Short-Term Memory)** and the predictions are explained with **SHAP (SHapley Additive exPlanations)** for interpretability.

---

## 📌 Project Overview

Many real-world problems depend on time-based data like:
- Weather patterns  
- Electricity demand  
- Stock market  
- Sensor readings  

In this project:
- We forecast **Energy Consumption** using past values of:
  - Temperature  
  - Humidity  
  - Wind Speed  
  - Pressure  
  - Date (converted into features)

We also use SHAP to understand **which features influence the prediction** the most.

---

## 🗂 Dataset Description

Your dataset columns:

- `date`
- `temperature`
- `humidity`
- `wind_speed`
- `pressure`
- `energy_consumption` (Target)

Dataset is multivariate and supports deep learning models like LSTM.

---

## 🧪 Model Used: LSTM

We used LSTM because:
- It handles sequential data
- Remembers long-term patterns
- Works well for forecasting

### Steps performed:
1. Convert date column to datetime  
2. Create lag features  
3. Normalize values  
4. Split into train & test  
5. Build LSTM model  
6. Train and evaluate  
7. Predict future values  

---

## 💡 Explainable AI (SHAP)

We used SHAP to:
- Understand feature importance  
- Check which sensor/feature affects energy usage  
- Improve model transparency  

---

## 📁 Project Files

✔ `notebook.ipynb` – Full Colab code  
✔ `data.csv` – Dataset  
✔ `model.py` – Model building code (optional)  
✔ `plots/` – All saved graphs  
✔ `README.md` – Project documentation  

---

## 📊 Visualizations Included

- Energy consumption trend graph  
- Temperature vs consumption  
- Train vs Test prediction plot  
- Feature importance (SHAP summary plot)

---

## ⚙️ How to Run Locally

### 1. Install required libraries
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow shap
