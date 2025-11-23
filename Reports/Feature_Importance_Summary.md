#  Feature Importance Summary (SHAP Analysis)

This report summarizes the feature importance for the LSTM-based energy consumption forecasting model using SHAP (SHapley Additive exPlanations).

---

##  1. Global Feature Importance (SHAP Summary Plot)

The SHAP summary plot shows how much each input feature contributes to the model’s predictions across the entire test dataset.

### **Top Important Features**
1. **Lag_1** – Highest influence. Yesterday’s energy usage strongly affects tomorrow's usage.  
2. **Lag_2** – Moderate impact. Two-day lag helps capture short-term trends.  
3. **Temperature** – Higher temperature increases energy consumption (fans/AC load).  
4. **Humidity** – Mild effect due to weather-related variations.  
5. **Wind Speed & Pressure** – Least impact.

 **Plot:**  
![SHAP Summary Plot](../outputs/shap_summary_plot.png)

---

##  2. Local Explanation (SHAP Force Plot)

A SHAP force plot shows why the model made a specific prediction for one example (sample *i* from the test set).  
It highlights features pushing the prediction **higher (red)** or **lower (blue)**.

 **Plot:**  
![SHAP Force Plot](../outputs/force_plot_sample.png)

---

##  3. Interpretation Summary

- **Lag-based features dominate**, meaning past consumption trends are the strongest predictors.  
- **Weather features** (temperature, humidity) also contribute but less compared to lag features.  
- The **model is behaving logically**—following real-world energy usage patterns.  
- SHAP confirms that the LSTM model is not learning noise and is focusing on meaningful signals.

---

##  4. Files Used

- `outputs/shap_summary_plot.png`
- `outputs/force_plot_sample.png`
- Notebook: `Notebooks/LSTM_SHAP_Forecasting.ipynb`

---

## Conclusion

SHAP analysis provides clear and consistent explainability for the LSTM model.  
This feature importance summary helps validate that the model is learning correct patterns and making stable predictions.

