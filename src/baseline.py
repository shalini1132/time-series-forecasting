"""
Baseline forecasting model using Naive method.
Baseline = predict next value as the previous value.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def baseline_forecast(y_test):
    """
    Computes baseline MAE and RMSE using naive forecasting.
    y_test: numpy array or list of test actual values.
    """

    y_true = y_test[1:]       # actual future values
    y_pred = y_test[:-1]      # previous actual as prediction

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return mae, rmse
