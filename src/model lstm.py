# src/model_lstm.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation="tanh", input_shape=input_shape))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

def prepare_lstm_input(X):
    """
    X shape: (samples, features) -> reshape to (samples, 1, features)
    """
    return X.reshape(X.shape[0], 1, X.shape[1])

def train_lstm(df, features, target="energy_consumption",
               test_size=0.2, epochs=20, batch_size=32, save_dir="outputs"):
    """
    Train LSTM and save a basic actual vs predicted plot into outputs/.
    Returns model, scaler (if used externally), X_test_lstm, y_test, y_pred
    """
    # features: list of columns
    X = df[features].values
    y = df[target].values

    # train/test split (time series: no shuffle)
    n_test = int(len(X) * test_size)
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]

    X_train_lstm = prepare_lstm_input(X_train)
    X_test_lstm = prepare_lstm_input(X_test)

    model = build_lstm((X_train_lstm.shape[1], X_train_lstm.shape[2]))
    model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=2)

    # predictions
    y_pred = model.predict(X_test_lstm).flatten()

    # metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Test RMSE: {rmse:.4f}  MAE: {mae:.4f}")

    # save plot
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.plot(y_test, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "actual_vs_predicted.png"))
    plt.close()

    return model, X_test_lstm, y_test, y_pred
