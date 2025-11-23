# src/data_prep.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_data(path):
    """
    Load CSV, parse date, sort by date.
    """
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def create_lag_features(df, target="energy_consumption", lags=(1,2,3)):
    """
    Create lag features for the target column.
    """
    df_copy = df.copy()
    for l in lags:
        df_copy[f"lag{l}"] = df_copy[target].shift(l)
    df_copy = df_copy.dropna().reset_index(drop=True)
    return df_copy

def scale_features(X_train, X_test):
    """
    MinMax scale features; returns scaler, X_train_scaled, X_test_scaled.
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled
