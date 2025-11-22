# run.py
from src.data_prep import load_data, create_lag_features, scale_features
from src.model_lstm import train_lstm, prepare_lstm_input
from src.shap_explain import explain_with_shap
import numpy as np
import os

def main():
    data_path = "data/energy_data.csv"
    print("Loading data...")
    df = load_data(data_path)

    print("Creating lag features...")
    df = create_lag_features(df, target="energy_consumption", lags=(1,2,3))

    # configure features used for model
    features = ["temperature","humidity","wind_speed","pressure","lag1","lag2","lag3"]

    # scale features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X = df[features].values
    X_scaled = scaler.fit_transform(X)

    # split manually to keep scaler consistent
    test_size = 0.2
    n_test = int(len(X_scaled) * test_size)
    X_train_scaled, X_test_scaled = X_scaled[:-n_test], X_scaled[-n_test:]
    y = df["energy_consumption"].values
    y_train, y_test = y[:-n_test], y[-n_test:]

    # reshape for LSTM
    X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

    print("Training LSTM...")
    model, X_test_for_model, y_test_out, y_pred = train_lstm(
        df=pd.concat([df.iloc[:-n_test].reset_index(drop=True), df.iloc[-n_test:].reset_index(drop=True)]),
        features=features,
        target="energy_consumption",
        test_size=test_size,
        epochs=20,
        batch_size=32,
        save_dir="outputs"
    )

    # NOTE: train_lstm returns model, X_test_lstm, y_test, y_pred (X_test_lstm corresponds to last n_test rows)
    # But for SHAP we need background X_train; rebuild X_train_lstm quickly:
    X_train_scaled = X_scaled[:-n_test]
    X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])

    print("Running SHAP explainability...")
    explain_with_shap(model, X_train_lstm, X_test_scaled.reshape(X_test_scaled.shape[0],1,X_test_scaled.shape[1]), feature_names=features, save_dir="outputs")

    print("Done. Check the outputs/ folder for plots and shap html.")

if __name__ == "__main__":
    main()
