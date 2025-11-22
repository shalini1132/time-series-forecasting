# src/utils.py
import pandas as pd
import os

def ensure_outputs():
    os.makedirs("outputs", exist_ok=True)

def save_df_sample(df, path="outputs/data_sample.csv", n=100):
    df.head(n).to_csv(path, index=False)
