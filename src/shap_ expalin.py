# src/shap_explain.py
import shap
import numpy as np
import matplotlib.pyplot as plt
import os

def explain_with_shap(model, X_train_lstm, X_test_lstm, feature_names, save_dir="outputs", background_samples=100):
    """
    X_train_lstm, X_test_lstm shapes: (samples, 1, features)
    Convert to 2D for plotting.
    Uses DeepExplainer with a background sample.
    """
    os.makedirs(save_dir, exist_ok=True)

    # reshape to 2D for shap plotting
    X_test_2d = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[2])
    X_train_2d = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[2])

    # background for explainer: pick up to background_samples from train
    background = X_train_2d[np.random.choice(X_train_2d.shape[0], min(background_samples, X_train_2d.shape[0]), replace=False)]

    # Use DeepExplainer (works better with TF models)
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(X_test_2d)

    # summary plot (dot)
    shap.summary_plot(shap_values, X_test_2d, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_summary.png"))
    plt.close()

    # bar plot (global importance)
    shap.summary_plot(shap_values, X_test_2d, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_bar.png"))
    plt.close()

    # force plot for first test record (saved as html)
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test_2d[0], feature_names=feature_names, matplotlib=False)
    shap.save_html(os.path.join(save_dir, "shap_force_plot.html"), force_plot)

    print(f"SHAP images saved to {save_dir}")
    return shap_values
