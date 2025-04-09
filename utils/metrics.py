import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    return {
        "R²": round(r2_score(y_true, y_pred), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "MSE": round(mean_squared_error(y_true, y_pred), 4)
    }

def plot_predictions(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.plot(y_true, label="Actual")
    ax.plot(y_pred, label="Predicted", linestyle="--")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    return fig

def plot_residuals(y_true, y_pred):
    residuals = np.array(y_true) - np.array(y_pred)
    fig, ax = plt.subplots()
    ax.hist(residuals, bins=30, alpha=0.7)
    ax.set_title("Residuals Histogram")
    return fig

def calculate_train_test_metrics(model, X_train, y_train, X_test, y_test):
    """Compute train and test metrics and return as a formatted DataFrame."""
    train_pred = model.predict(X_train).ravel()
    test_pred = model.predict(X_test).ravel()

    train_metrics = calculate_metrics(y_train, train_pred)
    test_metrics = calculate_metrics(y_test, test_pred)

    metrics_df = pd.DataFrame([train_metrics, test_metrics], index=["Train", "Test"])
    return metrics_df, train_pred, test_pred
