"""
python/src/evaluation/metrics.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def smape(y_true, y_pred, eps=1e-8):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)
    ) * 100)


def evaluate(y_true, y_pred, model: str, hotel_id: str = None) -> dict:
    return {
        "hotel_id": hotel_id,
        "model":    model,
        "MAE":      float(mean_absolute_error(y_true, y_pred)),
        "RMSE":     float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "sMAPE":    smape(y_true, y_pred),
    }


def summarize(records: list) -> pd.DataFrame:
    df = pd.DataFrame(records).dropna(subset=["MAE"])
    return (df.groupby("model")
            .agg(MAE_mean=("MAE","mean"), MAE_std=("MAE","std"),
                 RMSE_mean=("RMSE","mean"), sMAPE_mean=("sMAPE","mean"),
                 n_hotels=("hotel_id","nunique"))
            .round(4).sort_values("MAE_mean"))
