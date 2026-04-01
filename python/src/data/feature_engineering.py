"""
python/src/data/feature_engineering.py
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.config import LAG_DAYS, ROLL_WINDOWS, WEATHER_COLS, CALENDAR_COLS, TIME_COLS

_encoders: dict = {}


def fit_encoders(df: pd.DataFrame) -> dict:
    global _encoders
    for col in ["hotel_id", "brand_tier", "hotel_district", "district_functional_tier"]:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        _encoders[col] = le
    return _encoders


def get_encoders() -> dict:
    return _encoders


def apply_encoders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, le in _encoders.items():
        known = set(le.classes_)
        df[col + "_enc"] = df[col].astype(str).apply(
            lambda x: int(le.transform([x])[0]) if x in known else -1
        )
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"]  = df["record_date"].dt.dayofweek
    df["month"]        = df["record_date"].dt.month
    df["day_of_month"] = df["record_date"].dt.day
    df["quarter"]      = df["record_date"].dt.quarter
    df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype(int)
    df["month_sin"]    = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]    = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]      = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]      = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CALENDAR_COLS:
        df[c] = df[c].fillna(0).astype(int) if c in df.columns else 0
    return df


def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["hotel_id", "record_date"])
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df.groupby("hotel_id")["occupancy_rate"].shift(lag)
    for w in ROLL_WINDOWS:
        df[f"rolling_mean_{w}"] = df.groupby("hotel_id")["occupancy_rate"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        df[f"rolling_std_{w}"]  = df.groupby("hotel_id")["occupancy_rate"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).std())
        df[f"rolling_max_{w}"]  = df.groupby("hotel_id")["occupancy_rate"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).max())
        df[f"rolling_min_{w}"]  = df.groupby("hotel_id")["occupancy_rate"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).min())

    all_lag_roll = (
        [f"lag_{l}" for l in LAG_DAYS]
        + [f"rolling_{t}_{w}" for t in ["mean","std","max","min"] for w in ROLL_WINDOWS]
    )
    for col in all_lag_roll:
        df[col] = df.groupby("hotel_id")[col].transform(lambda x: x.ffill().bfill())
        df[col] = df[col].fillna(df[col].median())
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in WEATHER_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0 if col in ["prcp","snow"] else df[col].median())
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    lag_cols  = [f"lag_{l}" for l in LAG_DAYS]
    roll_cols = [f"rolling_{t}_{w}"
                 for t in ["mean","std","max","min"] for w in ROLL_WINDOWS]
    time_feats = TIME_COLS + ["month_sin","month_cos","dow_sin","dow_cos"]
    cal_feats  = [c for c in CALENDAR_COLS if c in df.columns]
    wx_feats   = [c for c in WEATHER_COLS  if c in df.columns]
    enc_feats  = ["brand_tier_enc","hotel_district_enc",
                  "district_functional_tier_enc","is_chain"]
    all_feats  = lag_cols + roll_cols + time_feats + cal_feats + wx_feats + enc_feats
    return [f for f in all_feats if f in df.columns]


def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    fit_encoders(df)
    df = apply_encoders(df)
    df = add_time_features(df)
    df = add_calendar_features(df)
    df = add_lag_rolling_features(df)
    df = add_weather_features(df)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if df[c].isnull().any():
            med = df[c].median()
            df[c] = df[c].fillna(med if not np.isnan(med) else 0)
    return df
