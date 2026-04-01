"""
python/src/models/baselines.py

模型角色：
  ARIMA          per-hotel baseline（对比实验）
  perhotel_xgb   per-hotel XGBoost baseline（对比实验）
  global_xgb     Global XGBoost 主模型（在线推理 + 消融基准）
  xgb_ablation   基于 Global XGBoost 的特征组消融实验
"""
import warnings
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

from src.config import XGB_PARAMS, XGB_PERHOTEL_PARAMS, ARIMA_ORDER, MODELS_DIR
from src.evaluation.metrics import evaluate


LAG_FEATS      = [f"lag_{l}" for l in [1, 2, 3, 7, 14, 21]]
ROLLING_FEATS  = [f"rolling_{t}_{w}"
                  for t in ["mean", "std", "max", "min"]
                  for w in [7, 14, 30]]
CALENDAR_FEATS = ["is_public_holiday", "is_workday",
                  "is_school_vacation", "is_weekend"]
WEATHER_FEATS  = ["tavg", "tmin", "tmax", "prcp", "snow",
                  "wdir", "wspd", "wpgt", "pres", "tsun"]
HOTEL_ATTR_FEATS = ["brand_tier_enc", "hotel_district_enc",
                    "district_functional_tier_enc", "is_chain"]

XGB_ABLATION_CONFIGS = {
    "xgb_full":           [],
    "xgb_w/o_lag":        LAG_FEATS,
    "xgb_w/o_rolling":    ROLLING_FEATS,
    "xgb_w/o_calendar":   CALENDAR_FEATS,
    "xgb_w/o_weather":    WEATHER_FEATS,
    "xgb_w/o_hotel_attr": HOTEL_ATTR_FEATS,
    "xgb_lag_only":       ROLLING_FEATS + CALENDAR_FEATS + WEATHER_FEATS + HOTEL_ATTR_FEATS,
}


def _drop(all_feats: list, exclude: list) -> list:
    return [f for f in all_feats if f not in exclude]


def train_arima(train_df: pd.DataFrame) -> dict:
    models = {}
    for hid in tqdm(train_df["hotel_id"].unique(), desc="ARIMA fit", ncols=70):
        series = (train_df[train_df["hotel_id"] == hid]
                  .sort_values("record_date")["occupancy_rate"].values)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(series, order=ARIMA_ORDER).fit(
                    method_kwargs={"warn_convergence": False})
            models[hid] = {"fitted": fitted, "series": series.tolist()}
        except Exception:
            models[hid] = {"fitted": None, "series": series.tolist(),
                           "fallback": float(series[-1])}
    return models


def eval_arima(arima_models: dict, test_df: pd.DataFrame) -> list:
    records = []
    for hid, obj in arima_models.items():
        te = test_df[test_df["hotel_id"] == hid].sort_values("record_date")
        y_true = te["occupancy_rate"].values
        if obj.get("fitted") is not None:
            try:
                y_pred = obj["fitted"].forecast(steps=len(y_true))
            except Exception:
                y_pred = np.full(len(y_true), obj["series"][-1])
        else:
            y_pred = np.full(len(y_true), obj.get("fallback", 50.0))
        records.append(evaluate(y_true, y_pred, "arima", hid))
    return records


def export_arima(arima_models: dict):
    export = {}
    for hid, obj in arima_models.items():
        export[hid] = {
            "series":   obj["series"],
            "order":    list(ARIMA_ORDER),
            "fallback": obj.get("fallback", float(obj["series"][-1])),
        }
    out_path = MODELS_DIR / "arima" / "arima_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False)
    print(f"  [arima] -> {out_path}")


def run_perhotel_xgb(train_df, test_df, features) -> list:
    records = []
    for hid in tqdm(train_df["hotel_id"].unique(), desc="per-hotel XGB", ncols=70):
        try:
            tr = train_df[train_df["hotel_id"] == hid]
            te = test_df[test_df["hotel_id"] == hid]
            sc = StandardScaler()
            X_tr = sc.fit_transform(tr[features].fillna(0))
            X_te = sc.transform(te[features].fillna(0))
            m = XGBRegressor(**XGB_PERHOTEL_PARAMS)
            m.fit(X_tr, tr["occupancy_rate"].values)
            records.append(evaluate(
                te["occupancy_rate"].values, m.predict(X_te), "perhotel_xgb", hid))
        except Exception:
            records.append({"hotel_id": hid, "model": "perhotel_xgb",
                            "MAE": float("nan"), "RMSE": float("nan"), "sMAPE": float("nan")})
    return records


def train_global_xgb(train_df, test_df, features):
    """Returns (records, model, scaler, all_feats)"""
    all_feats = [f for f in features + ["hotel_id_enc"] if f in train_df.columns]
    sc = StandardScaler()
    X_tr = sc.fit_transform(train_df[all_feats].fillna(0))
    X_te = sc.transform(test_df[all_feats].fillna(0))

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, train_df["occupancy_rate"].values)
    y_pred_all = model.predict(X_te)

    records = []
    for hid in test_df["hotel_id"].unique():
        mask = test_df["hotel_id"].values == hid
        records.append(evaluate(
            test_df["occupancy_rate"].values[mask], y_pred_all[mask], "global_xgb", hid))
    return records, model, sc, all_feats


def export_global_xgb(model, scaler: StandardScaler, feature_cols: list,
                       encoders: dict, hotel_meta_df: pd.DataFrame):
    xgb_dir = MODELS_DIR / "xgboost"

    model_path = xgb_dir / "model.bin"
    model.get_booster().save_model(str(model_path))
    print(f"  [xgb] model        -> {model_path}")

    scaler_path = xgb_dir / "scaler.json"
    with open(scaler_path, "w") as f:
        json.dump({"mean": scaler.mean_.tolist(),
                   "std":  scaler.scale_.tolist(),
                   "features": feature_cols}, f)
    print(f"  [xgb] scaler       -> {scaler_path}")

    enc_map = {col: {str(i): cls for i, cls in enumerate(le.classes_)}
               for col, le in encoders.items()}
    meta_path = xgb_dir / "feature_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"feature_order":  feature_cols,
                   "label_encoders": enc_map,
                   "lag_days":       [1, 2, 3, 7, 14, 21],
                   "roll_windows":   [7, 14, 30]},
                  f, ensure_ascii=False, indent=2)
    print(f"  [xgb] feature_meta -> {meta_path}")

    hotel_path = xgb_dir / "hotel_meta.json"
    hotel_meta_df.to_json(hotel_path, orient="records", force_ascii=False)
    print(f"  [xgb] hotel_meta   -> {hotel_path}")


# XGBoost 消融实验（基于 Global XGBoost）
def run_xgb_ablation(train_df, test_df, all_feats) -> list:
    all_records = []
    for name, exclude in XGB_ABLATION_CONFIGS.items():
        feats = _drop(all_feats, exclude)
        if len(feats) == 0:
            print(f"  [{name}] 跳过：特征为空")
            continue

        print(f"  [{name}] 特征数: {len(feats)}")
        sc = StandardScaler()
        X_tr = sc.fit_transform(train_df[feats].fillna(0))
        X_te = sc.transform(test_df[feats].fillna(0))

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, train_df["occupancy_rate"].values)
        y_pred = model.predict(X_te)

        for hid in test_df["hotel_id"].unique():
            mask = test_df["hotel_id"].values == hid
            all_records.append(evaluate(
                test_df["occupancy_rate"].values[mask], y_pred[mask], name, hid))

    return all_records
