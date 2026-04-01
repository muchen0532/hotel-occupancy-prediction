"""
python/predict_xgboost.py
Go 调用做 XGBoost 推理
"""
import sys
import os
import json
import numpy as np

ROOT       = __import__("pathlib").Path(__file__).parent
MODELS_DIR = ROOT.parent / "models"


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "usage: predict_xgboost.py <hotel_id> <date>"}))
        sys.exit(1)

    hotel_id = sys.argv[1]
    date_str = sys.argv[2]

    # ── 读特征向量 ────────────────────────────────────────
    vec_str = os.environ.get("FEATURE_VEC", "")
    if not vec_str:
        print(json.dumps({"error": "FEATURE_VEC env not set"}))
        sys.exit(1)
    try:
        vec = np.array(json.loads(vec_str), dtype=np.float32)
    except Exception as e:
        print(json.dumps({"error": f"parse FEATURE_VEC: {e}"}))
        sys.exit(1)

    # ── 加载 scaler ───────────────────────────────────────
    try:
        with open(MODELS_DIR / "xgboost" / "scaler.json") as f:
            sc = json.load(f)
        mean = np.array(sc["mean"], dtype=np.float32)
        std  = np.array(sc["std"],  dtype=np.float32)
        std[std == 0] = 1.0
        vec_scaled = np.clip((vec - mean) / std, -5, 5)
    except Exception as e:
        print(json.dumps({"error": f"load scaler: {e}"}))
        sys.exit(1)

    # ── 加载模型并推理 ────────────────────────────────────
    try:
        from xgboost import XGBRegressor
        import xgboost as xgb

        model_path = MODELS_DIR / "xgboost" / "model.bin"
        booster = xgb.Booster()
        booster.load_model(str(model_path))

        dmat = xgb.DMatrix(vec_scaled.reshape(1, -1))
        pred = float(booster.predict(dmat)[0])
        pred = max(0.0, min(100.0, pred))
    except Exception as e:
        print(json.dumps({"error": f"inference: {e}"}))
        sys.exit(1)

    print(json.dumps({
        "hotel_id":   hotel_id,
        "date":       date_str,
        "prediction": round(pred, 2),
    }))


if __name__ == "__main__":
    main()
