"""
python/predict_arima.py
Go 调用做 ARIMA 预测
"""
import sys
import json
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from pathlib import Path
MODELS_DIR = Path(__file__).parent.parent / "models"


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "usage: predict_arima.py <hotel_id> <steps>"}))
        sys.exit(1)

    hotel_id = sys.argv[1]
    steps    = int(sys.argv[2])

    data_path = MODELS_DIR / "arima" / "arima_data.json"
    with open(data_path, encoding="utf-8") as f:
        arima_data = json.load(f)

    if hotel_id not in arima_data:
        print(json.dumps({"error": f"hotel_id {hotel_id} not found"}))
        sys.exit(1)

    obj    = arima_data[hotel_id]
    series = obj["series"]
    order  = tuple(obj["order"])

    try:
        from statsmodels.tsa.arima.model import ARIMA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ARIMA(series, order=order).fit(
                method_kwargs={"warn_convergence": False})
        preds = fitted.forecast(steps=steps).tolist()
        # clip to [0, 100]
        preds = [max(0.0, min(100.0, p)) for p in preds]
    except Exception as e:
        fallback = obj.get("fallback", series[-1])
        preds = [float(fallback)] * steps

    result = {
        "hotel_id":    hotel_id,
        "steps":       steps,
        "predictions": preds,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
