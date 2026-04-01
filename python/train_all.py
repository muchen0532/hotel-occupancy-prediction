"""
python/train_all.py
==============================================================
完整训练流程：
  Step 1  数据加载（select_hotels.csv, 含calendar + weather）
  Step 2  特征工程
  Step 3  ARIMA baseline
  Step 4  per-hotel XGBoost baseline
  Step 5  Global XGBoost 主模型 + 消融实验（特征组）
  Step 6  SHAP 分析
  Step 7  导出

运行方式：
    cd python
    python train_all.py
==============================================================
"""
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from src.config import MODELS_DIR, SEED
from src.data.loader import load_raw, split_dates
from src.data.feature_engineering import run_pipeline, get_feature_cols, get_encoders
from src.models.baselines import (
    train_arima, eval_arima, export_arima,
    run_perhotel_xgb,
    train_global_xgb, export_global_xgb,
    run_xgb_ablation,
)
from src.evaluation.metrics import summarize
from src.explainability.shap_analysis import run_shap_pipeline

np.random.seed(SEED)

RESULT_DIR = Path(__file__).parent / "outputs" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def build_hotel_meta(df):
    meta = (df.drop_duplicates("hotel_id")
            [["hotel_id", "brand_tier", "hotel_district",
              "district_functional_tier", "is_chain"]].copy())
    vol = (df.groupby("hotel_id")["occupancy_rate"].std()
           .reset_index().rename(columns={"occupancy_rate": "occ_std"}))
    q1, q3 = vol["occ_std"].quantile([0.33, 0.67])
    vol["volatility_group"] = pd.cut(
        vol["occ_std"], bins=[-np.inf, q1, q3, np.inf],
        labels=["low", "mid", "high"])
    return meta.merge(vol[["hotel_id", "volatility_group"]], on="hotel_id")


def main():
    t0 = time.time()

    # ══ Step 1: 数据加载 ══════════════════════════════════
    print("\n" + "="*60)
    print("  STEP 1/8  数据加载")
    print("="*60)
    df = load_raw()
    print(f"  数据: {len(df):,} 行, {df['hotel_id'].nunique()} 家酒店")

    # ══ Step 2: 特征工程 ══════════════════════════════════
    print("\n" + "="*60)
    print("  STEP 2/8  特征工程")
    print("="*60)
    df = run_pipeline(df)
    df, test_start, val_start = split_dates(df)
    features = get_feature_cols(df)
    encoders  = get_encoders()
    print(f"  特征数: {len(features)}")
    print(f"  训练 < {val_start.date()} | 验证 < {test_start.date()} | 测试 >= {test_start.date()}")

    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    test_df  = df[df["split"] == "test"].copy()
    print(f"  行数: 训练 {len(train_df):,} | 验证 {len(val_df):,} | 测试 {len(test_df):,}")

    hotel_meta = build_hotel_meta(df)

    all_records = []

    # ══ Step 3: ARIMA ═════════════════════════════════════
    print("\n" + "="*60)
    print("  STEP 3/8  ARIMA baseline (per-hotel)")
    print("="*60)
    arima_models  = train_arima(train_df)
    arima_records = eval_arima(arima_models, test_df)
    all_records  += arima_records
    export_arima(arima_models)

    # ══ Step 4: per-hotel XGBoost ═════════════════════════
    print("\n" + "="*60)
    print("  STEP 4/8  per-hotel XGBoost baseline")
    print("="*60)
    all_records += run_perhotel_xgb(train_df, test_df, features)

    # ══ Step 5: Global XGBoost 主模型 + 消融 ══════════════
    print("\n" + "="*60)
    print("  STEP 5/8  Global XGBoost 主模型 + XGBoost 消融实验")
    print("="*60)
    xgb_records, xgb_model, xgb_scaler, xgb_feats = \
        train_global_xgb(train_df, test_df, features)
    all_records += xgb_records
    export_global_xgb(xgb_model, xgb_scaler, xgb_feats, encoders, hotel_meta)

    print("\n  [XGBoost 消融实验]")
    ablation_records = run_xgb_ablation(train_df, test_df, xgb_feats)
    all_records += ablation_records

    # ══ Step 6: SHAP 分析 ═════════════════════════════════
    print("\n" + "="*60)
    print("  STEP 7/8  SHAP 特征重要性分析")
    print("="*60)
    X_test_np = xgb_scaler.transform(test_df[xgb_feats].fillna(0).values)
    y_pred_xgb = xgb_model.predict(X_test_np)
    run_shap_pipeline(
        xgb_model, X_test_np, test_df, xgb_feats,
        hotel_meta, test_df["occupancy_rate"].values, y_pred_xgb,
        result_dir=RESULT_DIR)

    # ══ Step 7: 保存评估报告 ══════════════════════════════
    print("\n" + "="*60)
    print("  STEP 8/8  保存评估报告")
    print("="*60)

    results_df = pd.DataFrame(all_records)
    results_df.to_csv(RESULT_DIR / "results_all_models.csv",
                      index=False, encoding="utf-8-sig")

    summary = summarize(all_records)
    summary.to_csv(RESULT_DIR / "results_summary.csv", encoding="utf-8-sig")

    # 消融结果单独输出
    from src.models.baselines import XGB_ABLATION_CONFIGS
    xgb_abl_names = list(XGB_ABLATION_CONFIGS.keys())
    xgb_abl_df = results_df[results_df["model"].isin(xgb_abl_names)]
    if not xgb_abl_df.empty:
        (xgb_abl_df.groupby("model")
         .agg(MAE_mean=("MAE","mean"), RMSE_mean=("RMSE","mean"), sMAPE_mean=("sMAPE","mean"))
         .round(4).sort_values("MAE_mean")
         .to_csv(RESULT_DIR / "results_xgb_ablation.csv", encoding="utf-8-sig"))

    # eval_report.json
    report = {
        "summary":    summary.reset_index().to_dict(orient="records"),
        "per_hotel":  results_df.fillna(-1).to_dict(orient="records"),
        "test_start": str(test_start.date()),
        "n_hotels":   int(df["hotel_id"].nunique()),
        "n_test_rows": len(test_df),
    }
    eval_path = MODELS_DIR / "eval_report.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  eval_report  -> {eval_path}")

    # history.json
    hist = df[["hotel_id","record_date","occupancy_rate",
               "brand_tier","district_functional_tier"]].copy()
    hist["record_date"] = hist["record_date"].dt.strftime("%Y-%m-%d")
    hist_path = MODELS_DIR / "history.json"
    hist.to_json(hist_path, orient="records", force_ascii=False)
    print(f"  history      -> {hist_path}")

    # 控制台汇总
    print("\n" + "="*60)
    print("  最终模型性能排名（测试集 MAE 均值）")
    print("="*60)
    print(summary[["MAE_mean", "MAE_std", "RMSE_mean", "sMAPE_mean"]].to_string())

    elapsed = (time.time() - t0) / 60
    print(f"\n✅  完成！耗时 {elapsed:.1f} 分钟")
    print(f"   模型目录: {MODELS_DIR}")
    print(f"   结果目录: {RESULT_DIR}")
    print(f"\n提示：Go 服务启动命令：")
    print(f"   go run cmd/server/main.go -python <venv路径>/python.exe")


if __name__ == "__main__":
    main()
