"""
python/src/explainability/shap_analysis.py
特征重要性分析
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"
FIGURE_DIR = Path(__file__).parent.parent.parent / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_LABELS = {
    "lag_1":"Lag-1d","lag_2":"Lag-2d","lag_3":"Lag-3d",
    "lag_7":"Lag-7d","lag_14":"Lag-14d","lag_21":"Lag-21d",
    "rolling_mean_7":"RollMean-7d","rolling_mean_14":"RollMean-14d","rolling_mean_30":"RollMean-30d",
    "rolling_std_7":"RollStd-7d","rolling_std_14":"RollStd-14d","rolling_std_30":"RollStd-30d",
    "rolling_max_7":"RollMax-7d","rolling_max_14":"RollMax-14d","rolling_max_30":"RollMax-30d",
    "rolling_min_7":"RollMin-7d","rolling_min_14":"RollMin-14d","rolling_min_30":"RollMin-30d",
    "day_of_week":"DayOfWeek","month":"Month","day_of_month":"DayOfMonth",
    "quarter":"Quarter","is_weekend":"IsWeekend",
    "month_sin":"MonthSin","month_cos":"MonthCos","dow_sin":"DowSin","dow_cos":"DowCos",
    "is_public_holiday":"PublicHoliday","is_workday":"IsWorkday","is_school_vacation":"SchoolVacation",
    "tavg":"TempAvg","tmin":"TempMin","tmax":"TempMax","prcp":"Precip","snow":"Snow",
    "wdir":"WindDir","wspd":"WindSpeed","wpgt":"WindGust","pres":"Pressure","tsun":"Sunshine",
    "hotel_id_enc":"HotelID","brand_tier_enc":"BrandTier",
    "hotel_district_enc":"District","district_functional_tier_enc":"FuncTier","is_chain":"IsChain",
}
FEAT_COLORS = {
    "lag":"#1976D2","rolling":"#42A5F5","calendar":"#FF5722",
    "weather":"#4CAF50","hotel":"#9C27B0",
}
WEATHER_COLS = ["tavg","tmin","tmax","prcp","snow","wdir","wspd","wpgt","pres","tsun"]
CALENDAR_COLS= ["is_public_holiday","is_workday","is_school_vacation"]


def _color(feat):
    if "lag" in feat:      return FEAT_COLORS["lag"]
    if "rolling" in feat:  return FEAT_COLORS["rolling"]
    if feat in CALENDAR_COLS or feat in ["is_weekend","month","day_of_week",
       "day_of_month","quarter","month_sin","month_cos","dow_sin","dow_cos"]:
        return FEAT_COLORS["calendar"]
    if feat in WEATHER_COLS: return FEAT_COLORS["weather"]
    return FEAT_COLORS["hotel"]


def compute_shap(model, X_test, features, sample_size=3000, seed=42):
    import shap
    np.random.seed(seed)
    idx = np.random.choice(len(X_test), min(sample_size, len(X_test)), replace=False)
    explainer   = shap.TreeExplainer(model.get_booster())
    shap_values = explainer.shap_values(X_test[idx])

    return pd.DataFrame(shap_values, columns=features), idx


def plot_global_importance(shap_df, top_n=20):
    mean_abs = shap_df.abs().mean().sort_values(ascending=False).head(top_n)
    labels   = [FEATURE_LABELS.get(f, f) for f in mean_abs.index]
    colors   = [_color(f) for f in mean_abs.index]
    fig, ax  = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(mean_abs)), mean_abs.values, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(mean_abs))); ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(f"Global Feature Importance (Top {top_n})\nGlobal XGBoost — Beijing Hotels",
                 fontsize=12, fontweight="bold")
    legend = [mpatches.Patch(facecolor=v, label=k.capitalize()) for k,v in FEAT_COLORS.items()]
    ax.legend(handles=legend, fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = FIGURE_DIR / "fig1_global_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved: {path}")
    return mean_abs


def plot_holiday_heterogeneity(shap_df, meta_df):
    if "is_public_holiday" not in shap_df.columns: return
    tmp = meta_df.copy(); tmp["h_shap"] = shap_df["is_public_holiday"].values
    stats = tmp.groupby("district_functional_tier")["h_shap"].agg(["mean","std","count"]).reset_index().sort_values("mean")
    labels = [s.replace("_","\n") for s in stats["district_functional_tier"]]
    colors = ["#E53935" if v>0 else "#1E88E5" for v in stats["mean"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(range(len(stats)), stats["mean"], color=colors, edgecolor="white", height=0.6)
    ax.errorbar(stats["mean"], range(len(stats)),
                xerr=stats["std"]/np.sqrt(stats["count"]), fmt="none", color="gray", capsize=3)
    ax.set_yticks(range(len(stats))); ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean SHAP Value — Public Holiday Feature", fontsize=11)
    ax.set_title("Holiday Effect Heterogeneity Across Functional Hotel Types", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = FIGURE_DIR / "fig2_holiday_heterogeneity.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved: {path}")


def plot_functype_heatmap(shap_df, meta_df):
    top_feats  = shap_df.abs().mean().sort_values(ascending=False).head(12).index.tolist()
    func_tiers = sorted(meta_df["district_functional_tier"].unique())
    matrix = np.array([shap_df[meta_df["district_functional_tier"]==t][top_feats].abs().mean().values
                       for t in func_tiers])
    col_max = matrix.max(axis=0, keepdims=True); col_max[col_max==0] = 1
    matrix_norm = matrix / col_max
    feat_labels = [FEATURE_LABELS.get(f,f) for f in top_feats]
    tier_labels = [t.replace("_","\n") for t in func_tiers]
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(matrix_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(top_feats))); ax.set_xticklabels(feat_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(func_tiers))); ax.set_yticklabels(tier_labels, fontsize=9)
    plt.colorbar(im, ax=ax, label="Relative |SHAP| (col-normalized)")
    ax.set_title("Feature Importance Heatmap by Functional Hotel Type", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = FIGURE_DIR / "fig3_functype_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved: {path}")


def generate_paper_table(shap_df, meta_df, features):
    top15 = shap_df.abs().mean().sort_values(ascending=False).head(15).index.tolist()
    g_imp = shap_df.abs().mean()
    rows  = []
    for feat in top15:
        row = {"Feature": FEATURE_LABELS.get(feat,feat), "Global": round(g_imp[feat],4)}
        for tier in sorted(meta_df["district_functional_tier"].unique()):
            mask = meta_df["district_functional_tier"] == tier
            row[tier] = round(shap_df[mask][feat].abs().mean(), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def run_shap_pipeline(model, X_test, test_df, features, hotel_meta, y_true, y_pred,
                       result_dir=None):
    """完整 SHAP 分析流水线，被 train_all.py 调用"""
    if result_dir is None:
        result_dir = Path(__file__).parent.parent.parent / "outputs" / "results"
    result_dir = Path(result_dir); result_dir.mkdir(parents=True, exist_ok=True)

    print("  计算SHAP值...")
    shap_df, sample_idx = compute_shap(model, X_test, features)

    test_reset = test_df.reset_index(drop=True)
    sampled_hids = test_reset.iloc[sample_idx]["hotel_id"].values
    meta_sample = (pd.DataFrame({"hotel_id": sampled_hids})
                   .merge(hotel_meta[["hotel_id","district_functional_tier","brand_tier"]],
                          on="hotel_id", how="left").reset_index(drop=True))

    print("  生成图表...")
    plot_global_importance(shap_df)
    plot_holiday_heterogeneity(shap_df, meta_sample)
    plot_functype_heatmap(shap_df, meta_sample)

    tbl = generate_paper_table(shap_df, meta_sample, features)
    tbl_path = result_dir / "shap_feature_table.csv"
    tbl.to_csv(tbl_path, index=False, encoding="utf-8-sig")
    print(f"  saved: {tbl_path}")
    return shap_df, tbl
