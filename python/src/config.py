"""
python/src/config.py — 全局配置
"""
from pathlib import Path

ROOT_DIR   = Path(__file__).parent.parent.parent
DATA_DIR   = Path(__file__).parent.parent / "data"
MODELS_DIR = ROOT_DIR / "models"

for d in [DATA_DIR, MODELS_DIR,
          MODELS_DIR / "arima",
          MODELS_DIR / "xgboost",
          MODELS_DIR / "transformer"]:
    d.mkdir(parents=True, exist_ok=True)

# ── 数据文件 ──────────────────────────────────────────────
SELECTED_CSV = DATA_DIR / "select_hotels.csv"
CALENDAR_CSV = DATA_DIR / "calendar_2021_2025.csv"
WEATHER_CSV  = DATA_DIR / "weather-beijing.csv"

# ── 实验切分 ──────────────────────────────────────────────
SEED       = 42
TEST_RATIO = 0.20
VAL_RATIO  = 0.10

# ── 特征工程 ──────────────────────────────────────────────
LAG_DAYS     = [1, 2, 3, 7, 14, 21]
ROLL_WINDOWS = [7, 14, 30]
WEATHER_COLS = ["tavg","tmin","tmax","prcp","snow","wdir","wspd","wpgt","pres","tsun"]
CALENDAR_COLS= ["is_public_holiday","is_workday","is_school_vacation"]
TIME_COLS    = ["day_of_week","month","day_of_month","quarter","is_weekend"]

# ── Global XGBoost ────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    random_state=SEED, verbosity=0, n_jobs=-1,
)
XGB_PERHOTEL_PARAMS = dict(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbosity=0,
)

# ── ARIMA ─────────────────────────────────────────────────
ARIMA_ORDER = (1, 1, 1)


# ── SHAP ──────────────────────────────────────────────────
SHAP_SAMPLE_SIZE = 3000
