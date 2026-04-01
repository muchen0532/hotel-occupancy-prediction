"""
python/src/data/loader.py
"""
import pandas as pd
from src.config import SELECTED_CSV, CALENDAR_CSV, WEATHER_CSV, TEST_RATIO, VAL_RATIO


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(SELECTED_CSV, parse_dates=["record_date"])
    df = df.drop(columns=[
        "date_x", "date_y", "date_only"
    ], errors="ignore")
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.sort_values(["hotel_id", "record_date"]).reset_index(drop=True)
    return df


def split_dates(df: pd.DataFrame):
    """全局日期切分，返回 (df_with_split_col, test_start, val_start)"""
    all_dates  = sorted(df["record_date"].unique())
    n_test     = int(len(all_dates) * TEST_RATIO)
    n_val      = int(len(all_dates) * (1 - TEST_RATIO) * VAL_RATIO)
    test_start = all_dates[-n_test]
    val_start  = all_dates[-(n_test + n_val)]

    df = df.copy()
    df["split"] = "train"
    df.loc[df["record_date"] >= val_start,  "split"] = "val"
    df.loc[df["record_date"] >= test_start, "split"] = "test"
    return df, test_start, val_start
