import pandas as pd
import numpy as np
from pathlib import Path


def load_parquet(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "data" / "raw" / "superstore.parquet"
    return pd.read_parquet(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Drop duplicates
    df = df.drop_duplicates()

    # Fix numeric types
    numeric_cols = ["sales", "quantity", "discount", "profit"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "quantity" in df.columns:
        df["quantity"] = df["quantity"].astype("Int64")

    # Date conversions
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    if "ship_date" in df.columns:
        df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce")

    # Clean string categories
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()

    # Feature engineering
    if "order_date" in df.columns:
        df["order_year"] = df["order_date"].dt.year
        df["order_month"] = df["order_date"].dt.month
        df["order_day"] = df["order_date"].dt.day
        df["order_week"] = df["order_date"].dt.isocalendar().week.astype(int)
        df["order_weekday"] = df["order_date"].dt.weekday

    if {"order_date", "ship_date"}.issubset(df.columns):
        df["shipping_time_days"] = (df["ship_date"] - df["order_date"]).dt.days

    if {"profit", "sales"}.issubset(df.columns):
        df["profit_margin"] = df["profit"] / df["sales"]

    if {"profit", "quantity"}.issubset(df.columns):
        df["profit_per_unit"] = df["profit"] / df["quantity"].replace(0, np.nan)

    # Mild outlier handling
    if "profit" in df.columns:
        lower = df["profit"].quantile(0.01)
        upper = df["profit"].quantile(0.99)
        df["profit"] = df["profit"].clip(lower=lower, upper=upper)

    return df


def save_processed(df: pd.DataFrame, base_dir: Path):
    out_path = base_dir / "data" / "processed" / "superstore_processed.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)