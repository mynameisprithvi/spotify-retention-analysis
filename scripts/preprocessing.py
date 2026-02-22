import pandas as pd
import numpy as np


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()

    # Ensure numeric audio features
    audio_cols = [
        "danceability", "energy", "valence", "loudness",
        "speechiness", "acousticness", "tempo", "liveness"
    ]

    for col in audio_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Log-transform retention (optional for modeling)
    if "weeks_on_chart" in df.columns:
        df["log_weeks"] = np.log1p(df["weeks_on_chart"])

    # Binary target: short-lived ≤ 5 weeks
    df["short_lived"] = (df["weeks_on_chart"] <= 5).astype(int)

    df = df.dropna()

    return df