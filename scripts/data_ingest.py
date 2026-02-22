from pathlib import Path
import pandas as pd


def load_raw(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "data" / "spotify_top_songs_audio_features.csv"

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)

    return df