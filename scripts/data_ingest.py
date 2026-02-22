from pathlib import Path
import pandas as pd


def load_raw(base_dir: Path) -> pd.DataFrame:
    """
    Load the raw Superstore CSV dataset.

    Parameters
    ----------
    base_dir : Path
        Root directory of the project (passed from run_training.py)

    Returns
    -------
    pd.DataFrame
        Raw dataset as DataFrame
    """
    path = base_dir / "data" / "raw" / "Sample - Superstore.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at expected location: {path}"
        )

    # Use latin-1 to handle Excel-style encodings
    df = pd.read_csv(path, low_memory=False, encoding="latin-1")

    return df


def save_raw_parquet(df: pd.DataFrame, base_dir: Path):
    """
    Save raw DataFrame to Parquet format for faster downstream processing.
    """
    out_path = base_dir / "data" / "raw" / "superstore.parquet"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_path)

    print(f"Raw data saved to parquet at: {out_path}")