from pathlib import Path

import pandas as pd


def load_dataset(path: str | Path, separator: str = ";", encoding: str = "utf-8") -> pd.DataFrame:
    """Load the source CSV without silently replacing malformed text."""
    return pd.read_csv(path, sep=separator, encoding=encoding)


def combine_text_columns(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Combine selected fields into one model input while preserving missing values safely."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing text columns: {missing}")
    return frame[columns].fillna("").astype(str).agg(" [SEP] ".join, axis=1)

