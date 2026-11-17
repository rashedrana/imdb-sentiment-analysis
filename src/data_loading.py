"""Load the IMDb movie-review sentiment dataset into pandas DataFrames.

The dataset is the Stanford aclImdb corpus (Maas et al., 2011): 25,000
labelled training reviews and 25,000 labelled test reviews, balanced
50/50 between positive and negative sentiment.

Data source
-----------
The canonical tarball (https://ai.stanford.edu/~amaas/data/sentiment/
aclImdb_v1.tar.gz) was unreachable from this machine, so the identical
data was obtained as Parquet files from the Hugging Face mirror
``stanfordnlp/imdb`` (``plain_text`` config), stored under ``data/``:

    data/train.parquet   (25,000 rows)
    data/test.parquet    (25,000 rows)

Both files carry the columns ``text`` (the review) and ``label``
(0 = negative, 1 = positive).

Usage
-----
    from src.data_loading import load_data
    train_df, test_df = load_data()
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Column names used by the Hugging Face parquet export.
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# Canonical aclImdb split sizes.
EXPECTED_TRAIN_SIZE = 25_000
EXPECTED_TEST_SIZE = 25_000


def _load_split(filename: str) -> pd.DataFrame:
    """Load one split from ``data/`` and normalise its columns."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. "
            "Download train.parquet / test.parquet from "
            "https://huggingface.co/datasets/stanfordnlp/imdb "
            "(plain_text config) into the data/ directory."
        )
    df = pd.read_parquet(path)
    if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"{filename} must contain '{TEXT_COLUMN}' and '{LABEL_COLUMN}' "
            f"columns, got: {list(df.columns)}"
        )
    df = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
    return df


def load_data(data_dir: str | os.PathLike | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train and test splits.

    Parameters
    ----------
    data_dir:
        Directory holding ``train.parquet`` and ``test.parquet``.
        Defaults to the project ``data/`` directory.

    Returns
    -------
    (train_df, test_df):
        Two DataFrames with columns ``text`` and ``label``
        (0 = negative, 1 = positive).
    """
    global DATA_DIR
    if data_dir is not None:
        DATA_DIR = Path(data_dir)
    train_df = _load_split("train.parquet")
    test_df = _load_split("test.parquet")
    return train_df, test_df


if __name__ == "__main__":
    train, test = load_data()
    print(f"Train: {len(train):,} reviews")
    print(f"Test:  {len(test):,} reviews")
    print("\nTrain label balance:")
    print(train[LABEL_COLUMN].value_counts().to_string())
    print("\nTest label balance:")
    print(test[LABEL_COLUMN].value_counts().to_string())
