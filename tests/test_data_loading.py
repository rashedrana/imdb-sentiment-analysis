"""Tests for src.data_loading.load_data.

The full-data checks pin down the canonical aclImdb splits (25,000 /
25,000, balanced 50/50) that every later stage relies on. The error
cases use a temporary directory so they never touch the real data.
"""

from __future__ import annotations

import pandas as pd
import pytest

import src.data_loading as dl
from src.data_loading import load_data


@pytest.fixture(autouse=True)
def restore_data_dir():
    """load_data() mutates the module-level DATA_DIR — restore it."""
    original = dl.DATA_DIR
    yield
    dl.DATA_DIR = original


def test_load_data_returns_canonical_splits():
    train, test = load_data()
    assert list(train.columns) == ["text", "label"]
    assert list(test.columns) == ["text", "label"]
    assert len(train) == 25_000
    assert len(test) == 25_000


def test_load_data_labels_balanced():
    train, test = load_data()
    assert set(train["label"].unique()) == {0, 1}
    assert set(test["label"].unique()) == {0, 1}
    assert train["label"].value_counts().to_dict() == {0: 12_500, 1: 12_500}
    assert test["label"].value_counts().to_dict() == {0: 12_500, 1: 12_500}


def test_load_data_reviews_are_nonempty_text():
    train, _ = load_data()
    assert train["text"].str.len().min() > 0
    assert train["text"].str.strip().str.len().min() > 0


def test_load_data_missing_files_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.parquet"):
        load_data(data_dir=tmp_path)


def test_load_data_missing_columns_raises(tmp_path):
    pd.DataFrame({"review": ["good film"], "sentiment": [1]}).to_parquet(
        tmp_path / "train.parquet"
    )
    pd.DataFrame({"text": ["good film"], "label": [1]}).to_parquet(
        tmp_path / "test.parquet"
    )
    with pytest.raises(ValueError, match="must contain 'text' and 'label'"):
        load_data(data_dir=tmp_path)


def test_load_data_accepts_custom_directory(tmp_path):
    pd.DataFrame({"text": ["a great film", "a bad film"], "label": [1, 0]}).to_parquet(
        tmp_path / "train.parquet"
    )
    pd.DataFrame({"text": ["a great film"], "label": [1]}).to_parquet(
        tmp_path / "test.parquet"
    )
    train, test = load_data(data_dir=tmp_path)
    assert len(train) == 2
    assert len(test) == 1
    assert train["label"].tolist() == [1, 0]
