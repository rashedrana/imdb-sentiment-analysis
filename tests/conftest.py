"""Shared test setup for the IMDb sentiment project.

Inserts the project root on ``sys.path`` so ``from src...`` imports
work when pytest is run from anywhere, and provides a tiny synthetic
dataset that lets the pipeline tests train a model in under a second
(no 25,000-review download needed).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

# Tiny, clearly separable reviews: positive ones praise the film,
# negative ones pan it. A linear TF-IDF model separates these easily,
# so tests can assert exact predictions without training on real data.
TINY_TEXTS = [
    "A wonderful fantastic film, I loved every minute of it",
    "Brilliant acting and a superb story, truly excellent",
    "An amazing masterpiece, the best film I have seen",
    "I loved this movie, wonderful and gripping from start to finish",
    "A terrible boring film, I hated every minute of it",
    "Awful acting and a dull story, truly disappointing",
    "A horrible waste of time, the worst film I have seen",
    "I hated this movie, boring and predictable throughout",
]
TINY_LABELS = [1, 1, 1, 1, 0, 0, 0, 0]


@pytest.fixture
def tiny_data():
    """(texts, labels) for fast pipeline tests."""
    return list(TINY_TEXTS), list(TINY_LABELS)


@pytest.fixture
def tiny_model(tiny_data):
    """A TF-IDF + LogisticRegression pipeline fitted on the tiny data."""
    from src.train import build_pipeline

    texts, labels = tiny_data
    model = build_pipeline()
    model.fit(texts, labels)
    return model
