"""Predict the sentiment of a movie review with the trained IMDb model.

Loads the saved pipeline from ``models/model.joblib`` (trained in
Stage 2) and prints the predicted sentiment (positive / negative) plus
the model's confidence.

Usage
-----
    python predict.py "This movie was a fantastic, gripping ride."
    python predict.py                  # reads the review from stdin
    echo "I hated every minute." | python predict.py

Empty input is handled gracefully: the script prints a short note and
exits with status 0 instead of crashing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

CLASS_NAMES = ["negative", "positive"]


def predict_sentiment(review: str, model) -> tuple[str, float]:
    """Return (sentiment, confidence) for one review string."""
    proba = model.predict_proba([review])[0]
    label = int(proba.argmax())
    return CLASS_NAMES[label], float(proba[label])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict the sentiment of a movie review."
    )
    parser.add_argument(
        "review",
        nargs="?",
        default=None,
        help="the review text (if omitted, it is read from stdin)",
    )
    args = parser.parse_args()

    review = args.review
    if review is None:
        review = sys.stdin.read()

    if review is None or not review.strip():
        print(
            "No review text given — please pass a review as an argument, "
            "e.g. python predict.py \"I loved this film!\""
        )
        return 0

    if not MODEL_PATH.exists():
        print(
            f"Trained model not found at {MODEL_PATH}. "
            "Train it first with: python -m src.train"
        )
        return 1

    model = joblib.load(MODEL_PATH)
    sentiment, confidence = predict_sentiment(review, model)

    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
