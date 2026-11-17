"""Train the baseline IMDb sentiment model: TF-IDF + LogisticRegression.

Pipeline
--------
1. Light text cleaning (lowercase, strip HTML line breaks, keep letters
   and digits only).
2. ``TfidfVectorizer`` — word unigrams + bigrams, sublinear TF scaling,
   vocabulary capped at 20,000 features.
3. ``LogisticRegression`` — linear classifier, well suited to sparse
   TF-IDF features and fast on CPU.

The model is trained on the full 25,000-review training split, evaluated
on the held-out 25,000-review test split, and saved as a single
scikit-learn pipeline to ``models/model.joblib`` so that evaluation and
prediction reuse exactly the same preprocessing.

Usage
-----
    python -m src.train
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from src.data_loading import load_data
from src.preprocessing import clean_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

# Cap the vocabulary: 25k long reviews would otherwise produce several
# hundred thousand n-grams, which is slow and mostly noise.
MAX_FEATURES = 20_000
NGRAM_RANGE = (1, 2)
MIN_DF = 2


def build_pipeline() -> Pipeline:
    """Return the TF-IDF + LogisticRegression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    max_features=MAX_FEATURES,
                    ngram_range=NGRAM_RANGE,
                    min_df=MIN_DF,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    train, test = load_data()
    print(f"Train: {len(train):,} reviews | Test: {len(test):,} reviews")

    model = build_pipeline()

    t0 = time.time()
    model.fit(train["text"], train["label"])
    print(f"Training took {time.time() - t0:.1f} s")

    train_acc = accuracy_score(train["label"], model.predict(train["text"]))
    test_acc = accuracy_score(test["label"], model.predict(test["text"]))
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH} ({MODEL_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
