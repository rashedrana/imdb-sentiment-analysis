"""Interactive demo for the IMDb sentiment model (Gradio).

Type or paste a movie review and the demo shows the predicted
sentiment (positive / negative) with the model's confidence for
both classes.

The demo loads the trained pipeline from ``models/model.joblib``.
If no trained model is present it trains one first
(``python -m src.train``, under a minute with the dataset in
``data/``) — the same train-if-needed behaviour as ``entrypoint.sh``.
Without a model and without the dataset it prints clear setup
instructions instead of crashing.

Usage
-----
    pip install -r requirements-demo.txt
    python app.py
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import joblib

from predict import MODEL_PATH, predict_sentiment
from src import train as train_module

PROJECT_ROOT = Path(__file__).resolve().parent

EXAMPLES = [
    "This movie was a fantastic, gripping ride from start to finish.",
    "Dull, predictable, and far too long.",
    "I loved every minute of it, truly superb.",
]


def load_or_train_model():
    """Return the trained pipeline, training one if needed.

    Raises
    ------
    FileNotFoundError
        If there is no saved model and no dataset to train one from.
    """
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    data_dir = PROJECT_ROOT / "data"
    if not (data_dir / "train.parquet").exists():
        raise FileNotFoundError(
            "No trained model at models/model.joblib and no dataset in "
            "data/ to train one. Run: python -m src.train (after placing "
            "train.parquet and test.parquet in data/, see the README)."
        )
    train_module.main()
    return joblib.load(MODEL_PATH)


try:
    _MODEL = load_or_train_model()
    _LOAD_ERROR = None
except FileNotFoundError as exc:
    _MODEL = None
    _LOAD_ERROR = str(exc)


def classify(review: str) -> dict[str, float]:
    """Return {sentiment: probability} for both classes."""
    if _MODEL is None:
        raise gr.Error(_LOAD_ERROR)
    if not review or not review.strip():
        raise gr.Error("Please type or paste a review first.")
    sentiment, confidence = predict_sentiment(review, _MODEL)
    other_label = "negative" if sentiment == "positive" else "positive"
    return {sentiment: confidence, other_label: 1.0 - confidence}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(
        label="Movie review",
        placeholder="Type or paste a movie review here…",
        lines=4,
    ),
    outputs=gr.Label(label="Sentiment"),
    examples=EXAMPLES,
    title="IMDb Movie Review Sentiment Analysis",
    description=(
        "TF-IDF + Logistic Regression trained on 25,000 IMDb reviews "
        "(test accuracy 89.9%, ROC AUC 0.963). The bars show the model's "
        "confidence for each class."
    ),
)


if __name__ == "__main__":
    demo.launch()
