"""Tests for predict.py — the prediction CLI.

predict_sentiment() is tested on the tiny fitted model. The CLI itself
is tested end-to-end where it does not need the trained model (empty
input), and against the real saved model only when it exists — on a
fresh checkout / in CI there is no model until `python -m src.train`
runs, so that test skips instead of failing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"


def test_predict_sentiment_returns_label_and_confidence(tiny_model):
    from predict import predict_sentiment

    sentiment, confidence = predict_sentiment(
        "A wonderful brilliant film, I loved it", tiny_model
    )
    assert sentiment == "positive"
    assert 0.5 < confidence <= 1.0

    sentiment, confidence = predict_sentiment(
        "A terrible boring film, I hated it", tiny_model
    )
    assert sentiment == "negative"
    assert 0.5 < confidence <= 1.0


def _run_cli(*args, input_text=None):
    return subprocess.run(
        [sys.executable, "predict.py", *args],
        cwd=PROJECT_ROOT,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_cli_empty_argument_prints_hint_and_exits_zero():
    result = _run_cli("")
    assert result.returncode == 0
    assert "No review text given" in result.stdout


def test_cli_empty_stdin_prints_hint_and_exits_zero():
    result = _run_cli(input_text="")
    assert result.returncode == 0
    assert "No review text given" in result.stdout


@pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="no trained model — run python -m src.train first"
)
def test_cli_positive_review_with_real_model():
    result = _run_cli("This movie was a fantastic, gripping ride from start to finish.")
    assert result.returncode == 0
    assert "Sentiment: positive" in result.stdout
    assert "Confidence:" in result.stdout


@pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="no trained model — run python -m src.train first"
)
def test_cli_negative_review_with_real_model():
    result = _run_cli("I hated every minute of this boring mess.")
    assert result.returncode == 0
    assert "Sentiment: negative" in result.stdout
