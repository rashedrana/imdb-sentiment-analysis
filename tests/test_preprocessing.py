"""Tests for src.preprocessing.clean_text — the shared text cleaner.

This function is also the pipeline's preprocessor (pickled into the
saved model), so a regression here would silently change predictions.
"""

from __future__ import annotations

import pytest

from src.preprocessing import clean_text


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("I LOVED This Film!", "i loved this film"),
        ("Great acting!<br />Terrible plot.", "great acting terrible plot"),
        ("Great acting!<br/>Terrible plot.", "great acting terrible plot"),
        ("So   much   whitespace\n\nand\ttabs", "so much whitespace and tabs"),
        ("Don't stop: 10/10, five stars!!!", "don t stop 10 10 five stars"),
        ("  padded review  ", "padded review"),
        ("", ""),
        ("!!!", ""),
        ("Room 101 was great", "room 101 was great"),
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_clean_text_keeps_digits_and_lowercases():
    assert clean_text("TOP 10 Films Of 2024") == "top 10 films of 2024"


def test_clean_text_is_deterministic():
    text = "A <br /> Mixed, REVIEW!"
    assert clean_text(text) == clean_text(text)
