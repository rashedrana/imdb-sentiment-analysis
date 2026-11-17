"""Text cleaning for the IMDb sentiment pipeline.

Kept in its own module (rather than inside ``train.py``) on purpose:
scikit-learn pickles a custom preprocessor by module reference, so the
function must live in an importable module for the saved pipeline in
``models/model.joblib`` to load from any script (evaluate, predict, ...).
"""

from __future__ import annotations

import re

# HTML line breaks are the only markup left in the plain-text reviews.
HTML_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, drop HTML breaks and punctuation, collapse whitespace."""
    text = text.lower()
    text = HTML_BR_RE.sub(" ", text)
    text = NON_ALNUM_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text
