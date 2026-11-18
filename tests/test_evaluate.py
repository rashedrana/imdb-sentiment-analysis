"""Tests for src.evaluate.error_note — the error-analysis labelling aid.

Only the pure function is tested here: the full evaluation (main())
re-runs the whole 25,000-review test set and is Stage 3's job.
"""

from __future__ import annotations

from src.evaluate import error_note


def test_error_note_mixed_sentiment():
    note = error_note(
        "Great acting but a terrible boring plot", proba_pos=0.60, true_label=0
    )
    assert note.startswith("Mixed sentiment")
    assert "positive cue(s)" in note
    assert "negative cue(s)" in note


def test_error_note_borderline_call():
    # No cue words on either side, low confidence in the wrong label.
    note = error_note("It was a film about a dog", proba_pos=0.55, true_label=0)
    assert note.startswith("Borderline call")


def test_error_note_confident_error():
    # No cue words, high confidence in the wrong label.
    note = error_note("It was a film about a dog", proba_pos=0.95, true_label=0)
    assert note.startswith("Confident error")


def test_error_note_confidence_uses_true_label():
    # Same probabilities, opposite true label: confidence is measured
    # against the wrong prediction, so the wording flips accordingly.
    mixed_wrong_neg = error_note(
        "Great acting but a terrible plot", proba_pos=0.60, true_label=0
    )
    mixed_wrong_pos = error_note(
        "Great acting but a terrible plot", proba_pos=0.60, true_label=1
    )
    assert "60%" in mixed_wrong_neg
    assert "40%" in mixed_wrong_pos
