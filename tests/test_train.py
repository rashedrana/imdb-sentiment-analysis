"""Tests for src.train.build_pipeline — the TF-IDF + LogisticRegression model.

Trains only on the tiny synthetic fixture (8 reviews), so the suite
stays fast; the full 25,000-review training run is Stage 2's job and
is verified by its recorded accuracy (0.8992), not re-run here.
"""

from __future__ import annotations

from sklearn.metrics import accuracy_score

from src.train import MAX_FEATURES, MIN_DF, NGRAM_RANGE, build_pipeline


def test_pipeline_structure_matches_design():
    model = build_pipeline()
    assert list(model.named_steps) == ["tfidf", "clf"]
    tfidf = model.named_steps["tfidf"]
    assert tfidf.max_features == MAX_FEATURES == 20_000
    assert tfidf.ngram_range == NGRAM_RANGE == (1, 2)
    assert tfidf.min_df == MIN_DF == 2
    assert tfidf.sublinear_tf is True
    clf = model.named_steps["clf"]
    assert clf.C == 1.0
    assert clf.max_iter == 1000
    assert clf.random_state == 42


def test_pipeline_fits_and_predicts_tiny_data(tiny_data):
    texts, labels = tiny_data
    model = build_pipeline()
    model.fit(texts, labels)
    preds = model.predict(texts)
    assert set(preds.tolist()) <= {0, 1}
    # Only 8 toy reviews and min_df=2 drops rare words, so perfect
    # separation is not guaranteed — the point is the pipeline fits
    # and predicts sensibly, not that it aces 8 hand-written lines.
    assert accuracy_score(labels, preds) >= 0.75
    assert set(preds.tolist()) == {0, 1}


def test_pipeline_predict_proba_sums_to_one(tiny_model, tiny_data):
    texts, _ = tiny_data
    proba = tiny_model.predict_proba(texts)
    assert proba.shape == (len(texts), 2)
    assert (abs(proba.sum(axis=1) - 1.0) < 1e-9).all()


def test_pipeline_separates_clear_sentiment(tiny_model):
    assert tiny_model.predict(["A wonderful brilliant film, I loved it"])[0] == 1
    assert tiny_model.predict(["A terrible boring film, I hated it"])[0] == 0
