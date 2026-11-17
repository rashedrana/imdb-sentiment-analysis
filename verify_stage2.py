"""Stage 2 verification: load the saved model and predict three sample
reviews — one clearly positive, one clearly negative, one mixed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib

MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.joblib"

SAMPLES = [
    (
        "positive",
        "This movie was absolutely wonderful — brilliant acting, a gripping "
        "story, and beautiful cinematography. I loved every minute of it and "
        "would happily watch it again.",
    ),
    (
        "negative",
        "A complete waste of time. Terrible acting, a boring and confusing "
        "plot, and awful dialogue. I hated it and walked out halfway through.",
    ),
    (
        "mixed",
        "The acting was great and some scenes were genuinely moving, but the "
        "story dragged on and the ending felt rushed and disappointing.",
    ),
]

LABELS = {0: "negative", 1: "positive"}


def main() -> None:
    assert MODEL_PATH.exists(), f"saved model not found: {MODEL_PATH}"
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH} ({MODEL_PATH.stat().st_size:,} bytes)")

    for expected, review in SAMPLES:
        pred = int(model.predict([review])[0])
        proba = model.predict_proba([review])[0]
        print(
            f"\n[{expected}] -> predicted: {LABELS[pred]} "
            f"(P(positive) = {proba[1]:.3f})"
        )
        print(f"  {review[:120]}...")

    # The clearly positive and clearly negative samples must be right;
    # the mixed one is genuinely ambiguous, so it is reported, not asserted.
    assert int(model.predict([SAMPLES[0][1]])[0]) == 1, "positive sample misclassified"
    assert int(model.predict([SAMPLES[1][1]])[0]) == 0, "negative sample misclassified"
    print("\nStage 2 verification passed: positive and negative samples classified correctly.")


if __name__ == "__main__":
    main()
