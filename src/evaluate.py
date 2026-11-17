"""Full held-out evaluation of the saved IMDb sentiment model.

Loads ``models/model.joblib`` (trained in Stage 2) and evaluates it on
the 25,000-review held-out test split. Reports:

- accuracy
- precision / recall / F1 per class (negative / positive)
- confusion matrix
- ROC AUC
- error analysis: 5 misclassified reviews, each with a one-line note

Results are printed to stdout and written to ``outputs/evaluation.md``.

Usage
-----
    python -m src.evaluate
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src.data_loading import load_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
REPORT_PATH = PROJECT_ROOT / "outputs" / "evaluation.md"

CLASS_NAMES = ["negative", "positive"]
N_ERROR_EXAMPLES = 5

# Small cue-word lists used only to describe *why* a review may have been
# misclassified (mixed praise/criticism, negation, sarcasm markers).
# These are not features — just a labelling aid for the error analysis.
POSITIVE_CUES = {
    "great", "excellent", "brilliant", "fantastic", "wonderful", "amazing",
    "love", "loved", "best", "enjoyed", "superb", "outstanding", "masterpiece",
}
NEGATIVE_CUES = {
    "terrible", "awful", "horrible", "boring", "worst", "hate", "hated",
    "waste", "dull", "bad", "disappointing", "poor", "stupid",
}


def error_note(text: str, proba_pos: float, true_label: int) -> str:
    """One-line explanation of why a review was probably misclassified."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    n_pos = len(words & POSITIVE_CUES)
    n_neg = len(words & NEGATIVE_CUES)
    confidence = proba_pos if true_label == 0 else 1 - proba_pos
    if n_pos and n_neg:
        return (
            f"Mixed sentiment (mentions both praise and criticism: "
            f"{n_pos} positive cue(s), {n_neg} negative cue(s)); "
            f"model was {confidence:.0%} confident in the wrong label."
        )
    if confidence < 0.65:
        return (
            f"Borderline call — model only {confidence:.0%} confident in "
            f"the wrong label; review reads near-neutral."
        )
    return (
        f"Confident error — model was {confidence:.0%} sure of the wrong "
        f"label; likely sarcasm, irony, or praise of one aspect "
        f"(e.g. acting) while the overall verdict is the opposite."
    )


def main() -> None:
    _, test = load_data()
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
    print(f"Evaluating on {len(test):,} held-out test reviews...")

    y_true = test["label"]
    y_pred = model.predict(test["text"])
    y_proba_pos = model.predict_proba(test["text"])[:, 1]

    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba_pos)
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4
    )
    cm = confusion_matrix(y_true, y_pred)
    # Rows = true label, columns = predicted label.
    tn, fp, fn, tp = cm.ravel()

    print(f"\nAccuracy: {acc:.4f}")
    print(f"ROC AUC:  {roc_auc:.4f}\n")
    print("Classification report:")
    print(report)
    print("Confusion matrix (rows = true, columns = predicted):")
    print(f"                 pred negative  pred positive")
    print(f"true negative        {tn:>6,}       {fp:>6,}")
    print(f"true positive        {fn:>6,}       {tp:>6,}")

    # ---- Error analysis: first 5 misclassified reviews ------------------
    wrong = test[y_true != y_pred].copy()
    wrong["proba_pos"] = y_proba_pos[y_true != y_pred]
    examples = wrong.head(N_ERROR_EXAMPLES)

    print(f"\nError analysis — {len(wrong):,} misclassified out of "
          f"{len(test):,} ({len(wrong) / len(test):.1%})")
    error_lines = []
    for i, (_, row) in enumerate(examples.iterrows(), start=1):
        true_name = CLASS_NAMES[int(row["label"])]
        pred_name = CLASS_NAMES[1 - int(row["label"])]
        note = error_note(row["text"], row["proba_pos"], int(row["label"]))
        snippet = " ".join(row["text"].split())[:300]
        print(f"\n{i}. True: {true_name} | Predicted: {pred_name} "
              f"| P(positive) = {row['proba_pos']:.3f}")
        print(f"   {note}")
        print(f'   Review: "{snippet}..."')
        error_lines.append(
            f"### {i}. True: {true_name} | Predicted: {pred_name} "
            f"| P(positive) = {row['proba_pos']:.3f}\n\n"
            f"{note}\n\n"
            f"> \"{snippet}...\"\n"
        )

    # ---- Write the report ----------------------------------------------
    report_md = f"""# Evaluation — IMDb Sentiment Analysis

Model: TF-IDF (unigrams + bigrams, 20,000 features) + LogisticRegression,
trained on the full 25,000-review training split (Stage 2) and evaluated
here on the held-out 25,000-review test split.

## Headline metrics

| Metric | Value |
| --- | --- |
| Accuracy | {acc:.4f} |
| ROC AUC | {roc_auc:.4f} |
| Test set size | {len(test):,} reviews |

Chance level is 50% (the test set is balanced 50/50).

## Per-class results

```
{report}```

## Confusion matrix

Rows = true label, columns = predicted label.

|  | Predicted negative | Predicted positive |
| --- | --- | --- |
| True negative | {tn:,} | {fp:,} |
| True positive | {fn:,} | {tp:,} |

## Error analysis

{len(wrong):,} of {len(test):,} test reviews were misclassified
({len(wrong) / len(test):.1%}). Five examples:

{chr(10).join(error_lines)}
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_md)
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
