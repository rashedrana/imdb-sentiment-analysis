"""Stage 1 verification: load the splits, print sizes and samples,
and write an EDA summary to outputs/eda.md."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loading import (
    EXPECTED_TEST_SIZE,
    EXPECTED_TRAIN_SIZE,
    load_data,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    train, test = load_data()

    print(f"Train: {len(train):,} reviews (expected {EXPECTED_TRAIN_SIZE:,})")
    print(f"Test:  {len(test):,} reviews (expected {EXPECTED_TEST_SIZE:,})")
    assert len(train) == EXPECTED_TRAIN_SIZE, "train size mismatch"
    assert len(test) == EXPECTED_TEST_SIZE, "test size mismatch"

    print("\n--- Sample positive review (train) ---")
    pos = train[train["label"] == 1].iloc[0]["text"]
    print(pos[:600])
    print("\n--- Sample negative review (train) ---")
    neg = train[train["label"] == 0].iloc[0]["text"]
    print(neg[:600])

    # ---- EDA ----
    def length_stats(df):
        lengths = df["text"].str.split().str.len()
        return {
            "mean": round(float(lengths.mean()), 1),
            "median": int(lengths.median()),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "p90": int(lengths.quantile(0.9)),
        }

    train_stats = length_stats(train)
    test_stats = length_stats(test)
    train_bal = train["label"].value_counts(normalize=True).sort_index()
    test_bal = test["label"].value_counts(normalize=True).sort_index()

    eda = f"""# Exploratory Data Analysis — IMDb sentiment dataset

Source: Stanford aclImdb corpus (Maas et al., 2011), obtained as Parquet
from the Hugging Face mirror `stanfordnlp/imdb` (`plain_text` config)
because the canonical Stanford tarball was unreachable from this machine.
Both files are the full canonical splits — no subsetting.

## Split sizes

| Split | Reviews |
|-------|--------:|
| Train | {len(train):,} |
| Test  | {len(test):,} |
| Total | {len(train) + len(test):,} |

## Label balance

| Split | Negative (0) | Positive (1) |
|-------|-------------:|-------------:|
| Train | {int((train['label'] == 0).sum()):,} ({train_bal[0]:.1%}) | {int((train['label'] == 1).sum()):,} ({train_bal[1]:.1%}) |
| Test  | {int((test['label'] == 0).sum()):,} ({test_bal[0]:.1%}) | {int((test['label'] == 1).sum()):,} ({test_bal[1]:.1%}) |

The dataset is perfectly balanced — a 50% accuracy baseline is the
chance level, and accuracy is a fair headline metric.

## Review length (words)

| Split | Mean | Median | Min | Max | 90th pct |
|-------|-----:|-------:|----:|----:|---------:|
| Train | {train_stats['mean']} | {train_stats['median']} | {train_stats['min']} | {train_stats['max']} | {train_stats['p90']} |
| Test  | {test_stats['mean']} | {test_stats['median']} | {test_stats['min']} | {test_stats['max']} | {test_stats['p90']} |

Reviews are long (median ~170 words), so word-level TF-IDF features
capture plenty of signal. A small number of very long reviews
(max ~2,500 words) justifies capping TF-IDF vocabulary rather than
truncating text.
"""
    out_path = PROJECT_ROOT / "outputs" / "eda.md"
    out_path.write_text(eda)
    print(f"\nEDA summary written to {out_path}")


if __name__ == "__main__":
    main()
