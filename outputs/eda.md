# Exploratory Data Analysis — IMDb sentiment dataset

Source: Stanford aclImdb corpus (Maas et al., 2011), obtained as Parquet
from the Hugging Face mirror `stanfordnlp/imdb` (`plain_text` config)
because the canonical Stanford tarball was unreachable from this machine.
Both files are the full canonical splits — no subsetting.

## Split sizes

| Split | Reviews |
|-------|--------:|
| Train | 25,000 |
| Test  | 25,000 |
| Total | 50,000 |

## Label balance

| Split | Negative (0) | Positive (1) |
|-------|-------------:|-------------:|
| Train | 12,500 (50.0%) | 12,500 (50.0%) |
| Test  | 12,500 (50.0%) | 12,500 (50.0%) |

The dataset is perfectly balanced — a 50% accuracy baseline is the
chance level, and accuracy is a fair headline metric.

## Review length (words)

| Split | Mean | Median | Min | Max | 90th pct |
|-------|-----:|-------:|----:|----:|---------:|
| Train | 233.8 | 174 | 10 | 2470 | 458 |
| Test  | 228.5 | 172 | 4 | 2278 | 444 |

Reviews are long (median ~170 words), so word-level TF-IDF features
capture plenty of signal. A small number of very long reviews
(max ~2,500 words) justifies capping TF-IDF vocabulary rather than
truncating text.
