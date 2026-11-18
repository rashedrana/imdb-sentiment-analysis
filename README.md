# IMDb Movie Review Sentiment Analysis

Sentiment analysis on the full IMDb movie reviews dataset — TF-IDF + Logistic Regression, with full evaluation.

## Dataset

The Stanford aclImdb corpus (Maas et al., 2011): 50,000 labelled movie reviews, split into 25,000 training and 25,000 test reviews, balanced 50/50 positive/negative. The dataset files live in `data/` (git-ignored — download them yourself, see below).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The canonical tarball is https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz — if that is unreachable, the identical data is available as Parquet from the Hugging Face mirror [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) (`plain_text` config). Place `train.parquet` and `test.parquet` in `data/`.

## Usage

```bash
# Train the baseline model
python -m src.train

# Evaluate on the held-out test set
python -m src.evaluate

# Predict the sentiment of a review
python predict.py "This movie was a fantastic, gripping ride from start to finish."
# Sentiment: positive
# Confidence: 94.41%
```

You can also pipe a review in via stdin (`echo "I hated every minute." | python predict.py`). Passing no text prints a short hint instead of an error.

## Results

Baseline model — TF-IDF (unigrams + bigrams, 20,000 features) + LogisticRegression, trained on the full 25,000-review training split and evaluated on the held-out 25,000-review test split:

| Metric | Value |
| --- | --- |
| Accuracy | **0.8992** |
| ROC AUC | **0.9629** |

| Class | Precision | Recall | F1-score |
| --- | --- | --- | --- |
| Negative | 0.9031 | 0.8942 | 0.8987 |
| Positive | 0.8953 | 0.9041 | 0.8997 |

Confusion matrix (rows = true, columns = predicted): 11,178 true negatives, 1,322 false positives, 1,199 false negatives, 11,301 true positives — 2,521 misclassified (10.1%). Most errors are borderline or mixed-sentiment reviews (e.g. praising the acting while disliking the film overall). Full details and error examples: [`outputs/evaluation.md`](outputs/evaluation.md).

## Project structure

```
├── src/
│   ├── data_loading.py   # loads train/test splits into pandas
│   ├── preprocessing.py  # text cleaning (shared by train/evaluate/predict)
│   ├── train.py          # TF-IDF + LogisticRegression training (Stage 2)
│   └── evaluate.py       # held-out evaluation + error analysis (Stage 3)
├── predict.py            # prediction CLI (Stage 4)
├── tests/                # automated tests (Week 3): preprocessing, data loading,
│                         # training pipeline, prediction CLI, error notes
├── data/                 # dataset files (git-ignored)
├── models/               # trained model (git-ignored)
├── outputs/              # EDA and evaluation reports
├── Dockerfile            # container setup (Stage 5)
└── requirements.txt
```

## Running tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
# 30 passed
```

The suite covers text cleaning, data loading (including the canonical
25,000 / 25,000 splits), the training pipeline, the prediction CLI and
the evaluation error notes. Pipeline tests train on 8 hand-written
reviews so the suite runs in ~10 seconds; the two CLI tests that need
the real saved model skip gracefully on a fresh checkout until
`python -m src.train` has been run.

## Run with Docker

Build the image (the image contains code only — no dataset, no trained model):

```bash
docker build -t imdb-sentiment .
```

The container needs a model. Either mount the dataset and let it train on first start (about 30 seconds), or mount a pre-trained model:

```bash
# Train on first start, then predict the piped-in review
echo "A gripping, superbly acted film — I loved every minute." \
  | docker run -i --rm -v ./data:/app/data imdb-sentiment

# Or skip training: mount a model trained locally with `python -m src.train`
echo "Dull, predictable, and far too long." \
  | docker run -i --rm -v ./models:/app/models imdb-sentiment

# A review can also be passed as an argument instead of piped in
docker run --rm -v ./models:/app/models imdb-sentiment \
  "A gripping, superbly acted film — I loved every minute."
```

> Note: Docker is not installed on the machine this project was developed on, so the `docker build` step itself is unverified there. Everything the image would run — `pip install -r requirements.txt`, the entrypoint's train-if-needed logic, and `predict.py` — was validated in a fresh virtual environment with the same Python version (3.12) and the same commands, and produces the verified results above.

## License

MIT — see [LICENSE](LICENSE).
