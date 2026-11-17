#!/bin/sh
# Entrypoint for the imdb-sentiment container.
#
# The image ships code only — no trained model and no dataset (both are
# git-ignored). On start it works out how to get a model:
#   1. If models/model.joblib exists (mounted in), use it.
#   2. Else, if the dataset is mounted at data/, train it now
#      (takes about 30 seconds on CPU) and then predict.
#   3. Else, print how to supply one and exit 1.
# After that it hands over to predict.py: a review passed as an
# argument is predicted once; with no argument it reads reviews from
# stdin (one per run — pipe them in with `docker run -i`).
set -e
# In the image WORKDIR is /app; when run outside a container (e.g. local
# validation), work from the script's own directory instead.
if [ -d /app ]; then
    cd /app
else
    cd "$(dirname "$0")"
fi

if [ ! -f models/model.joblib ]; then
    if [ -f data/train.parquet ] && [ -f data/test.parquet ]; then
        echo "No trained model found — training now (about 30 seconds)..."
        python -m src.train
    else
        echo "No trained model found at models/model.joblib."
        echo "Either mount the dataset so the model can be trained:"
        echo "  docker run -v ./data:/app/data -i imdb-sentiment"
        echo "or mount a pre-trained model:"
        echo "  docker run -v ./models:/app/models -i imdb-sentiment"
        exit 1
    fi
fi

exec python predict.py "$@"
