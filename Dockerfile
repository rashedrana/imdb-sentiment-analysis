# IMDb sentiment analysis — container image
#
# Slim Python base; installs only what requirements.txt lists, then
# copies the application code. The trained model (models/model.joblib)
# and the raw dataset (data/) are git-ignored and NOT baked into the
# image — see entrypoint.sh for how the container gets a model:
# either mount the dataset and let it train on first start, or mount
# a pre-trained model.
#
# Build (needs Docker; not available on the author's dev VM, so the
# image build itself is unverified — the app inside it is verified,
# see README "Run with Docker"):
#   docker build -t imdb-sentiment .
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY predict.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
CMD []
