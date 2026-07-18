# Docker image for the UFC Fight Predictor Flask app — used by the Hugging Face
# Space (Docker SDK). The serving path needs no xgboost/scraper deps, so the
# image installs only requirements-app.txt and runs gunicorn on HF's port 7860.
FROM python:3.12-slim

# HF Spaces run the container as a non-root user (uid 1000)
RUN useradd -m -u 1000 user
WORKDIR /app

# install deps first for layer caching
COPY --chown=user:user requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# app code + the runtime data the app loads at startup (snapshots, fights,
# models); .dockerignore keeps the venv, caches, raw data and notebooks out
COPY --chown=user:user . .

USER user
EXPOSE 7860
CMD ["gunicorn", "app.app:app", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120"]
