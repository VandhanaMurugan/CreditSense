# CreditSense scoring API (FastAPI). Build from the repository root:
#   docker build -t creditsense-api .
#   docker run -p 7860:7860 creditsense-api          -> http://localhost:7860/docs
# Hugging Face Spaces (Docker SDK) uses port 7860; Render and similar hosts set $PORT.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# OpenMP runtime needed by xgboost and lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Code, the frozen v2.0.0 bundle, and the dataset used only to derive age bands for /policy_simulation group rates
COPY creditsense/ creditsense/
COPY models/creditsense_bundle_v2.0.0.joblib models/
COPY data/GiveMeSomeCredit_dataset.csv data/

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','7860'))"
CMD ["sh", "-c", "uvicorn creditsense.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
