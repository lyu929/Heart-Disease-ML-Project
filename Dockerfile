# REST API (default) or Streamlit UI (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HEARTRISK_BUNDLE=/app/models/heartrisk.joblib MPLBACKEND=Agg
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api,ui]"

COPY data ./data
COPY app ./app
# The bundle is trained at build time from the versioned data, so the image is reproducible.
RUN heartrisk train --out models/heartrisk.joblib

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["uvicorn", "heartrisk.api:app", "--host", "0.0.0.0", "--port", "8000"]
