FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests

RUN python -m pip install --upgrade pip && \
    python -m pip install .

RUN mkdir -p /app/data/raw /app/data/processed /app/artifacts /app/reports/plots

ENTRYPOINT ["python", "-m", "src"]
CMD ["--help"]
