FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for psycopg + cryptography wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
      curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install -e .

COPY . .
RUN chmod +x docker/start-api.sh

EXPOSE 8000

# Default command runs the API entrypoint (migrations + uvicorn).
# docker-compose overrides for the worker (no migrations, just arq).
CMD ["./docker/start-api.sh"]
