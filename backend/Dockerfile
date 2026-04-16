FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-backend.txt /app/requirements-backend.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/requirements-backend.txt

COPY backend /app/backend

CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
