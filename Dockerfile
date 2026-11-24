FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app/src DEV_MODE=0
# Production: set AUTH_SECRET, DB_PATH (or swap StateStore for Postgres) and pin manifests with scripts/pin_manifests.py
EXPOSE 8000
CMD ["uvicorn", "assistant.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
