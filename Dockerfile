FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY shared/ ./shared/
COPY data/raw/sensors_registry.json ./data/raw/sensors_registry.json
COPY step_3_eda/ ./step_3_eda/
COPY step_4_regression/ ./step_4_regression/
COPY step_5_classification/ ./step_5_classification/
COPY artifacts/ ./artifacts/

EXPOSE 8080

CMD ["sh","-c","uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
