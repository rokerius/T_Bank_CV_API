FROM python:3.10-slim

WORKDIR /app

# Установим зависимости для OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY tests/ tests/
COPY weights/ weights/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
