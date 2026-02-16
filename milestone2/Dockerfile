# -------- Builder stage --------
FROM python:3.11-slim AS builder
WORKDIR /build

COPY app/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# -------- Runtime stage --------
FROM python:3.11-slim
WORKDIR /app

COPY --from=builder /install /usr/local
COPY app/ /app/app/

EXPOSE 8080

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]
