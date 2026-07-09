# Multi-stage ROCm build for RouteZero
# Stage 1: Install dependencies
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM rocm/dev-ubuntu-22.04:latest AS runtime

# Install Python
RUN apt-get update && apt-get install -y python3.12 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY . .

# Run batch mode on startup, then keep server alive
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000"]
