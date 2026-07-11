# Multi-stage CPU build for RouteZero
# Stage 1: Install pure-Python dependencies
FROM --platform=linux/amd64 python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime — CPU-only with llama.cpp
FROM --platform=linux/amd64 python:3.12-slim

# Install system dependencies for llama.cpp BLAS support
RUN apt-get update && apt-get install -y libopenblas-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Install llama-cpp-python with CPU BLAS optimization (OpenBLAS)
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
    pip install --no-cache-dir llama-cpp-python

# Copy application code only — do NOT copy .env into the image
COPY routezero/ routezero/
COPY app.py .
COPY main.py .

# Copy the local GGUF model
COPY models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf /app/models/

# Default: run batch mode (reads /input/tasks.json, writes /output/results.json, exits 0)
CMD ["python", "main.py"]
