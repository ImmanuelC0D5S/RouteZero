FROM python:3.12-slim
WORKDIR /app

# llama-cpp-python needs a C++ build toolchain to compile from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY model_cache/ ./model_cache/
COPY . ./
CMD ["python", "main.py"]