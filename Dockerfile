FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pre-downloaded embedding model instead of fetching from
# HuggingFace at build time (avoids flaky/slow network calls during build)
COPY model_cache/ ./model_cache/

COPY . ./
CMD ["python", "main.py"]