# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Environment optimizations
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set the working directory
WORKDIR /app

# Install system dependencies required for OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the rest of your project files
COPY . .

# Expose port
EXPOSE 7860

# Use Gunicorn with Uvicorn workers for better concurrency
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:7860", "api:app"]