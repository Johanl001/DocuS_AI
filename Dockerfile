# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# System dependencies for OCR (tesseract)
RUN apt-get update && apt-get install -y --no-install-recommends \
	tesseract-ocr \
	libtesseract-dev \
	&& rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose a default port (optional; Render injects $PORT)
EXPOSE 8000

# Run the FastAPI app; default to 8000 locally, respect $PORT on Render
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}