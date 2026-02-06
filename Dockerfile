# Use the official Python base image
FROM python:3.12-slim-bookworm

# Install system dependencies for OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Use a venv outside bind-mounted /app to avoid conflicts
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
# Force uv to use the container's Python instead of downloading one
ENV UV_PYTHON_PREFERENCE=only-system

# Copy lock files first for better Docker layer caching
COPY pyproject.toml ./

# Install dependencies into /opt/venv
RUN uv sync --frozen || uv sync

# Now copy the rest of the project
COPY . .

# Ensure environment matches pyproject after full copy
RUN uv sync

# Set timezone
ENV TZ=Asia/Almaty
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Run worker
CMD ["python", "worker.py"]
