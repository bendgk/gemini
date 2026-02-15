FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# git might be needed if installing packages from git, but here we stick to pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create data directory
RUN mkdir -p data

# Default command (can be overridden in docker-compose)
CMD ["python", "recorder.py"]
