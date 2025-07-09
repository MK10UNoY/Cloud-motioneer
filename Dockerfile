# Use Python base image
FROM python:3.10-slim

# Set environment variables to prevent interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all app files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for checkpoint (optional)
RUN mkdir -p /app/checkpoints

# Expose FastAPI port
EXPOSE 8080

# Run FastAPI app (do not download checkpoint here)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
