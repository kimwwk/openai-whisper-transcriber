# Base image with NVIDIA CUDA support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user and group
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# Copy requirements file and install Python dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./app /app/app

# Change ownership to the appuser
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
