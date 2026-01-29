# RunPod optimized Dockerfile for LLM Distillation
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Mamba fast kernels (must be built with CUDA — installed separately for build cache)
RUN pip install --no-cache-dir causal-conv1d>=1.2.0 mamba-ssm>=1.2.0 tensorboard

# Copy application code
COPY . .

# Create data and models directories
RUN mkdir -p /app/data /app/models /app/outputs

# Default command (can be overridden)
CMD ["python", "-u", "handler.py"]
