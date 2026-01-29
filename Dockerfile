# RunPod optimized Dockerfile for LLM Distillation
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Verify pre-installed PyTorch CUDA version (should be 12.1 from base image)
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

# Copy requirements first for caching
# NOTE: requirements.txt must NOT contain torch — it's pre-installed in base image
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Mamba fast kernels — must match the base image CUDA (12.1)
# Install causal-conv1d first (mamba-ssm depends on it)
RUN pip install --no-cache-dir causal-conv1d mamba-ssm tensorboard

# Copy application code
COPY . .

# Create data and models directories
RUN mkdir -p /app/data /app/models /app/outputs

# Default command (can be overridden)
CMD ["python", "-u", "handler.py"]
