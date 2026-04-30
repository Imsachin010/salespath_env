FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# HuggingFace Spaces runs on port 7860 by default
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git curl \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Pin NumPy to avoid breakage
RUN pip install --no-cache-dir --upgrade pip && \
    pip install "numpy<2"

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the salespath_env package and training scripts
COPY salespath_env/ ./salespath_env/
COPY training/ ./training/
COPY scripts/ ./scripts/

# Install the salespath_env package
RUN pip install -e . --no-deps || true

# Copy and set permissions for the training entrypoint
COPY run_hf_training.sh ./run_training.sh
RUN sed -i 's/\r$//' ./run_training.sh && chmod +x ./run_training.sh

# NO HEALTHCHECK — the entrypoint script starts a background health server
# to keep HF Spaces alive during long training runs

CMD ["/bin/bash", "./run_training.sh"]
