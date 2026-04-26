FROM python:3.11-slim

# HuggingFace Spaces runs on port 7860 by default
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the salespath_env package and training scripts
COPY salespath_env/ ./salespath_env/
COPY training/ ./training/

# Copy and set permissions for the training script
COPY run_hf_training.sh ./run_hf_training.sh
RUN sed -i 's/\r$//' ./run_hf_training.sh && chmod +x ./run_hf_training.sh

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start the FastAPI server on HF Spaces port
CMD ["sh", "-c", "uvicorn salespath_env.server.app:app --host 0.0.0.0 --port ${PORT}"]
