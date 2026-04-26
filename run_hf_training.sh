#!/bin/bash

# Start the environment server in the background
echo "Starting SalesPath environment server..."
uvicorn salespath_env.server.app:app --host 0.0.0.0 --port 8000 &

# Give the server a few seconds to start up completely
sleep 5

# Start the GRPO Training using standard HuggingFace (PEFT)
echo "Starting 7B GRPO Training..."
PYTORCH_ALLOC_CONF=expandable_segments:True python -m training.grpo_train \
    --mode grpo \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --grpo-steps 150 \
    --grpo-dataset-size 128 \
    --num-generations 4 \
    --max-completion-length 256 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 8 \
    --output-dir ./salespath_out \
    --logging-steps 10 \
    --push-to-hub \
    --hub-repo Imsachin010/salespath-qwen25-7b

echo "Training complete and pushed to hub! Keeping container alive for logs..."
tail -f /dev/null
