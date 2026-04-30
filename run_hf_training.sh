#!/usr/bin/env bash
set -euo pipefail
cd /app

# ====================================================================
# SalesPath Training Pipeline — Configuration
# ====================================================================
# Override any of these via HF Space "Variables and secrets" settings.
#
# GPU VRAM Guide (for GRPO with LoRA 4-bit):
#   T4 (16GB)  → 0.5B-3B models   → num_generations=2-4, batch=1-2
#   L4 (24GB)  → 7B models         → num_generations=2, batch=1
#   A10G (24GB)→ 7B models         → num_generations=4, batch=2
#   A100 (40GB)→ 14B-32B models    → num_generations=4, batch=4
#
# Example for 7B on L4:
#   MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
#   NUM_GENERATIONS=2
#   PER_DEVICE_BATCH=1
#   MAX_SEQ_LEN=512
# ====================================================================

export PORT="${PORT:-7860}"
export HF_MODEL_REPO="${HF_MODEL_REPO:-Imsachin010/salespath-qwen25-0.5b}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
export OUTPUT_DIR="${OUTPUT_DIR:-/app/salespath_out}"
export GRPO_STEPS="${GRPO_STEPS:-100}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
export PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export EVAL_EPISODES="${EVAL_EPISODES:-4}"

echo "========================================"
echo "  SalesPath Training Pipeline"
echo "  Model:       $MODEL_NAME"
echo "  HF Repo:     $HF_MODEL_REPO"
echo "  GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Port:        $PORT"
echo "========================================"

# ------------------------------------------------------------------
# 1. Background health server (keeps HF Spaces happy during training)
# ------------------------------------------------------------------
echo "Starting background health server on port $PORT..."
python3 -c "
import http.server, socketserver, os
PORT = int(os.environ.get('PORT', 7860))
class H(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200); self.end_headers(); self.wfile.write(b'OK')
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *a): pass
with socketserver.TCPServer(('', PORT), H) as httpd:
    httpd.serve_forever()
" &
HEALTH_PID=$!
echo "Health server PID: $HEALTH_PID"
sleep 2

# ------------------------------------------------------------------
# 2. HF login (if token is set as secret)
# ------------------------------------------------------------------
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "Logging in to Hugging Face Hub..."
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

# ------------------------------------------------------------------
# 3. Pre-flight check
# ------------------------------------------------------------------
echo "=== Pre-flight check ==="
python training/preflight_check.py || echo "Pre-flight warning (non-fatal)"

# ------------------------------------------------------------------
# 4. Start environment server (needed for rollout-based training)
# ------------------------------------------------------------------
echo "Starting SalesPath environment server on port 8000..."
uvicorn salespath_env.server.app:app --host 0.0.0.0 --port 8000 &
ENV_PID=$!
sleep 3

# Verify environment server is healthy
python3 -c "
import httpx, time
for i in range(10):
    try:
        r = httpx.get('http://127.0.0.1:8000/health', timeout=5)
        if r.status_code == 200: print('Environment server OK'); break
    except: pass
    time.sleep(2)
"

# ------------------------------------------------------------------
# 5. GRPO Training
# ------------------------------------------------------------------
echo ""
echo "=== GRPO Training with $MODEL_NAME ==="
echo "Steps: $GRPO_STEPS | Generations: $NUM_GENERATIONS | Batch: $PER_DEVICE_BATCH"

PYTORCH_ALLOC_CONF=expandable_segments:True \
python -u -m training.grpo_train \
    --mode grpo \
    --env-url http://127.0.0.1:8000 \
    --model-name "$MODEL_NAME" \
    --grpo-steps "$GRPO_STEPS" \
    --grpo-dataset-size 128 \
    --num-generations "$NUM_GENERATIONS" \
    --max-completion-length "$MAX_SEQ_LEN" \
    --per-device-train-batch-size "$PER_DEVICE_BATCH" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --output-dir "$OUTPUT_DIR" \
    --logging-steps "$LOGGING_STEPS"

TRAINING_EXIT=$?
echo "GRPO training exit code: $TRAINING_EXIT"

# ------------------------------------------------------------------
# 6. Evaluation: baseline vs trained
# ------------------------------------------------------------------
if [[ $TRAINING_EXIT -eq 0 ]]; then
    echo ""
    echo "=== Evaluation: Baseline vs Trained ==="
    python training/eval_baseline_vs_trained.py \
        --base "$MODEL_NAME" \
        --trained "$OUTPUT_DIR/grpo_final" \
        --env-url http://127.0.0.1:8000 \
        --episodes-per-level "$EVAL_EPISODES" \
        --output "$OUTPUT_DIR/eval_results.json"

    echo ""
    echo "=== Generating reward plots ==="
    python training/plot_rewards.py \
        --input "$OUTPUT_DIR/reward_history.txt" \
        --output "$OUTPUT_DIR/reward_graph.png" || echo "Plotting skipped"
fi

# ------------------------------------------------------------------
# 7. Upload artifacts to Hugging Face Hub
# ------------------------------------------------------------------
if [[ $TRAINING_EXIT -eq 0 && -n "${HF_TOKEN:-}" ]]; then
    echo ""
    echo "=== Uploading to $HF_MODEL_REPO ==="

    # Upload GRPO adapters
    huggingface-cli upload "$HF_MODEL_REPO" "$OUTPUT_DIR/grpo_final" . --repo-type model || true

    # Upload logs and plots
    for f in reward_history.txt eval_results.json reward_graph.png; do
        if [[ -f "$OUTPUT_DIR/$f" ]]; then
            huggingface-cli upload "$HF_MODEL_REPO" "$OUTPUT_DIR/$f" "training_artifacts/$f" --repo-type model || true
        fi
    done

    echo "Upload complete!"
fi

# ------------------------------------------------------------------
# 8. Keep container alive for log inspection
# ------------------------------------------------------------------
echo ""
echo "Training pipeline complete."
echo "Container will stay alive. Check logs via HF Spaces dashboard."
echo "Stop the Space manually when done to avoid further billing."

# Kill background servers
kill $HEALTH_PID 2>/dev/null || true
kill $ENV_PID 2>/dev/null || true

# Start keepalive app
exec uvicorn training.hf_keepalive_app:app --host 0.0.0.0 --port "$PORT"
