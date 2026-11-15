#!/bin/bash

# Exit immediately if any command fails
set -e

ENV_FILE="/home/amodh/projects/Recon/.env"

if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
    echo "[INFO] Successfully loaded .env from: $ENV_FILE"
else
    echo "[ERROR] .env file not found at $ENV_FILE" >&2
    exit 1
fi

# Fallback defaults if variables are missing in .env
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
CTX_SIZE="${CTX_SIZE:-4096}"
GPU_LAYERS="${GPU_LAYERS:-40}"
THREADS="${THREADS:-4}"

# Check if model path is defined
if [ -z "$MODEL_PATH" ]; then
    echo "[ERROR] MODEL_PATH is not defined in your .env file." >&2
    exit 1
fi

echo "[INFO] Starting Llama Server..."
echo "[INFO] Port: $PORT | Context: $CTX_SIZE | GPU Layers: $GPU_LAYERS"

exec llama-server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --ctx-size "$CTX_SIZE" \
    --n-gpu-layers "$GPU_LAYERS" \
    --threads "$THREADS"