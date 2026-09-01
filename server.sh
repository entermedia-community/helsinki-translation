#!/bin/bash
lsof -ti :8600 | xargs -r kill -9

# Load environment variables if .env exists
if [ -f "/root/.env" ]; then
    set -a
    source "/root/.env"
    set +a
elif [ -f "$(dirname "$0")/.env" ]; then
    set -a
    source "$(dirname "$0")/.env"
    set +a
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi


LOGFILE=/root/logs/uvicorn
mkdir -p "$LOGFILE"

uvicorn main:app \
  --port 8600 \
  --host 0.0.0.0 \
  --workers 1 2>&1 | multilog t s5000000 n3 "$LOGFILE" &