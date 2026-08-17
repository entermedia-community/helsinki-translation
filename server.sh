#!/bin/bash
lsof -ti :8600 | xargs -r kill -9

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

export CUDA_VISIBLE_DEVICES=1
mkdir -p /root/logs/uvicorn

uvicorn main:app \
  --port 8600 \
  --host 0.0.0.0 \
  --workers 1 2>&1 | multilog t s5000000 n3 /root/logs/uvicorn &