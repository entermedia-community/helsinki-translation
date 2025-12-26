cd /workspace/translation
export HF_HOME=./models
killall uvicorn
CUDA_VISIBLE_DEVICES=0 uvicorn main:app --host 0.0.0.0 --port 5000 --workers 1 &