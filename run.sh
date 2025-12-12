export HF_HOME=./models
killall uvicorn
CUDA_VISIBLE_DEVICES=1 uvicorn main:app --host 0.0.0.0 --port 5000 --workers 1 &
