from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from functools import lru_cache
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=32)
def load_model(source: str, target: str):
  """Load and cache Helsinki-NLP model."""
  model_name = f"Helsinki-NLP/opus-mt-{source}-{target}"

  tokenizer = AutoTokenizer.from_pretrained(model_name)#, local_files_only=True)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name)#, local_files_only=True)
  model.to(DEVICE)

  return tokenizer, model



with open("config.json", "r") as f:
  config = json.load(f)
for lang_pair in config["language_pairs"]:
  source, target = lang_pair.split("-")

  if(source == "zht" or target == "zht"):
    print("Handled with OpenCC, skipping download.")
    continue
  
  print(f"Downloading model for {source} -> {target}...")
  load_model(source, target)
  print(f"Model for {source} -> {target} downloaded.")