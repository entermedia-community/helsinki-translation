from typing import Union, List
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from functools import lru_cache
from opencc import OpenCC

app = FastAPI(title="Helsinki-NLP Translation API")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=32)
def load_model(source: str, target: str):
  """Load and cache Helsinki-NLP model."""

  if source == "pt" or source == "pt_BR":
    source = "ROMANCE"
  
  if target == "pt" or target == "pt_BR":
    target = "ROMANCE"
  
  if target == "bn":
    target = "inc"

  model_name = f"Helsinki-NLP/opus-mt-{source}-{target}"

  tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
  model.to(DEVICE)

  return tokenizer, model


def translate_text(text: str, source: str, target: str) -> str:
  tokenizer, model = load_model(source, target)
  
  if target == "pt" or target == "pt_BR":
    text = f">>pt<< {text}"
  
  if target == "bn":
    text = f">>ben<< {text}"

  inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
  outputs = model.generate(**inputs, max_length=512)
  return tokenizer.decode(outputs[0], skip_special_tokens=True)

def t2s(text: str) -> str:
  cc = OpenCC('t2s')
  return cc.convert(text)

def s2t(text: str) -> str:
  cc = OpenCC('s2t')
  return cc.convert(text)

@app.get("/")
def read_root():
  return {"message": "Welcome to the eMedia Translation API"}

@app.get("/health")
def health_check():
  return {"status": "ok"}


class TranslateRequest(BaseModel):
  text: str
  source: str  # e.g. "en"
  target: Union[str, List[str]]  # e.g. "fr" or ["fr", "de"]

@app.post("/translate")
def translate(req: TranslateRequest):
  text = req.text
  source = req.source.lower()
  targets_arr = req.target

  if isinstance(targets_arr, str):
    targets_arr = [targets_arr.lower()]

  result = {}
  for target_current in targets_arr:

    target = target_current.lower()

    if source == "zht":
      text = t2s(text)
      source = "zh"

    if target == "zht":
      target = "zh"

    if source != "en":
      text_en = translate_text(text, source, "en")
    else:
      text_en = text

    if target != "en":
      result[target_current] = translate_text(text_en, "en", target)
    else:
      result[target_current] = text_en
    
    if target_current == "zht":
      result[target_current] = s2t(result[target_current])

  return {"translatedText": result}
