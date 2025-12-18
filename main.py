from typing import Union, List
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from functools import lru_cache
from opencc import OpenCC

app = FastAPI(title="eMedia Translation API")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
model.to(DEVICE)

available_languages = {
  "en": "eng_Latn",
  "fr": "fra_Latn",
  "de": "deu_Latn",
  "es": "spa_Latn",
  "pt": "por_Latn",
  "pt_BR": "por_Brai",  # Brazilian Portuguese
  "ru": "rus_Cyrl",
  "zh": "zho_Hans",
  "zht": "zho_Hant",
  "hi": "hin_Deva",
  "ar": "ara_Arab",
  "bn": "ben_Beng",
  "ur": "urd_Arab",
  "sw": "swa_Latn"
}


def translate_text(text: str, src: str, target: str) -> str:
  inputs = tokenizer(text, src_lang=src, return_tensors="pt").to(DEVICE)
  outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target])
  return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

@app.get("/")
def read_root():
  return {"message": "Welcome to the eMedia Translation API"}

@app.get("/health")
@app.get("/health.ico")
def health_check():
  return {"status": "ok"}


def verify_langs(source: str, targets: List[str]) -> Union[bool, str]:
  if source not in available_languages.keys():
    return False, f"Source language '{source}' is not supported. Available languages: {', '.join(available_languages)}"
  
  for target in targets:
    if target not in available_languages.keys():
      return False, f"Target language '{target}' is not supported. Available languages: {', '.join(available_languages)}"
  
  return True, ""

class TranslateRequest(BaseModel):
  q: Union[str, List[str]]
  source: str  # e.g. "en"
  target: Union[str, List[str]]  # e.g. "fr" or ["fr", "de"]

@app.post("/translate")
def translate(req: TranslateRequest):
  text_arr = req.q
  if isinstance(text_arr, str):
    text_arr = [text_arr]
  
  source = req.source.lower()
  targets_arr = req.target

  if isinstance(targets_arr, str):
    targets_arr = [targets_arr.lower()]
  
  valid, err = verify_langs(source, targets_arr)
  if not valid:
    return {"error": err}

  result = {}
  for target_current in targets_arr:

    target = target_current.lower()

    result[target_current] = []

    for text in text_arr:
      translation = translate_text(text, src=available_languages[source], target=available_languages[target])
      result[target_current].append(translation)

  return {"translatedText": result}
