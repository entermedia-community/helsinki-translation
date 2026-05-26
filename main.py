from typing import Union, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
# from functools import lru_cache
# import requests

# LLM_SERVER_URL = "http://0.0.0.0:7600/v1/chat/completions"

app = FastAPI(title="eMedia Translation API")

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "facebook/nllb-200-1.3B"

def load_model():
  model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
  print(f"Model loaded in {device}")
  return model

model = load_model()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

available_languages = {
  "en": {"code": "eng_Latn", "name": "English"},
  "fr": {"code": "fra_Latn", "name": "French"},
  "de": {"code": "deu_Latn", "name": "German"},
  "es": {"code": "spa_Latn", "name": "Spanish"},
  "pt": {"code": "por_Latn", "name": "Portuguese"},
  "pt_br": {"code": "por_Latn", "name": "Brazilian Portuguese"},
  "ru": {"code": "rus_Cyrl", "name": "Russian"},
  "zh": {"code": "zho_Hans", "name": "Chinese (Simplified)"},
  "zht": {"code": "zho_Hant", "name": "Chinese (Traditional)"},
  "hi": {"code": "hin_Deva", "name": "Hindi"},
  "ar": {"code": "arb_Arab", "name": "Arabic"},
  "bn": {"code": "ben_Beng", "name": "Bengali"},
  "ur": {"code": "urd_Arab", "name": "Urdu"},
  "sw": {"code": "swh_Latn", "name": "Swahili"}
}

# @lru_cache(maxsize=64)
def translate_text(text: str, src: str, target: str, max_length: Optional[int] = None) -> str:
  if max_length is None:
    max_length = len(text.split()) * 3 + 50
  tokenizer.src_lang = src
  inputs = tokenizer(
    text, 
    return_tensors="pt", 
    padding=True, 
    truncation=True, 
    max_length=max_length
  ).to(device)

  target_id = tokenizer.convert_tokens_to_ids(target)
  generated_tokens = model.generate(
    **inputs, 
    forced_bos_token_id=target_id, 
    num_beams=5, 
    early_stopping=True, 
    max_length=max_length
  )
  return tokenizer.batch_decode(
    generated_tokens, 
    skip_special_tokens=True
  )[0]

@app.get("/")
def read_root():
  return {"message": "Welcome to the eMedia Translation API"}

@app.get("/health")
@app.get("/health.ico")
def health_check():
  return {"status": "ok", "model": MODEL_NAME, "device": device}


def verify_langs(source: str, targets: List[str]) -> Union[bool, str]:
  if source not in available_languages.keys():
    return False, f"Source language '{source}' is not supported. Available languages: {', '.join(available_languages.keys())}"
  
  for target in targets:
    if target not in available_languages.keys():
      return False, f"Target language '{target}' is not supported. Available languages: {', '.join(available_languages.keys())}"
  
  return True, ""

class TranslateRequest(BaseModel):
  q: Union[str, List[str]] = Field(..., description="Text or list of texts to translate")
  source: str = Field(..., description="Source language code, e.g. 'en'")
  target: Union[str, List[str]] = Field(..., description="Target language code(s), e.g. 'fr' or ['fr', 'de']")
  max_length: Optional[int] = Field(None, description="Maximum length of the translated text")

class TranslationResponse(BaseModel):
  translatedText: dict = Field(..., description="Dictionary with target language codes as keys and list of translated texts as values")

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

  try:
    result = {}
    for target_current in targets_arr:

      target = target_current.lower()

      result[target_current] = []

      for text in text_arr:
        text = text.replace("\\/", "/")
        text = text.strip()
        translation = translate_text(text, src=available_languages[source]["code"], target=available_languages[target]["code"], max_length=req.max_length)
        result[target_current].append(translation.strip())

    return TranslationResponse(translatedText=result)

  except Exception as e:
    raise HTTPException(
      status_code=500, 
      detail=f"Translation error: {str(e)}"
    )
