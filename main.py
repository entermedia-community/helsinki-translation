from typing import Union, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
# from functools import lru_cache

app = FastAPI(title="eMedia Translation API")

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "facebook/nllb-200-1.3B"
TRANSLATION_PROFILE = "v2"

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


def get_language_id(lang_code: str) -> int:
  lang_code_map = getattr(tokenizer, "lang_code_to_id", None)
  if isinstance(lang_code_map, dict):
    lang_id = lang_code_map.get(lang_code)
    if lang_id is not None:
      return lang_id

  lang_id = tokenizer.convert_tokens_to_ids(lang_code)
  if lang_id is None:
    raise ValueError(f"Unsupported tokenizer language code: {lang_code}")

  unk_token_id = getattr(tokenizer, "unk_token_id", None)
  if unk_token_id is not None and lang_id == unk_token_id:
    raise ValueError(f"Unsupported tokenizer language code: {lang_code}")

  return lang_id

# @lru_cache(maxsize=64)
def translate_text(
  text: str,
  src: str,
  target: str,
  max_length: Optional[int] = None,
  profile: str = TRANSLATION_PROFILE,
) -> str:
  _ = profile
  word_count = max(1, len(text.split()))
  if max_length is None:
    max_length = min(512, max(16, word_count * 4 + 16))
  tokenizer.src_lang = src 
  inputs = tokenizer(
    text, 
    return_tensors="pt", 
    padding=True, 
    truncation=True,
    max_length=max_length
  ).to(device)
  
  target_id = get_language_id(target)

  if word_count <= 2:
    max_new_tokens = 4
  else:
    max_new_tokens = min(64, max(8, word_count * 3 + 6))

  generated_tokens = model.generate(
    **inputs, 
    forced_bos_token_id=target_id, 
    num_beams=5, 
    early_stopping=True, 
    max_new_tokens=max_new_tokens,
    repetition_penalty=1.05
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
        translation = translate_text(
          text,
          src=available_languages[source]["code"],
          target=available_languages[target]["code"],
          max_length=req.max_length,
          profile=TRANSLATION_PROFILE,
        )
        result[target_current].append(translation.strip())

    return TranslationResponse(translatedText=result)

  except Exception as e:
    raise HTTPException(
      status_code=500, 
      detail=f"Translation error: {str(e)}"
    )
