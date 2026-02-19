import os
import shutil
import tempfile
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel


app = FastAPI(title="splitwhisper", version="0.1.0")


MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
# int8_float16 só funciona em GPU; em CPU (ex.: Railway) usar int8
_COMPUTE_DEFAULT = "int8_float16" if DEVICE == "cuda" else "int8"
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", _COMPUTE_DEFAULT)
AUTH_TOKEN = os.getenv("SPLITWHISPER_AUTH_TOKEN", "").strip() or None


model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)


def _check_auth(header_value: str | None) -> None:
  if AUTH_TOKEN is None:
    return
  if not header_value or not header_value.startswith("Bearer "):
    raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
  token = header_value.removeprefix("Bearer ").strip()
  if token != AUTH_TOKEN:
    raise HTTPException(status_code=403, detail="Invalid token")


@app.post("/transcribe")
async def transcribe(
  file: UploadFile = File(...),
  authorization: str | None = None,
) -> JSONResponse:
  _check_auth(authorization or os.getenv("SPLITWHISPER_AUTH_HEADER"))

  suffix = os.path.splitext(file.filename or "")[1] or ".mp3"
  with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
    tmp_path = tmp.name
    shutil.copyfileobj(file.file, tmp)

  try:
    segments_iter, _info = model.transcribe(
      tmp_path,
      language="pt",
      beam_size=5,
      word_timestamps=False,
    )

    all_segments: List[dict] = []
    text_parts: List[str] = []
    for seg in segments_iter:
      text = (seg.text or "").strip()
      text_parts.append(text)
      all_segments.append(
        {
          "start": float(seg.start),
          "end": float(seg.end),
          "text": text,
        }
      )

    full_text = " ".join(t for t in text_parts if t).strip()

    return JSONResponse(
      {
        "text": full_text,
        "segments": all_segments,
      }
    )
  finally:
    try:
      os.remove(tmp_path)
    except OSError:
      pass


@app.get("/health")
async def health() -> dict:
  return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}

