import logging
import os
import shutil
import tempfile
from typing import List

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="splitwhisper", version="0.1.0")


# Modelo menor = menos CPU e memória (tiny, base, small, medium, large-v3)
MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
# int8_float16 só funciona em GPU; em CPU (ex.: Railway) usar int8
_COMPUTE_DEFAULT = "int8_float16" if DEVICE == "cuda" else "int8"
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", _COMPUTE_DEFAULT)
# beam_size 1 = decoding guloso, mais rápido e leve; 5 = melhor qualidade, mais custo
BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
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
  authorization: str | None = Header(None, alias="Authorization"),
) -> JSONResponse:
  _check_auth(authorization or os.getenv("SPLITWHISPER_AUTH_HEADER"))

  suffix = os.path.splitext(file.filename or "")[1] or ".mp3"
  with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
    tmp_path = tmp.name
    shutil.copyfileobj(file.file, tmp)

  file_size = os.path.getsize(tmp_path)
  log.info("Recebido arquivo %s, %.1f KB", file.filename or "(sem nome)", file_size / 1024)

  try:
    log.info("Iniciando transcrição (modelo=%s, beam_size=%s)...", MODEL_NAME, BEAM_SIZE)
    segments_iter, _info = model.transcribe(
      tmp_path,
      language="pt",
      beam_size=BEAM_SIZE,
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
    log.info(
      "Transcrição concluída: %d segmentos, %d caracteres",
      len(all_segments),
      len(full_text),
    )

    return JSONResponse(
      {
        "text": full_text,
        "segments": all_segments,
      }
    )
  except Exception as e:
    log.exception("Erro na transcrição: %s", e)
    raise
  finally:
    try:
      os.remove(tmp_path)
    except OSError:
      pass


@app.get("/health")
async def health() -> dict:
  return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}

