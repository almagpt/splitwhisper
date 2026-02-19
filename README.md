# splitwhisper

Servidor de transcrição para vídeos longos usando Faster-Whisper.

Pensado para rodar como um serviço separado (por exemplo, no Railway) e ser consumido
pelo worker de vídeos longos do SplitReact.

## Endpoint

- `POST /transcribe`
  - Body: `multipart/form-data` com o campo `file` contendo o áudio (mp3, m4a, wav, etc.).
  - Retorno:

  ```json
  {
    "text": "transcrição inteira...",
    "segments": [
      { "start": 0.0, "end": 3.5, "text": "..." },
      { "start": 3.5, "end": 10.2, "text": "..." }
    ]
  }
  ```

- `GET /health` — retorna `{ status: "ok", model, device }`.

## Variáveis de ambiente

- `WHISPER_MODEL` — nome do modelo Faster-Whisper (ex.: `small`, `medium`, `large-v3`). Padrão: `large-v3`.
- `WHISPER_DEVICE` — `cpu` ou `cuda`. No Railway use `cpu`. Padrão: `cpu`.
- `WHISPER_COMPUTE_TYPE` — tipo de compute. Em CPU o padrão é `int8`; em CUDA é `int8_float16`. Não é necessário definir no Railway.
- `SPLITWHISPER_AUTH_TOKEN` — opcional. Se definido, o servidor exige `Authorization: Bearer <TOKEN>` nas requisições de `/transcribe`.
- `HF_TOKEN` — opcional. Token do Hugging Face para rate limits melhores ao baixar o modelo (evita o warning no primeiro deploy).

## Rodando localmente

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export WHISPER_MODEL=medium
export WHISPER_DEVICE=cpu

uvicorn server:app --host 0.0.0.0 --port 8000
```

Testar com `curl`:

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@exemplo.mp3"
```

Se usar `SPLITWHISPER_AUTH_TOKEN`:

```bash
export SPLITWHISPER_AUTH_TOKEN=seu-token

curl -X POST http://localhost:8000/transcribe \
  -H "Authorization: Bearer seu-token" \
  -F "file=@exemplo.mp3"
```

