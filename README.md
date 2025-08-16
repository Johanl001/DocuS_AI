---
title: DocuSense AI
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---
# DocuSense AI – Hugging Face Space

This repo is ready to deploy as a Hugging Face Space (Gradio).

- Entry point: `app.py` (Gradio Blocks UI, exports `demo`)
- System deps: `apt.txt` installs Tesseract OCR
- Python deps: minimal `requirements.txt` for Space; optional `api-requirements.txt` for FastAPI backend
- Python runtime: `runtime.txt` pins Python 3.10

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Optional API server:
```bash
pip install -r api-requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Deploy on Hugging Face

1. Create a new Space (type: Gradio) and push this repo.
2. The Space builder will install `apt.txt`, `requirements.txt`, and use Python version from `runtime.txt`.
3. Optional secrets (for `api.py`): set `OPENROUTER_API_KEY` in Space settings.

Notes:
- Keep `requirements.txt` minimal to avoid build timeouts. Use `api-requirements.txt` locally or in a separate service if you need the API.

`app.py` includes:
- Query tab (demo JSON response)
- OCR tab (upload a PDF to see OCR results). Requires Tesseract, provided via `apt.txt`.

Note: `api.py` contains a FastAPI+LangChain backend prototype that depends on external API keys (OpenRouter) and ChromaDB. It’s not required for the Gradio Space to run. If you wish to expose it, convert the Space to Docker or run it separately.