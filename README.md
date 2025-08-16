# DocuSense AI – Hugging Face Space

This repo is ready to deploy as a Hugging Face Space (Gradio).

- Entry point: `app.py` (Gradio Blocks UI)
- System deps: `apt.txt` installs Tesseract OCR
- Python deps: `requirements.txt`

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Deploy on Hugging Face

1. Create a new Space (type: Gradio) and push this repo, or connect it.
2. The Space builder will install `apt.txt` and `requirements.txt` automatically.
3. Optionally, set secrets in the Space Settings if you plan to use `api.py`:
   - OPENROUTER_API_KEY

`app.py` includes:
- Query tab (demo JSON response)
- OCR tab (upload a PDF to see OCR results). Requires Tesseract, provided via `apt.txt`.

Note: `api.py` contains a FastAPI+LangChain backend prototype that depends on external API keys (OpenRouter) and ChromaDB. It’s not required for the Gradio Space to run. If you wish to expose it, convert the Space to Docker or run it separately.