# DocuS_AI

Run the API locally:

- Create a virtualenv and install requirements:
  - `pip install -r requirements.txt`
- Start the server:
  - `uvicorn api:app --host 0.0.0.0 --port 7860`

Build and run with Docker:

- `docker build -f Dokerfile.dockerfile -t docsense-api .`
- `docker run --rm -p 7860:7860 --env-file .env docsense-api`

## Deploy on Render

### Using Docker (recommended)
- Connect your repository to Render and create a new Web Service.
- Render will detect the `Dockerfile`.
- Set Environment Variables:
  - `OPENROUTER_API_KEY`
  - Optional: `WEB_CONCURRENCY=2`
- Health check path: `/health`
- Optional persistent disk to keep vector DB:
  - Name: `chroma-db`, mount to `/app/chroma_db`, size `1GB`+
- Alternatively, click New -> Blueprint and select `render.yaml` in the repo.

### Native (without Docker)
- Add `apt.txt` with:
  - `tesseract-ocr`
- Build command: `pip install -r requirements.txt`
- Start command:
  - `gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:$PORT api:app`
- Set `OPENROUTER_API_KEY` env var
- Health check path: `/health`