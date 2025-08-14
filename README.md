# DocuS_AI

Run the API locally:

- Create a virtualenv and install requirements:
  - `pip install -r requirements.txt`
- Start the server:
  - `uvicorn api:app --host 0.0.0.0 --port 7860`

Build and run with Docker:

- `docker build -f Dokerfile.dockerfile -t docsense-api .`
- `docker run --rm -p 7860:7860 --env-file .env docsense-api`