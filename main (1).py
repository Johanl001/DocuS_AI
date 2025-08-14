# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 12:05:50 2025

@author: prana
"""

"""
LLM-Powered Intelligent Query–Retrieval System
File: main.py (single-file FastAPI app)

Usage:
- Place your environment variables in a .env file (see .env.example below)
- Install requirements from requirements.txt
- Run: uvicorn main:app --reload --port 8000

This file contains:
- Document ingestion (PDF/DOCX/Email URLs)
- Chunking and embeddings using OpenAI
- Pinecone index creation & management (or FAISS fallback)
- /ingest endpoint to index a document
- /hackrx/run endpoint to accept the sample request and return structured JSON
- Clause-level retrieval, simple clause-matching, and GPT-4 based decision reasoning

Notes:
- Replace placeholders with your API keys in environment variables
- This implementation favors clarity and modularity; you can split into modules for production
"""

import os
import io
import json
import math
import uuid
import asyncio
import logging
from typing import List, Optional, Dict, Any

import dotenv
import requests
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel

# Document parsing
from PyPDF2 import PdfReader
import docx
import email
from email import policy as email_policy

# OpenAI & Pinecone
import openai
import pinecone

# Simple in-memory FAISS fallback (optional)
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# Load environment
dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "policy-clauses")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is required in environment")
# No 'openai.api_key' setting needed here for Gemini


# Initialize Pinecone if key present
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists (note: list_indexes() returns an object with .names())
if PINECONE_INDEX not in pc.list_indexes().names():
    # Create index with serverless spec if needed (adjust cloud & region as per your env)
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",    # or your cloud provider
            region=PINECONE_ENV  # e.g., "us-east-1"
        )
    )

pinecone_index = pc.Index(PINECONE_INDEX)


# Constants
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")  # set preferred model
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "5"))

app = FastAPI(title="LLM Retrieval System - FastAPI")
logging.basicConfig(level=logging.INFO)

# ---------------------------
# Pydantic models
# ---------------------------
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class ClauseHit(BaseModel):
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]

class AnswerItem(BaseModel):
    question: str
    answer: str
    decision: Optional[str]
    amount: Optional[Any]
    clauses: List[ClauseHit]

class RunResponse(BaseModel):
    answers: List[AnswerItem]

# ---------------------------
# Utilities: text extraction
# ---------------------------

def extract_text_from_pdf_bytes(b: bytes) -> str:
    reader = PdfReader(io.BytesIO(b))
    pages = []
    for p in reader.pages:
        text = p.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def extract_text_from_docx_bytes(b: bytes) -> str:
    doc = docx.Document(io.BytesIO(b))
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


def extract_text_from_email_bytes(b: bytes) -> str:
    msg = email.message_from_bytes(b, policy=email_policy.default)
    parts = []
    # Prefer plain text parts
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == 'text/plain':
                parts.append(part.get_content())
    else:
        parts.append(msg.get_content())
    return "\n".join([p for p in parts if p])


def download_file_to_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content


def extract_text_from_url(url: str) -> str:
    """Auto-detect file type from URL and extract text."""
    lower = url.lower()
    b = download_file_to_bytes(url)
    if lower.endswith('.pdf') or b[:4] == b"%PDF":
        return extract_text_from_pdf_bytes(b)
    if lower.endswith('.docx'):
        return extract_text_from_docx_bytes(b)
    if lower.endswith('.eml') or b.lstrip().startswith(b"From:"):
        return extract_text_from_email_bytes(b)
    # Fallback: try to decode text
    try:
        return b.decode('utf-8', errors='ignore')
    except Exception:
        return ""

# ---------------------------
# Chunking & Embeddings
# ---------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


#from google.cloud import aiplatform



import requests
import os


HF_API_KEY = os.getenv("HF_API_KEY")
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

def get_hf_embeddings(texts: list[str]) -> list[list[float]]:
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    # Use pipeline endpoint for feature-extraction
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL_ID}"

    response = requests.post(api_url, headers=headers, json={"inputs": texts})
    response.raise_for_status()
    data = response.json()

    # If you get nested token embeddings, average over tokens per text
    embeddings = []
    for item in data:
        # item shape: list of token embeddings (list of lists)
        # average pooling over tokens to get fixed vector per input text
        if isinstance(item[0], list):
            mean_emb = [sum(dim) / len(dim) for dim in zip(*item)]
        else:
            # Single vector (already pooled)
            mean_emb = item
        embeddings.append(mean_emb)
    return embeddings

# ---------------------------
# Storage: Pinecone ops (or local fallback)
# ---------------------------

def upsert_chunks_to_pinecone(chunks: List[str], source: str) -> List[str]:
    ids = []
    embeddings = get_hf_embeddings(chunks)
    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        uid = f"{source}-{uuid.uuid4()}"
        meta = {"text": chunk, "source": source, "chunk_index": i}
        vectors.append((uid, emb, meta))
        ids.append(uid)
    # Pinecone upsert in batches
    if USE_PINECONE and pinecone_index:
        pinecone_index.upsert(vectors=vectors)
    else:
        # For brevity, we don't implement full FAISS server here; production should add fallback
        logging.warning("Pinecone not configured — skipping upsert (demo mode)")
    return ids


def query_pinecone(query: str, top_k: int = TOP_K) -> List[ClauseHit]:
    q_emb = get_hf_embeddings([query])[0]
    results = []
    if USE_PINECONE and pinecone_index:
        res = pinecone_index.query(vector=q_emb, top_k=top_k, include_metadata=True)
        for match in res['matches']:
            hit = ClauseHit(id=match['id'], score=match['score'], text=match['metadata'].get('text',''), metadata=match['metadata'])
            results.append(hit)
    else:
        logging.warning("Pinecone not configured — returning empty results")
    return results

# ---------------------------
# Clause matching and reasoning
# ---------------------------

def build_reasoning_prompt(question: str, clause_hits: List[ClauseHit]) -> str:
    ctx_parts = []
    for i, c in enumerate(clause_hits):
        ctx_parts.append(f"CLAUSE_{i+1} (score={c.score}):\n{c.text}\n---")
    ctx = "\n".join(ctx_parts)

    prompt = f"""
You are an assistant that must answer user insurance/contract queries using ONLY the clauses below.
Give a short decision (Yes/No/Maybe), an amount if determinable (or null), and a precise justification mapping to clause identifiers.
Respond JSON only with keys: decision, amount, justification, clause_mapping (list of clause ids used).

Question: {question}

Clauses:
{ctx}

Rules:
- Use clause identifiers like CLAUSE_1, CLAUSE_2
- If clauses conflict, explain both and give a conservative decision (e.g., Maybe) and explain why
- Keep justification concise but cite exact clause identifiers

Provide output as valid JSON.
"""
    return prompt


def query_hf_qa(context: str, question: str) -> str:
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    api_url = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
    payload = {
        "inputs": {
            "question": question,
            "context": context
        }
    }
    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data.get("answer", "")

# ---------------------------
# API Endpoints
# ---------------------------
@app.post("/ingest")
async def ingest(doc_url: str = Body(..., embed=True)):
    """Download, extract, chunk and index document text."""
    try:
        text = extract_text_from_url(doc_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download or parse: {e}")
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from document")
    chunks = chunk_text(text)
    ids = upsert_chunks_to_pinecone(chunks, source=doc_url)
    return {"status": "indexed", "chunks": len(chunks), "ids_sample": ids[:5]}


@app.post("/hackrx/run", response_model=RunResponse)
async def hackrx_run(payload: RunRequest):
    """Endpoint to accept the hackrx sample request and return structured answers."""
    # Basic: ingest the single document (if not already ingested). For performance, you should persist index state.
    doc_url = payload.documents
    # Ensure we have indexed the doc (idempotent if already present)
    try:
        text = extract_text_from_url(doc_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch doc: {e}")
    chunks = chunk_text(text)
    # Upsert - in production, do duplicate deduplication; here we upsert each run for simplicity
    upsert_chunks_to_pinecone(chunks, source=doc_url)

    answers: List[AnswerItem] = []
    for q in payload.questions:
        # 1. Retrieve top clauses
        clause_hits = query_pinecone(q)
        # 2. Build prompt and call LLM
        prompt = build_reasoning_prompt(q, clause_hits)
        decision_json = query_hf_qa(prompt)

        # Normalize output
        ans_text = decision_json.get('justification') or ''
        decision = decision_json.get('decision')
        amount = decision_json.get('amount')
        # Map clause identifiers to actual clause texts
        used_clause_ids = decision_json.get('clause_mapping', [])
        used_clauses = []
        # Heuristic: clause_mapping contains CLAUSE_1 style strings; map them
        for cm in used_clause_ids:
            try:
                idx = int(cm.split('_')[-1]) - 1
                if 0 <= idx < len(clause_hits):
                    c = clause_hits[idx]
                    used_clauses.append(c)
            except Exception:
                continue

        # If clause_mapping empty, include retrieved clauses as trace
        if not used_clauses:
            used_clauses = clause_hits[:3]

        ai = AnswerItem(
            question=q,
            answer=ans_text,
            decision=decision,
            amount=amount,
            clauses=used_clauses
        )
        answers.append(ai)

    return RunResponse(answers=answers)

# ---------------------------
# Health & basic info
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok", "pinecone": USE_PINECONE}


# ---------------------------
# If run as script
# ---------------------------
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

# ---------------------------
# Additional files (place these alongside main.py)
# ---------------------------

# requirements.txt
requirements = r"""
fastapi
uvicorn[standard]
python-dotenv
requests
openai
pinecone-client
PyPDF2
python-docx
pydantic
faiss-cpu
"""

# .env.example
env_example = r"""
GEMINI_API_KEY=dummy
PINECONE_API_KEY=pcsk_4e3qPi_Hz7C2UZKe9dbynuggyRrjcXtGRs5ENCr1y376Rd4G83bkhUL6b3c8seDUGKJH11
PINECONE_ENV=us-east-1
PINECONE_INDEX=policy-clauses
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o
CHUNK_SIZE=400
CHUNK_OVERLAP=50
TOP_K=5
"""

# README (short)
readme = r"""
LLM Retrieval System - Quick Start
1. Copy main.py and the additional files into a project directory.
2. Create a virtualenv and install requirements: pip install -r requirements.txt
3. Create a .env file from .env.example and fill API keys.
4. Run: uvicorn main:app --reload --port 8000
5. Use POST /hackrx/run with the sample JSON to test. Ensure Pinecone is configured and index created.

Notes:
- For production, add authentication, persistent metadata storage (Postgres), batching, retries, and monitoring.
- Add deduplication when ingesting documents to avoid re-indexing identical content.
"""

# Write helper files to disk for user convenience (if the environment allowed). We don't write them here — they are included as strings above.

print('\n'.join(["Project helper content prepared:", "- main.py (this file)", "- requirements.txt (string variable 'requirements')", "- .env.example (string variable 'env_example')", "- README (string variable 'readme')"]))
