import os
import uvicorn
import json
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# OCR imports
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define paths
CHROMA_PATH = "chroma_db"

# Pydantic model for input and output
class QueryModel(BaseModel):
    query: str

class ResponseModel(BaseModel):
    decision: str
    amount: str
    justification: str
    clauses_used: List[str]

class OCRResponseModel(BaseModel):
    filename: str
    total_pages: int
    pages_with_regular_text: int
    pages_with_ocr: int
    total_text_length: int
    sample_text: str
    ocr_details: List[Dict[str, Any]]

# Initialize FastAPI
app = FastAPI()

# --- Lazily initialized global resources ---
embeddings = None
vector_db = None
llm = None
rag_chain = None


def initialize_pipeline() -> None:
    """Lazily initialize heavy ML components on first use to keep startup fast."""
    global embeddings, vector_db, llm, rag_chain
    if rag_chain is not None:
        return

    try:
        logger.info("Initializing embeddings and vector store...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            logger.warning("OPENROUTER_API_KEY not set. RAG queries may fail until it's provided.")

        logger.info("Initializing LLM client...")
        llm_client = ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=openrouter_api_key,
            model="openai/gpt-oss-20b:free",
            temperature=0.3,
        )

        logger.info("Setting up retrieval chain...")
        retriever = vector_db.as_retriever()
        retriever_prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
        history_aware_retriever = create_history_aware_retriever(llm_client, retriever, retriever_prompt)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
                "You are an expert claims officer. Your task is to evaluate a claim based on the provided documents. "
                "Your final response must be a JSON object with 'decision', 'amount', 'justification', and 'clauses_used'. "
                "Justify the decision by referencing the specific clauses from the documents. "
                "The decision must be 'Approved', 'Rejected', or 'Approved with conditions'. "
                "The amount should be a numeric value if applicable, otherwise 'N/A'. "
                "Context: {context}"
            ),
            ("user", "{input}")
        ])
        document_chain = create_stuff_documents_chain(llm_client, prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        # Assign to globals last to avoid half-initialized state on failure
        globals()["llm"] = llm_client
        globals()["rag_chain"] = rag_chain

        logger.info("Pipeline initialized successfully.")

    except Exception as e:
        logger.error(f"Error during pipeline initialization: {str(e)}")
        raise


def extract_text_from_pdf_with_ocr(pdf_path):
    """Extracts text from a PDF, performing OCR on scanned pages."""
    doc = fitz.open(pdf_path)
    text_content = ""
    ocr_results = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Try regular text extraction first
        regular_text = page.get_text()
        
        # Check if regular text extraction worked
        if regular_text.strip():
            text_content += regular_text
            ocr_results.append({
                "page": page_num + 1,
                "method": "regular_text",
                "text_length": len(regular_text),
                "sample": regular_text[:200]
            })
        else:
            # Perform OCR
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img)
            
            if ocr_text.strip():
                text_content += ocr_text
                ocr_results.append({
                    "page": page_num + 1,
                    "method": "ocr",
                    "text_length": len(ocr_text),
                    "sample": ocr_text[:200]
                })
            else:
                ocr_results.append({
                    "page": page_num + 1,
                    "method": "failed",
                    "text_length": 0,
                    "sample": ""
                })
    
    return text_content, ocr_results

@app.post("/process_query", response_model=ResponseModel)
async def process_query(query_model: QueryModel):
    """
    Processes a natural language query against the document knowledge base
    and returns a structured JSON response.
    """
    try:
        logger.info(f"Processing query: {query_model.query}")
        
        # Check if query is not empty
        if not query_model.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Ensure pipeline is ready
        initialize_pipeline()

        response = rag_chain.invoke({"input": query_model.query})
        
        llm_output_str = response.get("answer", "{}")
        logger.info(f"LLM raw output: {llm_output_str}")
        
        # --- Robust JSON parsing ---
        try:
            # Try to extract JSON if it's wrapped in markdown code blocks
            if "```json" in llm_output_str:
                start_idx = llm_output_str.find("```json") + 7
                end_idx = llm_output_str.find("```", start_idx)
                if end_idx != -1:
                    llm_output_str = llm_output_str[start_idx:end_idx].strip()
            
            llm_output_json = json.loads(llm_output_str)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Raw LLM output: {llm_output_str}")
            
            # Try to create a fallback response
            fallback_response = {
                "decision": "Error",
                "amount": "N/A",
                "justification": f"Unable to parse LLM response. Raw output: {llm_output_str[:200]}...",
                "clauses_used": []
            }
            
            # Validate the fallback response
            validated_response = ResponseModel(**fallback_response)
            return validated_response

        # Validate required fields
        required_fields = ["decision", "amount", "justification", "clauses_used"]
        missing_fields = [field for field in required_fields if field not in llm_output_json]
        
        if missing_fields:
            logger.warning(f"Missing fields in LLM response: {missing_fields}")
            # Add missing fields with defaults
            for field in missing_fields:
                if field == "clauses_used":
                    llm_output_json[field] = []
                else:
                    llm_output_json[field] = "N/A"

        validated_response = ResponseModel(**llm_output_json)
        logger.info(f"Successfully processed query. Decision: {validated_response.decision}")
        
        return validated_response
    
    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running"""
    return {"status": "healthy", "message": "API is running"}

@app.get("/ocr-demo", response_model=List[OCRResponseModel])
async def ocr_demo():
    """
    Demo endpoint to show OCR functionality on documents in the docs folder.
    This shows you how the OCR feature works during document ingestion.
    """
    try:
        doc_path = "docs"
        results = []
        
        for filename in os.listdir(doc_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(doc_path, filename)
                
                try:
                    text, ocr_details = extract_text_from_pdf_with_ocr(file_path)
                    
                    # Count pages by method
                    ocr_pages = [d for d in ocr_details if d["method"] == "ocr"]
                    regular_pages = [d for d in ocr_details if d["method"] == "regular_text"]
                    
                    result = OCRResponseModel(
                        filename=filename,
                        total_pages=len(ocr_details),
                        pages_with_regular_text=len(regular_pages),
                        pages_with_ocr=len(ocr_pages),
                        total_text_length=len(text),
                        sample_text=text[:500] if text else "No text extracted",
                        ocr_details=ocr_details
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    # Add error result
                    error_result = OCRResponseModel(
                        filename=filename,
                        total_pages=0,
                        pages_with_regular_text=0,
                        pages_with_ocr=0,
                        total_text_length=0,
                        sample_text=f"Error: {str(e)}",
                        ocr_details=[]
                    )
                    results.append(error_result)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in OCR demo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR demo error: {str(e)}")

@app.post("/upload-and-ocr")
async def upload_and_ocr(file: UploadFile = File(...)):
    """
    Upload a PDF file and see OCR in action.
    This endpoint allows you to test OCR on any PDF file.
    """
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        try:
            # Process with OCR
            text, ocr_details = extract_text_from_pdf_with_ocr(temp_path)
            
            # Count pages by method
            ocr_pages = [d for d in ocr_details if d["method"] == "ocr"]
            regular_pages = [d for d in ocr_details if d["method"] == "regular_text"]
            
            result = {
                "filename": file.filename,
                "total_pages": len(ocr_details),
                "pages_with_regular_text": len(regular_pages),
                "pages_with_ocr": len(ocr_pages),
                "total_text_length": len(text),
                "sample_text": text[:1000] if text else "No text extracted",
                "ocr_details": ocr_details
            }
            
            return result
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Error in upload and OCR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload and OCR error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)