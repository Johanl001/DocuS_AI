import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

import pytesseract
from PIL import Image
import fitz # PyMuPDF
import io

# --- Tesseract Configuration ---
# You need to install Tesseract OCR on your system and provide the path to its executable
# For Windows, uncomment and set the path:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define the paths for your documents and the vector store
DOC_PATH = "docs"
CHROMA_PATH = "chroma_db"

def extract_text_from_pdf_with_ocr(pdf_path):
    """Extracts text from a PDF, performing OCR on scanned pages."""
    doc = fitz.open(pdf_path)
    text_content = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_content += page.get_text()
        
        # If no text was found on the page, try OCR
        if not text_content.strip():
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text_content += pytesseract.image_to_string(img)
    return text_content

def extract_text_from_image_with_ocr(image_path):
    """Extracts text from an image using OCR."""
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)

def is_text_valid(text):
    """Simple check to ensure extracted text is not empty or gibberish."""
    return len(text.strip()) > 50

def ingest_documents():
    """
    Loads, splits, and embeds documents from the 'docs' folder into a ChromaDB vector store.
    """
    documents = []
    for filename in os.listdir(DOC_PATH):
        file_path = os.path.join(DOC_PATH, filename)
        
        extracted_text = ""
        if filename.endswith(".pdf"):
            extracted_text = extract_text_from_pdf_with_ocr(file_path)
        elif filename.endswith((".docx", ".txt")):
            # Use original loaders for non-image files
            if filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            else:
                loader = TextLoader(file_path)
            docs_from_loader = loader.load()
            if docs_from_loader:
                extracted_text = docs_from_loader[0].page_content
        elif filename.endswith((".jpg", ".jpeg", ".png")):
            extracted_text = extract_text_from_image_with_ocr(file_path)
            
        # Quality check: only proceed if the text is valid
        if extracted_text and is_text_valid(extracted_text):
            documents.append({"page_content": extracted_text, "metadata": {"source": filename}})
            
    if not documents:
        print("No valid documents found to ingest. Please check the 'docs' folder.")
        return

    print(f"Loaded {len(documents)} documents.")
    
    # Rest of the chunking and embedding process
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.create_documents([d["page_content"] for d in documents], [d["metadata"] for d in documents])
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Cleared existing data from {CHROMA_PATH}.")

    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    print(f"Successfully created a new ChromaDB instance at {CHROMA_PATH}.")

if __name__ == "__main__":
    ingest_documents()