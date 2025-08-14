import os
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import json

# Configure Tesseract path for Windows (uncomment and set your path)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_pdf_with_ocr(pdf_path):
    """Extracts text from a PDF, performing OCR on scanned pages."""
    doc = fitz.open(pdf_path)
    text_content = ""
    ocr_results = []
    
    print(f"Processing PDF: {pdf_path}")
    print(f"Total pages: {len(doc)}")
    print("-" * 50)
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Try regular text extraction first
        regular_text = page.get_text()
        
        # Check if regular text extraction worked
        if regular_text.strip():
            print(f"Page {page_num + 1}: Regular text extraction successful")
            print(f"Text length: {len(regular_text)} characters")
            print(f"Sample text: {regular_text[:200]}...")
            text_content += regular_text
            ocr_results.append({
                "page": page_num + 1,
                "method": "regular_text",
                "text_length": len(regular_text),
                "sample": regular_text[:200]
            })
        else:
            print(f"Page {page_num + 1}: No text found, attempting OCR...")
            
            # Perform OCR
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img)
            
            if ocr_text.strip():
                print(f"OCR successful! Text length: {len(ocr_text)} characters")
                print(f"OCR sample: {ocr_text[:200]}...")
                text_content += ocr_text
                ocr_results.append({
                    "page": page_num + 1,
                    "method": "ocr",
                    "text_length": len(ocr_text),
                    "sample": ocr_text[:200]
                })
            else:
                print(f"OCR failed - no text detected")
                ocr_results.append({
                    "page": page_num + 1,
                    "method": "failed",
                    "text_length": 0,
                    "sample": ""
                })
        
        print("-" * 30)
    
    return text_content, ocr_results

def test_ocr_on_documents():
    """Test OCR functionality on all documents in the docs folder"""
    doc_path = "docs"
    results = {}
    
    print("=== OCR Testing Tool ===")
    print("This will show you how the OCR feature works on your documents")
    print()
    
    for filename in os.listdir(doc_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(doc_path, filename)
            print(f"\n{'='*60}")
            print(f"Testing OCR on: {filename}")
            print(f"{'='*60}")
            
            try:
                text, ocr_details = extract_text_from_pdf_with_ocr(file_path)
                results[filename] = {
                    "total_text_length": len(text),
                    "pages_processed": len(ocr_details),
                    "ocr_details": ocr_details,
                    "sample_text": text[:500] if text else "No text extracted"
                }
                
                # Summary
                ocr_pages = [d for d in ocr_details if d["method"] == "ocr"]
                regular_pages = [d for d in ocr_details if d["method"] == "regular_text"]
                
                print(f"\n📊 SUMMARY for {filename}:")
                print(f"   Total pages: {len(ocr_details)}")
                print(f"   Pages with regular text: {len(regular_pages)}")
                print(f"   Pages requiring OCR: {len(ocr_pages)}")
                print(f"   Total text extracted: {len(text)} characters")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                results[filename] = {"error": str(e)}
    
    # Save results to file
    with open("ocr_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("OCR testing completed!")
    print("Results saved to: ocr_test_results.json")
    print("You can view the detailed results in that file.")
    
    return results

def show_ocr_capabilities():
    """Show what OCR can do"""
    print("\n🔍 OCR Capabilities:")
    print("1. Extract text from scanned PDF documents")
    print("2. Process images (JPG, PNG) to extract text")
    print("3. Handle handwritten text (with varying accuracy)")
    print("4. Process multi-language documents")
    print("5. Extract text from complex layouts")
    print("\n📝 To see OCR in action:")
    print("1. Run: python test_ocr.py")
    print("2. Check the output to see which pages used OCR")
    print("3. View ocr_test_results.json for detailed analysis")

if __name__ == "__main__":
    show_ocr_capabilities()
    test_ocr_on_documents()
