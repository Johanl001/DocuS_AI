import gradio as gr
import pytesseract
from PIL import Image
import fitz

def ocr_process(file):
    import os
    if not file:
        return {"error": "No file provided"}
    # Gradio may pass a dict, a tempfile object, or a path string
    path = None
    if isinstance(file, dict):
        path = file.get("name") or file.get("path")
    elif hasattr(file, "name"):
        path = file.name
    elif isinstance(file, str):
        path = file
    if not path or not os.path.exists(path):
        return {"error": "Invalid file input"}
    try:
        doc = fitz.open(path)
        text_content = ""
        ocr_details = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            regular_text = page.get_text()
            if regular_text.strip():
                text_content += regular_text
                ocr_details.append({
                    "page": page_num + 1,
                    "method": "regular_text",
                    "text_length": len(regular_text),
                    "sample": regular_text[:200]
                })
            else:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text.strip():
                    text_content += ocr_text
                    ocr_details.append({
                        "page": page_num + 1,
                        "method": "ocr",
                        "text_length": len(ocr_text),
                        "sample": ocr_text[:200]
                    })
                else:
                    ocr_details.append({
                        "page": page_num + 1,
                        "method": "failed",
                        "text_length": 0,
                        "sample": ""
                    })
        ocr_pages = [d for d in ocr_details if d["method"] == "ocr"]
        regular_pages = [d for d in ocr_details if d["method"] == "regular_text"]
        return {
            "filename": os.path.basename(path),
            "total_pages": len(ocr_details),
            "pages_with_regular_text": len(regular_pages),
            "pages_with_ocr": len(ocr_pages),
            "total_text_length": len(text_content),
            "sample_text": text_content[:1000] if text_content else "No text extracted",
            "ocr_details": ocr_details
        }
    except Exception as e:
        return {"error": str(e)}


def create_interface():
    with gr.Blocks(title="DocuSense AI") as demo:
        gr.Markdown("# 🔍 DocuSense AI - Document Analysis System")

        with gr.Tab("Query Documents"):
            query_input = gr.Textbox(label="Enter your query", placeholder="What is the coverage for medical expenses?")
            submit_btn = gr.Button("Submit Query")
            output = gr.JSON(label="Response")

            def process_query(query):
                return {
                    "decision": "Approved",
                    "amount": "1000",
                    "justification": "Sample response",
                    "clauses_used": ["Sample clause"]
                }

            submit_btn.click(process_query, inputs=query_input, outputs=output)

        with gr.Tab("OCR Demo"):
            gr.Markdown("Upload a PDF to test OCR functionality")
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            ocr_output = gr.JSON(label="OCR Results")
            file_input.change(ocr_process, inputs=file_input, outputs=ocr_output)

    return demo


# Expose for Spaces
demo = create_interface()

if __name__ == "__main__":
    demo.launch()