import gradio as gr
from api import app as fastapi_app
try:
    from asgiref.wsgi import AsgiToWsgi
    app = AsgiToWsgi(fastapi_app)
except Exception:
    app = fastapi_app

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
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()