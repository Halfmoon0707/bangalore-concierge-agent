# Pdf_extractor.py
from docling.document_converter import DocumentConverter
from pathlib import Path

def load_pdf_docling(pdf_path):
    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        if result.document:
            markdown_text = result.document.export_to_markdown()
            return markdown_text
        else:
            print(f"Warning: No content extracted from {pdf_path}")
            return None
    except Exception as e:
        print(f"Error processing PDF with Docling: {e}")
        return None

# Test the function
pdf_path = Path("The_Ultimate_Bangalore_Guide.pdf")  # Update with actual path
pdf_text = load_pdf_docling(pdf_path)
if pdf_text:
    print(f"Extracted text (first 500 characters): {pdf_text[:500]}")
    with open("bangalore_guide.md", "w", encoding="utf-8") as f:
        f.write(pdf_text)
else:
    print("Failed to extract text with Docling")