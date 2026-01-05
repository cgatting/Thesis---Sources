import fitz
import sys
import logging

# Mimic the logic in document_parser.py
def test_parse_pdf(document_path):
    print(f"Testing PDF: {document_path}")
    try:
        doc = fitz.open(document_path)
        print(f"Document opened. Page count: {len(doc)}")
        
        full_text = ""
        for page_num, page in enumerate(doc):
            try:
                print(f"Processing page {page_num}")
                page_text = page.get_text("text")
                full_text += f"\n{page_text}"
            except Exception as e:
                print(f"Failed to extract text from page {page_num}: {e}")
        
        doc.close()
        print("Document closed.")
        
        if not full_text.strip():
            print("No text extracted.")
        else:
            print(f"Extracted {len(full_text)} characters.")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_parse_pdf(r"c:\Users\cgatt\OneDrive\Pictures\Desktop\Thesis 6\Demo_Files\WALTPO-31.4.pdf")
