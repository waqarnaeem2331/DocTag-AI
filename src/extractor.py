# This file is created automatically
import pdfplumber
from docx import Document
import pandas as pd
import os

def extract_text_from_pdf(file_path):
    """Extracts all text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def extract_text_from_docx(file_path):
    """Extracts all text from a Word (.docx) file."""
    try:
        doc = Document(file_path)
        # Join all paragraphs with a newline
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        text = ""
    return text

def extract_text_from_excel(file_path):
    """Extracts all text from an Excel (.xlsx) file by converting sheets to strings."""
    try:
        # Read all sheets
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        combined_text = ""
        for sheet_name, df in all_sheets.items():
            # Convert the dataframe to a string to capture all data
            combined_text += df.to_string() + "\n"
    except Exception as e:
        print(f"Error reading Excel {file_path}: {e}")
        combined_text = ""
    return combined_text

def extract_text_from_txt(file_path):
    """Extracts text from a simple .txt file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
        text = ""
    return text

def get_document_text(file_path):
    """
    Main function to route the file to the correct extractor based on extension.
    """
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif extension == '.docx':
        return extract_text_from_docx(file_path)
    elif extension in ['.xlsx', '.xls']:
        return extract_text_from_excel(file_path)
    elif extension == '.txt':
        return extract_text_from_txt(file_path)
    else:
        print(f"Unsupported file extension: {extension}")
        return ""

# ==========================================
# TESTING BLOCK (Optional)
# ==========================================
if __name__ == "__main__":
    # To test this, place a sample file in your folder and put the path here
    test_file = "test_document.pdf" 
    if os.path.exists(test_file):
        content = get_document_text(test_file)
        print("--- Extracted Text ---")
        print(content[:500]) # Print first 500 characters
    else:
        print(f"Please provide a valid file path to test. {test_file} not found.")
