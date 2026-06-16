import streamlit as st
import joblib
import os
import re
from src.extractor import get_document_text

# --- 1. KEYWORD RULES (ANCHORS) ---
ANCHOR_RULES = {
    'Financial': ['balance sheet', 'profit and loss', 'fiscal year', 'revenue', 'tax return', 'assets', 'liabilities', 'equity'],
    'Invoice': ['invoice', 'bill to', 'amount due', 'purchase order', 'gst number', 'total amount', 'shipping address', 'payment terms'],
    'Legal': ['court of', 'plaintiff', 'defendant', 'agreement', 'contract', 'jurisdiction', 'article', 'clause', 'hereby', 'witnesseth'],
    'Academic': ['abstract', 'introduction', 'conclusion', 'references', 'citation', 'thesis', 'university', 'journal', 'research paper'],
    'HR': ['offer letter', 'employee handbook', 'payroll', 'recruitment', 'termination', 'interview', 'salary slip', 'resume', 'cv'],
    'Report': ['executive summary', 'quarterly report', 'annual report', 'analysis', 'findings', 'recommendations']
}

def rule_based_classifier(text):
    """Checks if any strong category keywords exist in the raw text directly."""
    text = text.lower()
    for category, keywords in ANCHOR_RULES.items():
        if any(word in text for word in keywords):
            return category
    return None

# --- 2. ML CLEANING LOGIC ---
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text) 
    return " ".join(text.split())

@st.cache_resource
def load_assets():
    model = joblib.load("models/model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="DocTag AI", page_icon="📄")
st.title("📄 DocTag AI")
st.write("Using **Direct Raw Text Extraction + ML Pattern Matching**.")

try:
    model, vectorizer = load_assets()
except Exception as e:
    st.error("Model files not found! Please run train_model.py first.")
    st.stop()

uploaded_file = st.file_uploader("Upload a Document", type=['pdf', 'docx', 'xlsx', 'xls', 'txt'])

if uploaded_file is not None:
    temp_path = os.path.join("temp_upload", uploaded_file.name)
    os.makedirs("temp_upload", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner('Extracting original text content...'):
        raw_text = get_document_text(temp_path)
        
        if raw_text.strip() == "":
            st.error("No readable text found in the file.")
        else:
            # STEP 1: Direct Rule-Based Match
            rule_prediction = rule_based_classifier(raw_text)
            
            # STEP 2: ML Model Pattern Prediction
            processed_text = clean_text(raw_text)
            vectorized = vectorizer.transform([processed_text])
            ml_prediction = model.predict(vectorized)[0]
            
            # FINAL DECISION PIPELINE
            if rule_prediction:
                final_category = rule_prediction
                method = "Keyword Anchor Engine"
            else:
                final_category = ml_prediction
                method = "Machine Learning Probability Pattern"

            st.success(f"### Final Prediction: **{final_category}**")
            st.info(f"Decision Method: **{method}**")
            
            st.markdown("### 📄 Original Extracted Snippet")
            st.caption("Exact raw text data pattern extracted directly from your source file:")
            
            # FIXED LINE: Added placeholder text with collapsed visibility parameter
            st.text_area(
                "Original Raw Text Snippet", 
                value=raw_text[:1500], 
                height=350, 
                disabled=True, 
                label_visibility="collapsed"
            )

    os.remove(temp_path)