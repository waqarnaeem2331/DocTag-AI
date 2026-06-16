import pandas as pd
import os

column_mapping = {
    'hr.csv': 'Sentence', 'financial.csv': 'text', 'spam_messages.csv': 'v2', 
    'academic_wiki.parquet': 'text', 'legal_documents.csv': 'case_text', 'report.csv': 'text'
}
label_mapping = {
    'hr.csv': 'HR', 'financial.csv': 'Financial', 'spam_messages.csv': 'Marketing',
    'academic_wiki.parquet': 'Academic', 'legal_documents.csv': 'Legal', 'report.csv': 'Report'
}

SAMPLES_PER_CATEGORY = 2000 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "master_train_data.csv")

all_data = []
for file_name, preferred_col in column_mapping.items():
    f_path = os.path.join(RAW_PATH, file_name)
    if not os.path.exists(f_path): continue
    
    try:
        if file_name.endswith('.csv'):
            # Fix for financial.csv header
            df = pd.read_csv(f_path, encoding='latin-1', header=None if file_name=='financial.csv' else 'infer', nrows=10000)
            text_col = 0 if file_name == 'financial.csv' else preferred_col
        elif file_name.endswith('.parquet'):
            df = pd.read_parquet(f_path).head(10000)
            text_col = preferred_col
        else: continue

        if not isinstance(text_col, int) and text_col not in df.columns:
            possible_cols = [c for c in df.columns if 'text' in c.lower() or 'message' in c.lower() or 'sentence' in c.lower()]
            text_col = possible_cols[0] if possible_cols else None
            if text_col is None: continue

        temp = df.iloc[:, text_col].to_frame() if isinstance(text_col, int) else df[[text_col]].copy()
        temp.columns = ['text']
        temp['category'] = label_mapping[file_name]
        
        if len(temp) > SAMPLES_PER_CATEGORY:
            temp = temp.sample(n=SAMPLES_PER_CATEGORY, random_state=42)
        all_data.append(temp)
        print(f"✅ Collected {file_name}")
    except Exception as e: print(f"❌ Error {file_name}: {e}")

if all_data:
    master_df = pd.concat(all_data, axis=0).sample(frac=1).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    master_df.to_csv(OUT_PATH, index=False)
    print(f"\n🚀 Raw Master Dataset Created at: {OUT_PATH}")
