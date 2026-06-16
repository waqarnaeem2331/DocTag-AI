import pandas as pd
import re
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "master_train_data.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
VECTORIZER_SAVE_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

# These words cause the HR bias. We remove them during training and prediction.
BIAS_WORDS = {'the', 'and', 'this', 'that', 'with', 'from', 'company', 'dated', 'subject', 'page', 'office', 'dear', 'sincerely', 'regards', 'document', 'of', 'to', 'in', 'for', 'on', 'is', 'as', 'it', 'by', 'an', 'be', 'at', 'which', 'been'}

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text) 
    words = text.split()
    filtered = [w for w in words if w not in BIAS_WORDS]
    return " ".join(filtered)

def train():
    print("🚀 Loading Raw Data...")
    df = pd.read_csv(DATA_PATH).fillna("") # Fill NaNs immediately
    
    # Apply cleaning on the fly
    df['text'] = df['text'].astype(str).apply(clean_text)
    df = df[df['text'].str.strip() != ""] # Remove empty rows

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, random_state=42, stratify=df['category'])

    print("🛠️ Vectorizing (Removing Noise)...")
    # max_df=0.6: ignores words that appear in >60% of documents (Kills the HR Bias)
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2), max_df=0.6, min_df=2)
    
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("🧠 Training High-Precision Model...")
    model = LinearSVC(class_weight='balanced', random_state=42)
    model.fit(X_train_tfidf, y_train)

    print(f"\n🎯 Accuracy: {accuracy_score(y_test, model.predict(X_test_tfidf)):.2%}")
    print(classification_report(y_test, model.predict(X_test_tfidf)))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(tfidf, VECTORIZER_SAVE_PATH)
    print("\n✅ Model and Vectorizer saved!")

if __name__ == "__main__":
    train()
