import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from sentence_transformers import SentenceTransformer
import warnings
import re

# Suppress messy warnings
warnings.filterwarnings("ignore")

def preprocess_text(text: str) -> str:
    """Clean and normalize comment text before embedding."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # Remove URLs
    text = re.sub(r"\s+", " ", text).strip()        # Normalize whitespace
    return text

def train_model(data_path: str):
    print("Loading Dataset...")
    df = pd.read_csv(data_path)
    
    # Expected columns: "text", "label" (0=human, 1=bot)
    df["clean_text"] = df["text"].fillna("").apply(preprocess_text)
    df = df[df["clean_text"].str.len() > 2] # Drop practically empty rows

    print(f"Dataset active rows: {len(df)}")

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("Loading LLM Embedding Model (all-MiniLM-L6-v2)...")
    # This is a highly efficient 384-dimensional dense semantic embedding model.
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Encoding Training Data into Semantic Vectors (This may take a moment)...")
    X_train_vec = embedder.encode(X_train_text, show_progress_bar=True)
    
    print("Encoding Testing Data...")
    X_test_vec  = embedder.encode(X_test_text, show_progress_bar=True)

    print("Training Multi-Layer Perceptron (Neural Network) Classifier...")
    # Using an MLP to find complex non-linear semantic bot patterns in the dense vectors
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64), 
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=300,
        random_state=42,
        early_stopping=True
    )
    clf.fit(X_train_vec, y_train)

    print("Evaluating Model...")
    y_pred = clf.predict(X_test_vec)
    print(classification_report(y_test, y_pred, target_names=["Human", "Bot"]))

    # Save the Neural Network
    joblib.dump(clf, "models/llm_mlp_model.pkl")
    print("✅ Model trained and saved as 'models/llm_mlp_model.pkl'.")
    print("Note: The SentenceTransformer 'all-MiniLM-L6-v2' does not need to be saved to a pkl, it is loaded natively via the library.")

if __name__ == "__main__":
    train_model("data/twibot22_sample.csv")
