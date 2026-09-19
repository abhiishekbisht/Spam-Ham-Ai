import os
import sys
import json
import pickle
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_and_export_model(csv_path: str = "spamham.csv", artifacts_dir: str = "artifacts"):
    """
    Trains an optimal TF-IDF + Classifier model pipeline for Spam vs Ham text classification,
    evaluates classification metrics, extracts top explainability keywords, and exports artifacts.
    """
    print("🚀 Starting Spam/Ham Model Training Pipeline...")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Standardize columns
    text_col = "Message" if "Message" in df.columns else df.columns[1]
    label_col = "Label" if "Label" in df.columns else df.columns[0]
    
    df = df.dropna(subset=[text_col, label_col])
    
    # Map labels: ham -> 0, spam -> 1
    df["target"] = df[label_col].astype(str).str.strip().str.lower().apply(lambda x: 1 if "spam" in x else 0)
        
    X = df[text_col].astype(str)
    y = df["target"]
    
    print(f"📊 Dataset Loaded: {len(df)} records ({y.sum()} Spam, {len(y) - y.sum()} Ham)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    candidates = [
        ("MultinomialNB", Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')),
            ("clf", MultinomialNB(alpha=0.1))
        ])),
        ("LogisticRegression", Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')),
            ("clf", LogisticRegression(C=5.0, max_iter=1000))
        ]))
    ]
    
    best_pipeline = None
    best_f1 = -1.0
    best_name = ""
    best_metrics = {}
    
    for name, pipeline in candidates:
        print(f"🔍 Fitting pipeline candidate: {name}...")
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        
        print(f"  └─ Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_pipeline = pipeline
            best_name = name
            best_metrics = {
                "accuracy": round(float(acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "confusion_matrix": confusion_matrix(y_test, preds).tolist()
            }
            
    print(f"\n🥇 Winner Model: {best_name} (F1 Score: {best_f1:.4f})")
    
    # Extract top spam keywords for explainability
    tfidf_vectorizer = best_pipeline.named_steps["tfidf"]
    clf = best_pipeline.named_steps["clf"]
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    
    if hasattr(clf, "feature_log_prob_"):
        # Log prob diff (spam vs ham)
        log_prob_diff = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        top_spam_idx = np.argsort(log_prob_diff)[-30:][::-1]
        top_spam_words = feature_names[top_spam_idx].tolist()
    elif hasattr(clf, "coef_"):
        top_spam_idx = np.argsort(clf.coef_[0])[-30:][::-1]
        top_spam_words = feature_names[top_spam_idx].tolist()
    else:
        top_spam_words = ["free", "winner", "prize", "cash", "urgent", "claim", "mobile", "txt", "call"]
        
    os.makedirs(artifacts_dir, exist_ok=True)
    model_file = os.path.join(artifacts_dir, "best_model.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(best_pipeline, f)
        
    metrics_file = os.path.join(artifacts_dir, "model_metrics.json")
    full_metrics = {
        "model_name": best_name,
        "dataset_size": len(df),
        "test_samples": len(y_test),
        "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": best_metrics,
        "top_spam_keywords": top_spam_words
    }
    
    with open(metrics_file, "w") as f:
        json.dump(full_metrics, f, indent=2)
        
    print(f"📦 Model saved to: {model_file}")
    print(f"📊 Metrics saved to: {metrics_file}")
    return full_metrics

if __name__ == "__main__":
    train_and_export_model()

