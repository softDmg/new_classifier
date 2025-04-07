# src/train_model.py

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import joblib
import os

from preprocess import load_bbc_dataset

# Load and clean data
df = load_bbc_dataset()

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["category"], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluation
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/bbc_model.joblib")
joblib.dump(vectorizer, "artifacts/tfidf_vectorizer.joblib")
print("✅ Model + vectorizer saved to /artifacts")