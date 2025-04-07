# app/app.py

import streamlit as st
import joblib

# Load model + vectorizer
model = joblib.load("artifacts/bbc_model.joblib")
vectorizer = joblib.load("artifacts/tfidf_vectorizer.joblib")

# App title
st.set_page_config(page_title="📰 BBC News Classifier")
st.title("📰 News Article Classifier")
st.markdown("Paste any article text below and I'll tell you what it's about.")

# Input box
user_input = st.text_area("✏️ Article Text", height=300)

if st.button("🔍 Predict Category"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize input
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        proba = model.predict_proba(input_vec).max()

        # Output
        st.success(f"📢 Predicted Category: **{prediction}**")
        st.markdown(f"🔢 Confidence: `{proba * 100:.2f}%`")