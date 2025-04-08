# app.py (Uses minimal sample data)
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Minimal sample data (replace with your actual categories)
categories = ["IT", "Business", "Engineering"]
sample_texts = [
    "software developer python java",
    "financial analyst accounting",
    "mechanical engineer CAD"
]

@st.cache_resource
def train_model():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sample_texts)
    y = np.array([0, 1, 2])  # Mock labels
    model = LogisticRegression().fit(X, y)
    return model, vectorizer

st.title("📄 Resume Classifier")
uploaded_file = st.file_uploader("Upload text file", type="txt")

if uploaded_file:
    model, vectorizer = train_model()
    text = uploaded_file.read().decode("utf-8")
    prediction = model.predict(vectorizer.transform([text]))[0]
    st.success(f"Predicted: {categories[prediction]}")
