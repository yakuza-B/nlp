import streamlit as st
import joblib
import pdfplumber
import re
from io import BytesIO

# Load small model files
@st.cache_resource
def load_models():
    return (
        joblib.load("model.joblib"),
        joblib.load("vectorizer.joblib"),
        joblib.load("categories.joblib")
    )

def extract_text(uploaded_file):
    with pdfplumber.open(BytesIO(uploaded_file.read())) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages)

st.title("📄 Resume Classifier")
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    model, vectorizer, categories = load_models()
    text = extract_text(uploaded_file)
    text_clean = re.sub(r'[^\w\s]', ' ', text)
    
    prediction = model.predict(vectorizer.transform([text_clean]))[0]
    st.success(f"Predicted Category: {prediction}")
