# streamlit_app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import PyPDF2
from io import BytesIO
import json

# --------- Load Models and Labels ---------
MODEL_PATH = "distilbert-base-uncased"
LABELS_PATH = "labels.json"

# Load the DistilBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained("path_to_your_finetuned_model")
model.eval()

# Load the labels
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to preprocess text for DistilBERT
def preprocess_text_bert(text):
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    return tokens

# Function to predict using DistilBERT
def predict_with_bert(text):
    inputs = preprocess_text_bert(text)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
    return labels[preds.item()]

# Streamlit App
st.title("Resume Classification App")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload your resume (PDF format):", type=["pdf"])

if uploaded_file is not None:
    # Extract text from the uploaded PDF
    resume_text = extract_text_from_pdf(uploaded_file)

    if resume_text.strip() == "":
        st.error("The uploaded PDF does not contain any text. Please upload a valid PDF.")
    else:
        st.success("Text extracted successfully!")
        st.write("**Extracted Text:**")
        st.write(resume_text[:500] + "...")  # Show only the first 500 characters for brevity
        
        # Prediction button
        if st.button("Predict"):
            st.info("Processing...")
            prediction = predict_with_bert(resume_text)
            st.success(f"Predicted Category: **{prediction}**")
else:
    st.warning("Please upload a PDF file to proceed.")

# Footer
st.markdown("---")
st.markdown("Developed by [Your Name]")
