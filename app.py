import streamlit as st
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch
from PyPDF2 import PdfReader
import os
import json

# Load the saved model, tokenizer, and labels
MODEL_PATH = "./resume_classifier_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# Load the labels
with open(f"{MODEL_PATH}/labels.json", "r") as f:
    LABELS = json.load(f)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to classify resume
def classify_resume(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return LABELS[predicted_class]

# Streamlit app
st.title("Resume Classifier")
st.write("Upload a PDF resume and get its category prediction.")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Display the uploaded file name
    st.write(f"Uploaded file: {uploaded_file.name}")
    
    # Extract text from the PDF
    resume_text = extract_text_from_pdf(uploaded_file)
    
    if resume_text.strip() == "":
        st.error("Failed to extract text from the uploaded PDF. Please try another file.")
    else:
        # Classify the resume
        with st.spinner("Classifying the resume..."):
            category = classify_resume(resume_text)
        
        # Display the result
        st.success(f"Predicted Category: **{category}**")
else:
    st.info("Please upload a PDF file to proceed.")
