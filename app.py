# streamlit_app.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
from io import BytesIO

# Load the Logistic Regression model and TF-IDF vectorizer
MODEL_PATH = "logistic_regression_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# Load the model and vectorizer
with open(MODEL_PATH, "rb") as f:
    clf = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to predict using Logistic Regression
def predict_with_logistic(text):
    # Debugging: Print the extracted text
    st.write("**Extracted Text:**")
    st.write(text[:500] + "...")  # Show only the first 500 characters for brevity

    # Transform text into TF-IDF features
    text_tfidf = vectorizer.transform([text])
    
    # Debugging: Check the shape of the TF-IDF vector
    st.write(f"**TF-IDF Vector Shape:** {text_tfidf.shape}")
    
    # Predict the category
    pred = clf.predict(text_tfidf)
    return pred[0]

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
        st.info("Text successfully extracted from the PDF!")
        
        # Prediction button
        if st.button("Predict"):
            st.info("Processing...")
            prediction = predict_with_logistic(resume_text)
            st.success(f"Predicted Category: **{prediction}**")
else:
    st.warning("Please upload a PDF file to proceed.")

# Footer
st.markdown("---")
st.markdown("Developed by [Your Name]")
