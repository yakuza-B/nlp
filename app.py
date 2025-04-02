# streamlit_app.py
import streamlit as st
import pandas as pd
import pickle
import os

# Paths to model and vectorizer
MODEL_PATH = "logistic_regression_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# Check if files exist
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
else:
    st.success("Model file found!")

if not os.path.exists(VECTORIZER_PATH):
    st.error(f"Vectorizer file not found at: {VECTORIZER_PATH}")
else:
    st.success("Vectorizer file found!")

# Load the model and vectorizer
try:
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

try:
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    st.success("Vectorizer loaded successfully!")
except Exception as e:
    st.error(f"Error loading vectorizer: {e}")

# Function to predict using Logistic Regression
def predict_with_logistic(text):
    text_tfidf = vectorizer.transform([text])
    pred = clf.predict(text_tfidf)
    return pred[0]

# Streamlit App
st.title("Resume Classification App")

# Text input for resume
resume_text = st.text_area("Paste your resume text here:", height=300)

# Prediction button
if st.button("Predict"):
    if resume_text.strip() == "":
        st.error("Please enter some text in the resume box.")
    else:
        st.info("Processing...")
        prediction = predict_with_logistic(resume_text)
        st.success(f"Predicted Category: **{prediction}**")

# Footer
st.markdown("---")
st.markdown("Developed by [Your Name]")
