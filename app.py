import streamlit as st
import pandas as pd
import pdfplumber
import re
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Configure Streamlit page
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="centered"
)

# App title and description
st.title("📄 Resume Category Predictor")
st.markdown("""
Upload your resume in PDF format to predict its job category.
This model analyzes your resume text and matches it with common job categories.
""")

# PDF text extraction
def extract_text(uploaded_file):
    """Extract text from PDF files"""
    with pdfplumber.open(BytesIO(uploaded_file.read())) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages if page.extract_text())

# Text cleaning
def clean_text(text):
    """Basic text preprocessing"""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove special chars
    return ' '.join(text.split())  # Remove extra whitespace

# Model training function
@st.cache_resource  # Cache the trained model
def train_model():
    """Train the model when first needed"""
    # Load your dataset (replace with your actual path)
    df = pd.read_csv("Resume.csv")
    
    # Prepare data
    X = df["Resume_str"]
    y = df["Category"]
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(X)
    
    # Train classifier
    model = LogisticRegression(max_iter=1000)
    model.fit(X_tfidf, y)
    
    return model, vectorizer, y.unique()

# Main app function
def main():
    # File upload section
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        with st.spinner("Analyzing your resume..."):
            try:
                # Train or load model
                model, vectorizer, categories = train_model()
                
                # Process PDF
                raw_text = extract_text(uploaded_file)
                cleaned_text = clean_text(raw_text)
                
                # Make prediction
                features = vectorizer.transform([cleaned_text])
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                
                # Display results
                st.success(f"**Predicted Category:** {prediction}")
                
                # Show confidence scores
                st.subheader("Category Probabilities")
                prob_df = pd.DataFrame({
                    "Category": categories,
                    "Confidence": probabilities
                }).sort_values("Confidence", ascending=False)
                
                st.dataframe(prob_df.style.format({"Confidence": "{:.2%}"}))
                
                # Show processed text (collapsible)
                with st.expander("View processed text"):
                    st.text(cleaned_text[:1500] + "...")  # Show first 1500 chars
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
