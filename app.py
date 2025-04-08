import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pdfplumber
import re
from io import BytesIO
import numpy as np

# Set page config
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide"
)

# Title and description
st.title("📄 Resume Category Classifier")
st.markdown("""
Upload your resume in PDF format to predict its job category.
This model was trained on resume data across multiple industries.
""")

# PDF processing function
def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file"""
    with pdfplumber.open(BytesIO(uploaded_file.read())) as pdf:
        text = " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text

# Text cleaning function
def clean_resume_text(text):
    """Basic text cleaning"""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove special chars
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = ' '.join(text.split())  # Remove extra whitespace
    return text

# Model training function (run this locally first)
def train_model():
    """Train and save the model (run this locally before deployment)"""
    df = pd.read_csv("data/Resume.csv")  # Update path as needed
    
    # Prepare data
    X = df["Resume_str"]
    y = df["Category"]
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(X)
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tfidf, y)
    
    # Save model components
    joblib.dump(model, "model/resume_model.pkl")
    joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
    joblib.dump(y.unique(), "model/categories.pkl")
    
    print("Model trained and saved successfully!")

# Load model components
@st.cache_resource
def load_model():
    """Load trained model components"""
    try:
        model = joblib.load("model/resume_model.pkl")
        vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
        categories = joblib.load("model/categories.pkl")
        return model, vectorizer, categories
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Main app function
def main():
    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF resume", type="pdf")
    
    if uploaded_file:
        # Load model
        model, vectorizer, categories = load_model()
        
        # Process PDF
        with st.spinner("Processing your resume..."):
            try:
                # Extract and clean text
                raw_text = extract_text_from_pdf(uploaded_file)
                cleaned_text = clean_resume_text(raw_text)
                
                # Vectorize and predict
                text_tfidf = vectorizer.transform([cleaned_text])
                prediction = model.predict(text_tfidf)[0]
                probabilities = model.predict_proba(text_tfidf)[0]
                
                # Display results
                st.success(f"**Predicted Category:** {prediction}")
                
                # Show confidence scores
                st.subheader("Prediction Confidence")
                prob_df = pd.DataFrame({
                    "Category": categories,
                    "Probability": probabilities
                }).sort_values("Probability", ascending=False)
                
                st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))
                
                # Show extracted text (collapsible)
                with st.expander("View processed text"):
                    st.text(cleaned_text[:2000] + "...")  # Show first 2000 chars
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# Run locally first to train model
if __name__ == "__main__":
    # Uncomment to train model (run locally first)
    # train_model()
    
    # Run the app
    main()
