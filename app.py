import streamlit as st
import joblib
import pdfplumber
import re
import base64
import pandas as pd
from io import BytesIO

# Load small model files
@st.cache_resource
def load_models():
    return (
        joblib.load("model.joblib"),
        joblib.load("vectorizer.joblib"),
        joblib.load("categories.joblib")
    )

def extract_text(file_bytes):
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages if page.extract_text())

def show_pdf_preview(file_bytes):
    """Display first page of PDF as preview"""
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        first_page = pdf.pages[0]
        
        # Display preview image
        st.image(first_page.to_image().original, caption="First Page Preview", width=300)
        
        # Display first few lines of text
        preview_text = first_page.extract_text() or "No text found on first page"
        with st.expander("Show text preview"):
            st.text(preview_text[:500] + ("..." if len(preview_text) > 500 else ""))

st.title("📄 Resume Classifier")
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    # Read file bytes once and store in memory
    file_bytes = uploaded_file.read()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Resume Preview")
        show_pdf_preview(file_bytes)
    
    with col2:
        st.subheader("Classification Results")
        model, vectorizer, categories = load_models()
        
        # Process full document
        with st.spinner("Analyzing resume..."):
            try:
                text = extract_text(file_bytes)
                text_clean = re.sub(r'[^\w\s]', ' ', text)
                
                prediction = model.predict(vectorizer.transform([text_clean]))[0]
                st.success(f"**Predicted Category:** {prediction}")
                
                # Show confidence scores if available
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(vectorizer.transform([text_clean]))[0]
                    prob_df = pd.DataFrame({
                        "Category": categories,
                        "Confidence": probabilities
                    }).sort_values("Confidence", ascending=False)
                    
                    st.dataframe(prob_df.style.format({"Confidence": "{:.2%}"}))
                
                # Show full processed text (collapsible)
                with st.expander("View full processed text"):
                    st.text(text_clean[:2000] + ("..." if len(text_clean) > 2000 else ""))
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
