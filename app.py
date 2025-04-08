import streamlit as st
import joblib
import pdfplumber
import re
import base64
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

def show_pdf_preview(uploaded_file):
    """Display first page of PDF as preview"""
    with pdfplumber.open(BytesIO(uploaded_file.read())) as pdf:
        first_page = pdf.pages[0]
        
        # Display preview image
        st.image(first_page.to_image().original, caption="First Page Preview", width=300)
        
        # Display first few lines of text
        preview_text = first_page.extract_text()
        if preview_text:
            with st.expander("Show text preview"):
                st.text(preview_text[:500] + "...")  # Show first 500 chars

st.title("📄 Resume Classifier")
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Resume Preview")
        show_pdf_preview(uploaded_file)
    
    with col2:
        st.subheader("Classification Results")
        model, vectorizer, categories = load_models()
        text = extract_text(uploaded_file)
        text_clean = re.sub(r'[^\w\s]', ' ', text)
        
        prediction = model.predict(vectorizer.transform([text_clean]))[0]
        st.success(f"**Predicted Category:** {prediction}")
        
        # Show confidence scores if available
        try:
            probabilities = model.predict_proba(vectorizer.transform([text_clean]))[0]
            prob_df = pd.DataFrame({
                "Category": categories,
                "Confidence": probabilities
            }).sort_values("Confidence", ascending=False)
            
            st.dataframe(prob_df.style.format({"Confidence": "{:.2%}"}))
        except:
            pass
