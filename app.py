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

def extract_text(file_bytes):
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages if page.extract_text())

def show_pdf_preview(file_bytes):
    """Display first page of PDF as preview"""
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        first_page = pdf.pages[0]
        
        # Display preview image
        st.image(first_page.to_image().original, 
                caption="First Page Preview", 
                use_column_width=True)
        
        # Display first few lines of text
        preview_text = first_page.extract_text() or "No text found on first page"
        st.text_area("Text Preview", 
                    value=preview_text[:300] + ("..." if len(preview_text) > 300 else ""),
                    height=150)

# Configure page
st.set_page_config(page_title="Resume Classifier", layout="centered")
st.title("📄 Resume Category Classifier")
st.write("Upload your resume PDF to predict its job category")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # Read file bytes once
    file_bytes = uploaded_file.read()
    
    # Create two columns
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Resume Preview")
        show_pdf_preview(file_bytes)
    
    with col2:
        st.subheader("Analysis Results")
        with st.spinner("Analyzing resume content..."):
            try:
                model, vectorizer, categories = load_models()
                text = extract_text(file_bytes)
                text_clean = re.sub(r'[^\w\s]', ' ', text)
                
                prediction = model.predict(vectorizer.transform([text_clean]))[0]
                
                # Display prediction with nice styling
                st.markdown(f"""
                **Predicted Category:**  
                <span style="font-size: 24px; color: #2e86de">{prediction}</span>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
