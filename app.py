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

def display_pdf(uploaded_file):
    # Read file as bytes
    pdf_bytes = uploaded_file.read()
    
    # Display PDF preview (2 options)
    col1, col2 = st.columns(2)
    
    # Option 1: PDF viewer (better for multi-page)
    with col1:
        st.subheader("PDF Preview")
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    
    # Option 2: First page image (faster rendering)
    with col2:
        st.subheader("First Page Snapshot")
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            first_page = pdf.pages[0]
            img = first_page.to_image()
            st.image(img.original, caption="First Page", use_column_width=True)

st.title("📄 Resume Classifier")
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    # Display PDF preview
    display_pdf(uploaded_file)
    
    # Process and predict
    with st.spinner("Analyzing resume..."):
        model, vectorizer, categories = load_models()
        text = extract_text(uploaded_file)
        text_clean = re.sub(r'[^\w\s]', ' ', text)
        
        prediction = model.predict(vectorizer.transform([text_clean]))[0]
        
        # Enhanced results display
        st.success(f"**Predicted Category:** {prediction}")
        
        # Show confidence scores
        probabilities = model.predict_proba(vectorizer.transform([text_clean]))[0]
        prob_df = pd.DataFrame({
            "Category": categories,
            "Confidence": probabilities
        }).sort_values("Confidence", ascending=False)
        
        st.subheader("Prediction Confidence")
        st.dataframe(prob_df.style.format({"Confidence": "{:.2%}"}))
        
        # Show processed text
        with st.expander("View extracted text"):
            st.text(text_clean[:2000] + ("..." if len(text_clean) > 2000 else ""))
