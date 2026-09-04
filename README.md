# 🛡️ Healthcare Data De-Identification Engine

An AI-powered system designed to identify and redact Protected Health Information (PHI) from medical documents, ensuring HIPAA compliance while preserving clinical context.

## 🚀 Features
- **Hybrid Detection**: Combines local NLP (SpaCy) with Multimodal AI (Gemini 1.5).
- **Format Support**: Handles PDF (text & scanned), Images (JPG, PNG), and plain text.
- **Context Preservation**: Smartly avoids redacting medical terms, dosages, and lab values.
- **Reversible Masking**: Securely mask PHI with placeholders and de-mask them using an Admin Vault.
- **Explainability**: Detailed reports on why specific text was identified as PHI.

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.9+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed on your system.
- Google Gemini API Key.

### 2. Installation
```bash
# Clone the repository (if applicable)
# Install dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Running the App
```bash
streamlit run app.py
```

## 📂 Project Structure
- `agents/`: AI Agent logic for advanced PHI detection.
- `engine/`: Core de-identification and vault logic.
- `utils/`: PDF and OCR processing utilities.
- `dataset/`: Sample medical reports for testing.

## 🔒 Security Note
This engine is a tool to assist in de-identification. Always perform a final human review for sensitive clinical data.
