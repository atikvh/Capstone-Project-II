# IDOS – Quick Run Guide

This guide explains how to run the Intelligent Document Organization System (IDOS) on your local machine.

---

## 1. Prerequisites

Before running the system, ensure the following are installed:

### Software
- Python **3.9 or later**
- **Tesseract OCR**
- **Poppler** (required for PDF processing)

### Tesseract OCR Installation
- **Windows:**  
  Download from https://github.com/UB-Mannheim/tesseract/wiki  
  Make sure Tesseract is added to system PATH.
- **macOS:**  
  `brew install tesseract`
- **Linux:**  
  `sudo apt install tesseract-ocr`

### Poppler Installation
- **Windows:**  
  Download Poppler binaries and add to system PATH.
- **macOS:**  
  `brew install poppler`
- **Linux:**  
  `sudo apt install poppler-utils`

---

## 2. Activate Virtual Environment

From the project root directory:

**Windows:**
venv\Scripts\activate

**macOS/Linux:**
source venv/bin/activate

---

## 3. Download Python Libraries

Required Python libraries must be installed using:
pip install -r requirements.txt 

---

## 4. Run the Application (GUI)

streamlit run app.py

---

## 5. How to Use the System

- Upload a PDF or image file (.pdf, .jpg, .jpeg, .png)
- The system will: 
    -> Validate the file
    -> Extract text using OCR
    -> Automatically classify the document
- The predicted category will be displayed
- If the document is suitable, click "Generate Summary"
- Optionally generate an alternative summary

Notes:
* The classification model is pre-trained and included
* No additional configuration is required
* Summarization is disabled for form-based documents