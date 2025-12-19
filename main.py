"""
Main pipeline runner for IDOS.
Includes:
- Document validation
- OCR
- Preprocessing
- Categorization
- Summarization
"""

import os

# --- IMPORT MODULES ---
from modules.documentValidator_module import DocumentValidator
from modules.ocr_module import OCRManager
from modules.preprocess_module import TextPreprocessor
from modules.category_module import DocumentCategorizer
from modules.summary_module import TextSummarizer


# ------------------ FILE TYPE DETECTION ------------------
def detect_file_type(path):
    ext = os.path.splitext(path.lower())[1]
    if ext == ".pdf":
        return "pdf"
    return "image"


# ------------------ OCR TEXT EXTRACTION ------------------
def extract_text_from_ocr(ocr_results):
    """
    Supports two OCR output formats:
    1) List of strings (recommended)
    2) List of dicts with "blocks"
    """

    if not ocr_results:
        return ""

    # If OCR returned simple list of strings
    if all(isinstance(p, str) for p in ocr_results):
        return "\n\n".join(p.strip() for p in ocr_results if p.strip())

    all_text = []

    # Handle dict-style (older format)
    for page in ocr_results:
        # If page is string (defensive)
        if isinstance(page, str):
            if page.strip():
                all_text.append(page.strip())
            continue

        # If page contains blocks
        if isinstance(page, dict) and "blocks" in page:
            for block in page["blocks"]:
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    all_text.append(text.strip())
            continue

        # Last fallback
        try:
            as_text = str(page).strip()
            if as_text:
                all_text.append(as_text)
        except:
            pass

    return "\n\n".join(all_text)


# ------------------ MAIN PIPELINE ------------------
def run_pipeline(input_path):
    print("\n===================================")
    print("📌 IDOS FULL PIPELINE INITIALIZED")
    print("===================================\n")

    # --- STEP 0: Validation ---
    print("🛂 STEP 0 — Validating document...")
    validator = DocumentValidator()
    validator.validate(input_path)
    print("   ✔ Document validation passed.\n")

    # --- STEP 1: OCR ---
    print("🔍 STEP 1 — Extracting text (OCR)...")

    ocr = OCRManager(lang="msa+eng")
    file_type = detect_file_type(input_path)

    if file_type == "pdf":
        ocr_results = ocr.process_pdf(input_path)
    else:
        single_page = ocr.process_image(input_path)
        ocr_results = [single_page] if isinstance(single_page, str) else [single_page]

    raw_text = extract_text_from_ocr(ocr_results)

    print("   ✔ OCR completed.")
    print(f"   ✔ Extracted {len(raw_text)} characters.\n")

    # --- STEP 2: Preprocessing ---
    print("🧹 STEP 2 — Preprocessing text...")

    preprocessor = TextPreprocessor(lang="msa+eng")
    cleaned_text = preprocessor.preprocess(raw_text)

    print("   ✔ Preprocessing completed.")
    print(f"   ✔ Cleaned length: {len(cleaned_text)} characters.\n")

    # --- STEP 3: Categorization ---
    print("🗂 STEP 3 — Categorizing document...")

    categorizer = DocumentCategorizer(
        model_type="svm",
        model_path="models/categorizer.pkl"
    )
    
    categorizer.load_model()
    predicted_category = categorizer.predict(cleaned_text)

    print("   ✔ Categorization completed.")
    print(f"   📌 Predicted Category: {predicted_category}\n")

    # --- STEP 4: Summarization ---
    print("📝 STEP 4 — Summarizing document...")

    summarizer = TextSummarizer()

    if "Application & Forms" in predicted_category or "Financial & Procurement" in predicted_category:
        summary = "Summary not applicable for form-based documents."
        print("   ⚠ Summarization skipped for form-type documents.")
    else:
        summary = summarizer.summarize(raw_text, sentence_count=3, randomize=False)
        print("   ✔ Summarization completed.")

    # --- OUTPUT SUMMARY ---
    print("\n===================================")
    print("🎉 PIPELINE RESULT SUMMARY")
    print("===================================\n")

    print("📁 File:", input_path)
    print("📌 Category:", predicted_category)
    print("\n📝 Summary Output:\n")
    print(summary)
    print("\n===================================\n")

    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "category": predicted_category,
        "summary": summary
    }


# ------------------ RUN ------------------
if __name__ == "__main__":
    path = input("Enter the file path (PDF or Image): ").strip()
    run_pipeline(path)