import streamlit as st
from PIL import Image
import tempfile
import os
import random
import nltk

# attempt to ensure punkt tokenizer is available for sentence splitting
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# import your modules (adjust paths if your modules are located elsewhere)
from modules.documentValidator_module import DocumentValidator
from modules.ocr_module import OCRManager
from modules.preprocess_module import TextPreprocessor
from modules.category_module import DocumentCategorizer
from modules.summary_module import TextSummarizer


# ---------------------------
# UI Helper Methods
# ---------------------------
def extract_text_from_ocr(ocr_results):
    """
    Accepts ocr_results which may be:
      - list of strings (recommended minimal OCR)
      - list of dicts with a 'text' or 'blocks' field (legacy)
    Returns a single string with page breaks preserved.
    """
    if not ocr_results:
        return ""

    # If it's already list of strings
    if all(isinstance(p, str) for p in ocr_results):
        return "\n\n".join(p.strip() for p in ocr_results if p.strip())

    all_text = []
    for page in ocr_results:
        # simple string page
        if isinstance(page, str) and page.strip():
            all_text.append(page.strip())
            continue

        # dict with direct 'text'
        if isinstance(page, dict) and "text" in page and isinstance(page["text"], str):
            t = page["text"].strip()
            if t:
                all_text.append(t)
            continue

        # dict with blocks
        if isinstance(page, dict) and "blocks" in page and isinstance(page["blocks"], list):
            for block in page["blocks"]:
                if isinstance(block, dict):
                    txt = block.get("text", "")
                    if isinstance(txt, str) and txt.strip():
                        all_text.append(txt.strip())
            continue

        try:
            s = str(page).strip()
            if s:
                all_text.append(s)
        except Exception:
            pass

    return "\n\n".join(all_text)


def alt_summary_from_text(text, ratio=0.25, max_sentences=5):
    """
    Fallback summarization that builds an alternative "summary" by sampling sentences.
    Not as advanced as TextRank but provides a different view.
    ratio: fraction of sentences to include (if small doc)
    max_sentences: upper cap of sentences returned
    """
    if not text or not isinstance(text, str):
        return ""

    # split into sentences
    sentences = nltk.tokenize.sent_tokenize(text)
    if not sentences:
        return ""

    # Decide number of sentences for summary
    k = max(1, min(max_sentences, int(len(sentences) * ratio)))
    k = min(k, len(sentences))

    # If doc is very short just return the first sentence(s)
    if len(sentences) <= k:
        return " ".join(sentences)

    # Randomly sample sentences (but keep original order for readability)
    sampled = sorted(random.sample(range(len(sentences)), k))
    chosen = [sentences[i] for i in sampled]
    return " ".join(chosen)


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="IDOS Document Analyzer", layout="wide")
st.title("Integerated Document Organization System")
st.write("Upload a PDF or Image to extract text, classify, and optionally summarize.")

if "summary_generated" not in st.session_state:
    st.session_state.summary_generated = False

if "alt_generated" not in st.session_state:
    st.session_state.alt_generated = False

if "_last_summary" not in st.session_state:
    st.session_state["_last_summary"] = None

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

uploaded_file = st.file_uploader("Choose a PDF or image (jpg/png/pdf)", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    # save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # validate
    validator = DocumentValidator()
    try:
        validator.validate(file_path)
    except Exception as e:
        st.error(f"Validation failed: {e}")
        st.stop()

    col_left, col_right = st.columns([1.2, 1])
    
    # Reset state ONLY if a new file is uploaded 
    if uploaded_file.name != st.session_state.last_uploaded_file:
        st.session_state.summary_generated = False
        st.session_state.alt_generated = False
        st.session_state["_last_summary"] = None
        st.session_state.last_uploaded_file = uploaded_file.name

    # LEFT: preview
    with col_left:
        st.subheader("📘 Document Preview")
        try:
            if uploaded_file.type == "application/pdf":
                from pdf2image import convert_from_path
                pages = convert_from_path(file_path)
                num_pages = len(pages)
                if num_pages > 0:
                    page_number = st.slider("Select page", 1, num_pages, 1)
                    st.image(pages[page_number -1], use_container_width=True)
                else:
                    st.info("PDF uploaded - image preview may not show. The document will still be processed.")
            else:
                img = Image.open(file_path)
                st.image(img, use_container_width=True)
        except Exception:
            st.info("Unable to preview document")

    # RIGHT: processing
    with col_right:
        st.subheader("⚙ Document Analysis")

        # OCR
        with st.spinner("Scanning document..."):
            ocr = OCRManager(lang="msa+eng")
            if uploaded_file.type == "application/pdf":
                ocr_results = ocr.process_pdf(file_path)
            else:
                # process_image might return dict or string depending on your module
                one = ocr.process_image(file_path)
                # ensure list shape
                ocr_results = [one] if not isinstance(one, list) else one

        raw_text = extract_text_from_ocr(ocr_results)

        # Preprocess
        preprocessor = TextPreprocessor(lang="msa+eng")
        cleaned_text = preprocessor.preprocess(raw_text)

        # Categorize
        categorizer = DocumentCategorizer(model_type="svm", model_path="models/categorizer.pkl")
        try:
            categorizer.load_model()
        except Exception as e:
            st.error(f"Failed to load categorization model: {e}")
            st.stop()

        predicted_category = categorizer.predict(cleaned_text)
        st.write("### 📝 Predicted Category")
        st.success(predicted_category)

        # Summarization controls
        st.write("### 📝 Content Takeaways")

        summarizer = TextSummarizer()
        STRUCTURED = ["Application & Forms", "Financial & Procurement"]

        if any(cat in predicted_category for cat in STRUCTURED):
            with st.spinner("Extracting key information..."):
                key_info = summarizer.extract_key_information(
                    raw_text, category=predicted_category
                )
            st.text(key_info)

        else:
            if st.session_state["_last_summary"]:
                st.subheader("Summary")
                st.write(st.session_state["_last_summary"])
            # STATE 1 — Initial: Generate Summary
            if not st.session_state.summary_generated and not st.session_state.alt_generated:
                if st.button("📝 Generate Summary"):
                    with st.spinner("Generating summary..."):
                        try:
                            primary = summarizer.summarize(
                                raw_text,
                                sentence_count=3,
                                randomize=False
                            )
                        except Exception:
                            primary = alt_summary_from_text(raw_text)

                        st.session_state["_last_summary"] = primary
                        st.session_state.summary_generated = True
                    st.rerun()

            # STATE 2 — After summary: Generate Another Version
            elif st.session_state.summary_generated and not st.session_state.alt_generated:
                if st.button("🔄 Generate Another Version"):
                    with st.spinner("Generating alternative summary..."):
                        try:
                            alt = summarizer.summarize(raw_text, randomize=True)
                        except Exception:
                            alt = alt_summary_from_text(
                                raw_text, ratio=0.25, max_sentences=5
                            )

                        st.session_state["_last_summary"] = alt
                        st.session_state.alt_generated = True
                    st.rerun()

            elif st.session_state.summary_generated and st.session_state.alt_generated:
                st.info("Summary generation completed.")


    # cleanup temp file
    try:
        os.remove(file_path)
    except Exception:
        pass