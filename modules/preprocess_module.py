"""
Cleans and preprocesses text extracted by OCRManager.
Supporting both English and Malay text.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Just to ensure NLTK resources are downloaded
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

class TextPreprocessor:
    def __init__(self, lang="msa+eng"):
        self.lang = lang
        self.stopwords_en = set(stopwords.words("english"))
        
        stopwords_msa_file = "datasets/malay_stopwords.txt" 
        with open(stopwords_msa_file, 'r', encoding='utf-8') as file:
            self.stopwords_ms = set(line.strip() for line in file if line.strip())
        
        self.stemmer = PorterStemmer()

    # Basic cleaning: remove special chars, punctuation, extra spaces.
    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text) 
        text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.strip()

    # Removes both English and Malay stopwords.
    def remove_stopwords(self, text: str) -> str:
        words = text.split()
        cleaned_words = [
            w for w in words
            if w not in self.stopwords_en and w not in self.stopwords_ms
        ]
        return " ".join(cleaned_words)

    # Applies stemming (English only).
    def stem_words(self, text: str) -> str:
        words = text.split()
        stemmed = [self.stemmer.stem(w) for w in words]
        return " ".join(stemmed)

    # master method
    def preprocess(self, text: str, apply_stemming=False) -> str:
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input text for preprocessing.")

        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        if apply_stemming:
            text = self.stem_words(text)

        return text

'''
# ========== TEST BLOCK ==========
if __name__ == "__main__":
    print("Starting Preprocessing Module Test...\n")

    # Path to your OCR output text file
    ocr_output_path = "datasets/ocr_result.txt"
    preprocessed_output_path = "datasets/preprocess_result.txt"

    try:
        # Initialize preprocessor
        preprocessor = TextPreprocessor(lang="msa+eng")

        # Step 1: Load OCR text
        with open(ocr_output_path, "r", encoding="utf-8") as f:
            original_text = f.read()

        print(f"[INFO] Loaded OCR text ({len(original_text)} characters).")

        # Step 2: Apply cleaning
        cleaned_text = preprocessor.clean_text(original_text)
        print("[TEST] Cleaned text sample:")
        print(cleaned_text[:300], "\n")  # show sample

        # Step 3: Remove stopwords
        no_stopwords = preprocessor.remove_stopwords(cleaned_text)
        print("[TEST] Text after stopword removal sample:")
        print(no_stopwords[:300], "\n")

        # Step 4: Optional stemming 
        stemmed_text = preprocessor.stem_words(no_stopwords)
        print("[TEST] Text after stemming sample:")
        print(stemmed_text[:300], "\n")

        # Step 5: Full pipeline test
        fully_processed = preprocessor.preprocess(original_text, apply_stemming=False)
        print("[TEST] Full preprocessing pipeline sample:")
        print(fully_processed[:300], "\n")

        # Step 6: Save result
        with open(preprocessed_output_path, "w", encoding="utf-8") as f:
            f.write(fully_processed)

        print(f"\nPreprocessing complete — cleaned text overwritten to {preprocessed_output_path}")

    except Exception as e:
        print(f"Preprocessing module test failed: {e}")
'''