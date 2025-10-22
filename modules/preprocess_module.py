"""
Cleans and preprocesses text extracted by OCRManager.
Supporting both English and Malay text.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK resources are downloaded (only runs once)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

class TextPreprocessor:
    def __init__(self, lang="msa+eng"):
        """
        Initialize preprocessor with language preference.
        - lang: 'msa+eng' means Malay + English.
        """
        self.lang = lang
        self.stopwords_en = set(stopwords.words("english"))
        
        stopwords_msa_file = "datasets/malay_stopwords.txt"
        with open(stopwords_msa_file, 'r') as file:
            lines = file.read()
        self.stopwords_ms = {lines} 
        
        self.stemmer = PorterStemmer()

    def clean_text(self, text: str) -> str:
        """Basic cleaning: remove special chars, punctuation, extra spaces."""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)  # multiple spaces → single space
        text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.strip()

    def remove_stopwords(self, text: str) -> str:
        """Removes both English and Malay stopwords."""
        words = text.split()
        cleaned_words = [
            w for w in words
            if w not in self.stopwords_en and w not in self.stopwords_ms
        ]
        return " ".join(cleaned_words)

    def stem_words(self, text: str) -> str:
        """Applies stemming (English only, lightweight)."""
        words = text.split()
        stemmed = [self.stemmer.stem(w) for w in words]
        return " ".join(stemmed)

    def preprocess(self, text: str, apply_stemming=False) -> str:
        """
        Full preprocessing pipeline.
        - Cleans, removes stopwords, and optionally stems.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input text for preprocessing.")

        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        if apply_stemming:
            text = self.stem_words(text)

        return text
    
# ========== TEST BLOCK ==========
Processor = TextPreprocessor()
print(Processor.stopwords_ms)
print(Processor.stopwords_en)