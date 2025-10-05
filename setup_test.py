# checking the activation of environment for IDOS

# 1. OCR
print("Testing OCR (Tesseract + Pytesseract)...")
import pytesseract
from PIL import Image, ImageDraw
# image to scan
img = Image.new("RGB", (200,60), color = "white")
d = ImageDraw.Draw(img)
d.text((10,10), "Hello  World", fill="black")
# Test OCR
ocr_text = pytesseract.image_to_string(img)
print("OCR result:", ocr_text.strip())

# 2. Preprocessing Text
print("\n Testing NLTK preprocessing...")
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
print("Sample of stopwords:", stopwords.words("malay")[:10])

# 3. Categorization Test (TF-IDF + SVM)
print("\n Testing categorization with scikit-learn...")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
# dummy dataset
docs = ["This is a memo about work",
        "Leave application form submitted",
        "Official approval letter from HR"]
labels = ["Memo", "Form", "Letter"]
# Text to numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
# Train a simple classifier
clf = LinearSVC()
clf.fit(X, labels)
# Test on new data
test_doc = ["Please read this HR memo"]
prediction = clf.predict(vectorizer.transform(test_doc))
print("Categorization result:", prediction[0])

# 4. Summarization test (Textrank)
print("\n Testing summarization...")
from summa.summarizer import summarize
long_text = """
Artificial intelligence is transforming industries across the world. From healthcare
to finance, AI is improving deision making and efficiency. However, challenges remain
in terms of regulation, ethics, and trust. Organization must balance innovation with
responsibility.
"""
summary = summarize(long_text, ratio=0.3) #30% of original text
print("Summary: \n", summary.strip())

# 5. Streamlit import test
print("\n Testing streamlit import...")
import streamlit
print("Streamlit version:", streamlit.__version__)

print("\n all checks completed successfully!")