# checking the activation of environment for IDOS

# 1. OCR (Eng + Msa)
print("Testing OCR (Tesseract + Pytesseract)...")
import pytesseract
from PIL import Image, ImageDraw
# image to scan
img = Image.new("RGB", (200,60), color = "white")
d = ImageDraw.Draw(img)
d.text((10,10), "Hello  World! Kerajaan Brunei Darussalam mengalu-alukan permohonan baru.", fill="black")
# Test OCR
ocr_text = pytesseract.image_to_string(img, lang="eng+msa")
print("OCR result:\n", ocr_text.strip())

# 2. Preprocessing Text (English and Malay - removing stopwords from sentence)
print("\n Testing NLTK preprocessing...")
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import re
# English stopwords from NLTK
eng_stops = set(stopwords.words("english"))
# Malay stopwords custom
malay_stops = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "pada", "dengan",
    "adalah", "saya", "kami", "anda", "mereka", "akan", "tidak", "boleh", "sebagai",
    "oleh", "daripada", "atau", "juga", "lebih", "masih", "kerana", "dalam", "bagi"
}
# Cleaning function
def clean_text(text, lang="eng"):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\u00C0-\u024F\u1E00-\u1EFF\s]", "", text)
    tokens = text.split()
    if lang == "eng":
        filtered = [word for word in tokens if word not in eng_stops]
    elif lang == "msa":
        filtered = [word for word in tokens if word not in malay_stops]
    else:
        filtered = tokens
    return " ".join(filtered)
# Example sentences to clean
eng_text = "The government of Brunei Darussalam urges for new applications."
msa_text = "Kerajaan Brunei Darussalam mengalu-alukan pemohonan baharu daripada rakyat."
# Execute cleaning
clean_eng = clean_text(eng_text, "eng")
clean_msa = clean_text(msa_text, "msa")
print("Original texts: \n", eng_text, "\n", msa_text)
print("Cleaned texts: \n", clean_eng, "\n", clean_msa)

# 3. Categorization Test (TF-IDF + SVM)
print("\n Testing categorization with scikit-learn...")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
# dummy dataset
docs = [
    "This is a memo about work",
    "Leave application form submitted",
    "Official approval letter from HR",
    "Permohonan cuti tahunan telah dihantar",
    "Surat rasmi daripada Jabatan Sumber Manusia"
]
labels = ["Memo", "Form", "Letter", "Form", "Letter"]
# Text to numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
# Train a simple classifier
clf = LinearSVC()
clf.fit(X, labels)
# Test on new data
test_doc = ["Sila baca memo dari HR", "Ini adalah surat rasmi"]
prediction = clf.predict(vectorizer.transform(test_doc))
for i, doc in enumerate(test_doc):
    print(f"Categorization result for '{doc}': {prediction[i]}")

# 4. Summarization test (Textrank)
print("\n Testing summarization...")
from summa.summarizer import summarize
long_text = """
Artificial Intelligence(AI) sedang mengubah pelbagai industri di seluruh dunia.
Merangkumi industri-industri seperti kesihatan dan kewangan, AI membantu meningkatkan
kecekapan dan pembuatan keputusan yang lebih baik. Namun, cabaran kekal
dalam aspek peraturan, etika dan kepercayaan. Organisasi mesti mengimbangi
inovasi dengan tanggungjawab dalam penggunaan.
"""
summary = summarize(long_text, ratio=0.3) #30% of original text
print("Summary: \n", summary.strip())

# 5. Streamlit import test
print("\n Testing streamlit import...")
import streamlit
print("Streamlit version:", streamlit.__version__)

print("\n all checks completed successfully!")