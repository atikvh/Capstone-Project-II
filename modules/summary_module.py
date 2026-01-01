'''
This is responsible for the summarization feature in the system.
TextRank is used here with some assistance from cleaning and sentences detection.
Take raw text directly from OCR result.
'''

import re
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt", quiet=True)

class TextSummarizer:

    def __init__(self):
        pass

    # Prepares the raw text from ocr before undergoing summarization
    def clean_text_for_summary(self, text):
        lines = text.split("\n")
        meaningful_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove likely header/meta lines
            if re.match(r"^[A-Z\s\W]+$", line):  # all caps
                continue
            if len(line.split()) <= 8 and re.search(r"[\d/:]", line):  # reference numbers / dates
                continue
            if len(line.split()) <= 3:  # very short lines
                continue
            if re.match(r"^\w+\s*:", line): # titles or subject line
                continue
            if len(line.split()) <= 12 and ":" in line: # with colon
                continue

            meaningful_lines.append(line)

        # fallback to original if all lines removed
        cleaned_text = " ".join(meaningful_lines) if meaningful_lines else text

        return cleaned_text


    # Fixes and splits sentences to overcome long sentences, missing punctuations and assist malay structures
    def fix_sentences(self, text):
        # Fix common lowercase "i" issue for English
        text = text.replace(" i ", " I ")

        sentences = nltk.sent_tokenize(text)

        # Identifying new sentence in malay when period comes
        refined = []
        for s in sentences:
            parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', s)
            for p in parts:
                p = p.strip()
                if len(p.split()) >= 4:
                    refined.append(p)

        return refined

    # for documents with structures -> not able to summarize
    def extract_key_information(self, text, category):
        if not text or not isinstance(text, str):
            return "(No content available.)"
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        extracted = []

        # Common patterns
        patterns = {
            "reference" : r"(reference|rujukan|ref\.?)\s*[:\-]?\s*(.+)",
            "date": r"(date|tarikh)\s*[:\-]?\s*(.+)",
            "subject": r"(subject|perkara|per)\s*[:\-]?\s*(.+)",
            "title_caps": r"^[A-Z\s]{8,}$"
        }

        # Predicted category: Application & Forms
        if "Application & Forms" in category:
            for line in lines[:30]:  # only scan top section
                if re.search(patterns["subject"], line, re.IGNORECASE):
                    extracted.append(f"• Application Title: {line}")
                elif re.search(patterns["date"], line, re.IGNORECASE):
                    extracted.append(f"• Date: {line}")
                elif re.search(patterns["reference"], line, re.IGNORECASE):
                    extracted.append(f"• Reference: {line}")

            if not extracted:
                extracted.append("• This document relates to important application request matters.")
                extracted.append("• This document contains structured fields and form-based content.")
        
        # Predicted category: Financial & Procurement
        elif "Financial & Procurement" in category:
            for line in lines[:40]:
                if re.search(patterns["title_caps"], line):
                    extracted.append(f"• Procurement Title: {line}")
                elif re.search(patterns["reference"], line, re.IGNORECASE):
                    extracted.append(f"• Tender Reference: {line}")
                elif re.search(patterns["date"], line, re.IGNORECASE):
                    extracted.append(f"• Important Date: {line}")

            if not extracted:
                extracted.append("• This document relates to important financial or procurement matters.")
                extracted.append("• This document contains structured fields and form-based content.")
            
        else:
            extracted.append("• No structured key information available.")
        return "\n".join(dict.fromkeys(extracted))


    # model method
    def textrank(self, sentences, num_sentences=3, randomize=False):

        if len(sentences) == 0:
            return ["(No meaningful content available.)"]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)
        sim_matrix = cosine_similarity(X)

        # Base score = how connected a sentence is
        scores = sim_matrix.sum(axis=1)

        # Instead of tiny noise, reshuffle scores in a meaningful way
        if randomize:
            # Mix TextRank score (70%) + random importance (30%)
            random_boost = np.random.uniform(0.0, 1.0, len(scores))
            scores = (scores * 0.7) + (random_boost * 0.3)

        ranked_idx = np.argsort(scores)[::-1]
        selected_idx = sorted(ranked_idx[:num_sentences])

        return [sentences[i] for i in selected_idx]

    # master method
    def summarize(self, text, sentence_count=3, randomize=False):

        if not text or not isinstance(text, str):
            return "(Invalid text provided.)"

        # Step 1: Cleaning
        cleaned = self.clean_text_for_summary(text)

        # Step 2: Sentence extraction
        sentences = self.fix_sentences(cleaned)

        if len(sentences) == 0:
            return "(No meaningful sentences found.)"

        # Step 3: Ranking
        best_sentences = self.textrank(
            sentences,
            num_sentences=sentence_count,
            randomize=randomize
        )

        # Step 4: Rebuild into clean paragraph
        summary = " ".join(best_sentences).strip()

        return summary

# ========== TEST BLOCK ==========
if __name__ == "__main__":
    ocr_output_path = "datasets/ocr_result.txt"
    with open(ocr_output_path, "r", encoding="utf-8") as f:
        text = f.read()

    s = TextSummarizer()
    print("SUMMARY:\n", s.summarize(text, 3))
    print("\nRANDOMIZED SUMMARY:\n", s.summarize(text, 3, randomize=True))