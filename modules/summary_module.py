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
        cleaned_lines = []
        for line in text.split("\n"): # OCR results can contain meaningless line breaks, can confuse model
            line = line.strip()
            if not line:
                continue
            cleaned_lines.append(line)
        return " ".join(cleaned_lines)  

    # Fixes and splits sentences to overcome long sentences, missing punctuations and assist malay structures
    def split_sentences(self, text):
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
        sentences = self.split_sentences(cleaned)

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