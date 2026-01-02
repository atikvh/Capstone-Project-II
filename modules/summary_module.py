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
        if not text or not isinstance(text, str):
            return ""

        lines = text.splitlines()
        meaningful_lines = []

        SUBJECT_PATTERNS = r"^\s*(subject|per|perkara|tajuk|re)\s*[:\-]"

        for line in lines:
            line = line.strip()

            if not line:
                continue
            # Ignore ALL CAPS lines (headers, departments, titles)
            if re.fullmatch(r"[A-Z\s\W]+", line):
                continue
            # Ignore subject / title lines
            if re.search(SUBJECT_PATTERNS, line, re.IGNORECASE):
                continue
            # Ignore very short lines (likely names, labels, noise)
            if len(line.split()) < 5:
                continue
            # Ignore reference / date-heavy metadata lines
            if len(line.split()) <= 8 and re.search(r"\b(\d{2,}|/|:|-)\b", line):
                continue
            meaningful_lines.append(line)

        # Fallback: if everything is removed, return original text
        return " ".join(meaningful_lines) if meaningful_lines else text

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

    # for documents with field or tables
    def extract_key_information(self, text, category):
        if not text or not isinstance(text, str):
            return "(No content available.)"

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        extracted = []

        # Patterns
        APPLICATION_DATE = r"(tarikh\s*permohonan|application\s*date|date\s*submitted|tarikh\s*hantar)"
        DATE_NUMERIC = r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}"
        DATE_WRITTEN = (
            r"\d{1,2}"
            r"(?:st|nd|rd|th)?\s+"
            r"(January|February|March|April|May|June|July|August|September|October|November|December)"
            r"\s+\d{4}"
        )
        GENERIC_DATE = (
            r"(?:tarikh|date)\s*[:\-]?\s*("
            + DATE_NUMERIC
            + r"|"
            + DATE_WRITTEN
            + r")"
        )
        REFERENCE_APP = r"(reference|rujukan|ref\.?)\s*(no\.?|number)?\s*[:\-]?\s*[A-Z0-9\/\-]{4,}"
        FORM_TITLE = r"^(per|perkara|subject|subjek|title)\s*[:\-]\s*([A-Za-z0-9\s\/\-\(\),]{5,120})"
        REFERENCE_FIN = r"(rfq|tender)\s*(reference)?\s*(no\.?|number)?\s*[:\-]?\s*[A-Z0-9\/\-]{5,}"
        ISSUE_DATE = r"(date\s*issued|tarikh\s*dikeluarkan)"
        CLOSING_DATE = r"(closing\s*date|tarikh\s*tutup|tarikh\s*akhir)"

        if "Application & Forms" in category:
            for line in lines[:40]:
                # Form title
                if re.search(FORM_TITLE, line, re.IGNORECASE):
                    extracted.append(f"• Form Title: {line}")
                    continue
                # Application/submission date
                if re.search(APPLICATION_DATE, line, re.IGNORECASE):
                    extracted.append(f"• Application Submission Date: {line}")
                    continue
                # Strict date only
                m = re.search(GENERIC_DATE, line, re.IGNORECASE)
                if m:
                    extracted.append(f"• Date: {m.group(1)}")
                    continue
                # Application reference
                if re.search(REFERENCE_APP, line, re.IGNORECASE):
                    extracted.append(f"• Application Reference: {line}")
                    continue

            if not extracted:
                extracted.append("• This document is an application or form containing structured administrative information.")
                extracted.append("• No explicit submission date, title, or reference number was detected.")

        elif "Financial & Procurement" in category:
            for line in lines[:60]:
                if (
                    line.isupper()
                    and 6 <= len(line.split()) <= 20
                    and not re.search(r"(government|ministry|department|darussalam)", line, re.IGNORECASE)
                ):
                    extracted.append(f"• Procurement Title: {line}")
                    continue
                # Tender reference 
                if re.search(REFERENCE_FIN, line, re.IGNORECASE):
                    extracted.append(f"• Tender Reference: {line}")
                    continue
                # Issue date
                if re.search(ISSUE_DATE, line, re.IGNORECASE):
                    extracted.append(f"• Issue Date: {line}")
                    continue
                # Closing date
                if re.search(CLOSING_DATE, line, re.IGNORECASE):
                    extracted.append(f"• Closing Date: {line}")
                    continue
            if not extracted:
                extracted.append("• This document relates to financial or procurement matters.")
                extracted.append("• No clear tender reference or key dates were detected.")
        else:
            extracted.append("• No structured key information available for this document type.")

        # Remove duplicates while preserving order
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
            scores = (scores * 0.6) + (random_boost * 0.4)

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
'''
# ========== TEST BLOCK ==========
if __name__ == "__main__":
    ocr_output_path = "datasets/ocr_result.txt"
    with open(ocr_output_path, "r", encoding="utf-8") as f:
        text = f.read()

    s = TextSummarizer()
    print("SUMMARY:\n", s.summarize(text, 3))
    print("\nRANDOMIZED SUMMARY:\n", s.summarize(text, 3, randomize=True))
'''