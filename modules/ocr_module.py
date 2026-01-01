"""
This handles the extraction and detection of content and layout of the
uploaded documents. Tesseract is used here supporting malay and english.
LayoutParser has been omitted from this module for lightweight processing.
"""
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import os


class TextExtractor:
    def __init__(self, lang="msa+eng", header_ratio=0.08, footer_ratio=0.08):
        self.lang = lang
        self.header_ratio = header_ratio
        self.footer_ratio = footer_ratio

    def crop_header_footer(self, image):
        # Remove top and bottom portions of image to reduce noice
        width, height = image.size
        top = int(height* self.header_ratio)
        bottom = int(height*(1-self.footer_ratio))

        return image.crop((0, top, width, bottom))

    # Extract text from image to strings with tesseract
    def extract_text(self, image): 
        image = self.crop_header_footer(image)
        text = pytesseract.image_to_string(image, lang=self.lang)
        return text.strip()


class DocumentConverter:
    # Converts documents in pdfs to images first so can extract
    @staticmethod
    def pdf_to_images(pdf_path, dpi=300):
        print(f"[PDF] Converting: {pdf_path}")
        return convert_from_path(pdf_path, dpi=dpi)

    @staticmethod
    def load_image(image_path):
        img = Image.open(image_path)
        return img.convert("RGB") if img.mode != "RGB" else img

# master method
class OCRManager:
    def __init__(self, lang="msa+eng"):
        self.extractor = TextExtractor(lang)
        self.converter = DocumentConverter()

    # Step 1: Process uploaded file (jpg/png/pdf)
    def process_pdf(self, pdf_path):
        images = self.converter.pdf_to_images(pdf_path)
        return [self.extractor.extract_text(img) for img in images]

    def process_image(self, image_path):
        img = self.converter.load_image(image_path)
        return [self.extractor.extract_text(img)]

    # detects if need to call for pdf or image processing
    def process(self, input_path):
        if input_path.lower().endswith(".pdf"):
            return self.process_pdf(input_path)
        return self.process_image(input_path)

# Save ocr result to txt file
class OCRResultManager:
    @staticmethod
    def save_clean_text(text_list, output_path):
        clean_text = "\n\n".join(text_list)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(clean_text)
        print(f"[Saved] Clean OCR text saved to: {output_path}")

# ========== TEST BLOCK ==========
if __name__ == "__main__":
    print("Running OCR Test...\n")

    input_path = "datasets/archived/Administrative.png" 
    output_path = "datasets/ocr_result.txt"

    try:
        ocr = OCRManager(lang="msa+eng")
        text_list = ocr.process(input_path)

        OCRResultManager.save_clean_text(text_list, output_path)

        print("\nOCR test completed successfully!")

    except Exception as e:
        print(f"OCR test failed: {e}")
