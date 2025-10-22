"""
This handles the extraction and detection of content and layout of the
uploaded documents. Tesseract is used here supporting malay and english.
LayoutParser has been omitted from this module for lightweight processing.
"""
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import os

class TextExtractor: #extract text from images using Tesseract
    def __init__(self, lang="msa+eng"):
        self.lang = lang
    
    def extract_text_from_image(self, image):
        text = pytesseract.image_to_string(image, lang=self.lang) #using OCR to extract texts from image
        return text.strip()


class DocumentConverter: #for tesseract to process so change pdf to image
    @staticmethod
    def pdf_to_images(pdf_path, dpi=300):
        print(f"[DocumentConverter] Converting PDF: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
        print(f"[DocumentConverter] Converted {len(images)} page(s)")
        return images
    
    @staticmethod
    def load_image(image_path):
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image #return PIL image object
    

class OCRManager: #initiates previous classes
    def __init__(self, lang= "msa+eng"):
        self.text_extractor = TextExtractor(lang=lang)
        self.converter = DocumentConverter()

    def process_image(self, image, page_num = 1):
        if isinstance(image, str):
            image = self.converter.load_image(image)
        '''else:
            if image.mode != 'RGB':
                image = image.convert('RGB')''' #keeping here jic error
        print(f"[OCRManager] Processing page {page_num}...")

        text = self.text_extractor.extract_text_from_image(image)
        w, h = image.size

        block_data = {
            "block_id": 0,
            "type": "Text",
            "coordinates": {"x1":0, "y1":0, "x2":w, "y2":h},
            "confidence": 1.0,
            "text": text
        }
        
        return { #obtain layout information in blocks
            'page': page_num,
            'num_blocks': 1,
            'blocks': [block_data]
        }
    
    def process_pdf(self, pdf_path, dpi=300):
        images = self.converter.pdf_to_images(pdf_path, dpi=dpi)
        results = []
        for page_num, image in enumerate(images, start=1):
            page_result = self.process_image(image, page_num=page_num)
            results.append(page_result)
        return results
    
    def get_full_text(self, results):
        all_text = []
        for page in results:
            for block in page["blocks"]:
                if block["text"]:
                    all_text.append(block["text"])
        return '\n\n'.join(all_text)
    
class OCRResultManager:
    @staticmethod
    def save_to_text_file(results, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            for page_result in results:
                OCRResultManager._write_page_header(f, page_result["page"])
                OCRResultManager._write_blocks(f, page_result["blocks"])
                OCRResultManager._write_full_text(f, page_result)
        print(f"[ResultManager] Saved OCR result to: {output_path}")

    @staticmethod
    def _write_page_header(file, page_num):
        file.write(f"{'='*70}\n")
        file.write(f"PAGE {page_num}\n")
        file.write(f"{'='*70}\n\n")

    @staticmethod
    def _write_blocks(file, blocks):
        file.write(f"Detected blocks: {len(blocks)}\n\n")
        for block in blocks:
            file.write(f"{block['type']} Confidence: {block['confidence']:.2f}\n")
            file.write(f"Location: {block['coordinates']}\n")
            file.write(f"Text: {block['text']}\n")
            file.write(f"{'-'*50}\n\n")
    
    @staticmethod
    def _write_full_text(file, page_result):
        file.write(f"\n{'#'*70}\n")
        file.write(f"FULL TEXT (PAGE {page_result['page']})\n")
        file.write(f"{'#'*70}\n\n")

        for block in page_result['blocks']:
            if block["text"]:
                file.write(f"{block['text']}\n\n")

# ========== TEST BLOCK ==========
if __name__ == "__main__":
    print("🔍 Starting OCR Module Test...\n")

    pdf_path = "datasets/archived/Borang Melapor SUJATI KL-2.pdf"
    output_path = "datasets/text_result.txt"

    try:
        ocr = OCRManager(lang="msa+eng")
        results = ocr.process_pdf(pdf_path)

        # Save the result to text file
        OCRResultManager.save_to_text_file(results, output_path)

        print("\n✅ OCR module test completed successfully!")
        print(f"Results saved to: {output_path}")

    except Exception as e:
        print(f"\n❌ OCR module test failed: {e}")

        
# .venv\Scripts\activate.bat