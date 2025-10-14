"""
This handles the extraction and detection of content and layout of the
uploaded documents. LayoutParser and Tesseract is used here.
"""
import layoutparser as lp
from layoutparser.models import Detectron2LayoutModel
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import pytesseract
import os

class LayoutDetector:
    def __init__(self, lang="msa+eng", threshold=0.8):
        self.lang = lang
        self.threshold = threshold #ignore below threshold
        self.model = None #detector model

    def load_model(self):
        if self.model is not None:
            print("[LayoutDetector] Model already loaded.")
            return
        
        print("[LayoutDetector] Loading model...")
        try:
            self.model = Detectron2LayoutModel( #loading model for layout
                config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                label_map={0: "Text", 1: "Title"},
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.threshold]
            )
            print("[LayoutDetector] Model loaded successfully.")
        except Exception as e:
            print(f"[LayoutDetector] Error loading model: {e}")
            raise

    def detect_layout(self, image): #implement layoutparser
        if self.model is None:
            self.load_model()

        if isinstance (image, Image.Image):
            image_array = np.array(image) #converting PIL image to numpy if needed
        else:
            image_array = image
        #use model to detect layout
        layout = self.model.detect(image_array) # type: ignore
        layout = layout.sort(key=lambda b: (b.coordinates[1], b.coordinates[0])) #sort blocks: top to bottom, left to right
        return layout


class TextExtractor: #extract text from specific layout region
    def __init__(self, lang="msa+eng"):
        self.lang = lang
    
    def extract_text_from_image(self, image):
        text = pytesseract.image_to_string(image, lang=self.lang) #using OCR to extract texts from image
        return text.strip()
    
    def extract_text_from_region(self, image, coordinates): #extract texts from specific region of an image
        x1, y1, x2, y2 = map(int, coordinates) #x1-y1 top-left corner, x2-y2 bottom-right corner (defining rectangle in image)
        cropped = image.crop((x1, y1, x2, y2)) 
        return self.extract_text_from_image(cropped) #ocr only takes text in that cropped region


class DocumentConverter: #for tesseract to process so change pdf to image
    @staticmethod
    def pdf_to_images(pdf_path, dpi=300):
        print(f"[DocumentConverter] Converting PDF: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=dpi)
        print(f"[DocumentConverter] Converted {len(images)} pages")
        return images
    
    @staticmethod
    def load_image(image_path):
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image #return PIL image object
    

class OCRManager: #initiates previous classes
    def __init__(self, lang= "msa+eng", threshold = 0.8):
        self.layout_detector = LayoutDetector(threshold=threshold)
        self.text_extractor = TextExtractor(lang=lang)
        self.converter = DocumentConverter()

    def process_image(self, image, page_num = 1):
        if isinstance(image, str):
            image = self.converter.load_image(image)
        '''else:
            if image.mode != 'RGB':
                image = image.convert('RGB')''' #keeping here jic error
        print(f"[OCRManager] Processing page {page_num}")

        layout = self.layout_detector.detect_layout(image) 
        print(f"[OCRManager] Found {len(layout)} blocks") # type: ignore

        blocks = []
        for idx, block in enumerate(layout): # type: ignore
            text = self.text_extractor.extract_text_from_region(image, block.coordinates)
            x1, y1, x2, y2 = map(int, block.coordinates)
            block_data = {
                'block_id': idx,
                'type': block.type,
                'coordinates': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                'confidence': block.score,
                'text': text
            }
            blocks.append(block_data)
        
        return { #obtain layout information in blocks
            'page': page_num,
            'num_blocks': len(blocks),
            'blocks': blocks
        }
    
    def process_pdf(self, pdf_path, dpi=300):
        images = self.converter.pdf_to_images(pdf_path, dpi=dpi)
        results = []
        for page_num, image in enumerate(images, start=1):
            page_result = self.process_image(image, page_num=page_num)
            results.append(page_result)
        return results
    
    def get_full_text(self, page_result):
        text_parts = []
        for block in page_result['blocks']:
            if not block['text']:
                continue
            if block['type'] == "Title":
                text_parts.append(f"\n{'='*50}")
                text_parts.append(block['text'])
                text_parts.append(f"{'='*50}\n")
            else:
                text_parts.append(block['text'])
        return '\n\n'.join(text_parts)
    
class ResultManager:
    @staticmethod
    def save_to_text_file(results, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            for page_result in results:
                ResultManager._write_page_header(f, page_result['page'])
                ResultManager._write_blocks(f, page_result['blocks'])
                ResultManager._write_full_text(f, page_result)
        print(f"[ResultManager] Saved to: {output_path}")

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
            if block['text']:
                if block['text'] == "Title":
                    file.write(f"\n{'='*50}\n")
                    file.write(f"{block['text']}\n")
                    file.write(f"{'='*50}\n\n")
                else:
                    file.write(f"{block['text']}\n\n")