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

    def detect_layout(self, image):
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