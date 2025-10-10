"""
This handles the extraction and detection of content and layout of the
uploaded documents. LayoutParser and Tesseract is used here.
"""
import layoutparser as lp
from layoutparser.models import Detectron2LayoutModel
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import os

class OCRManager:
    def __init__(self, lang="msa+eng", threshold=0.8):
        self.lang = lang
        self.threshold = threshold #ignore below threshold
        self.model = None

    def load_model(self):
        print("Loading LayoutParser model...(may take a while first time.)")
        self.model = Detectron2LayoutModel(
            config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
            label_map={0: "Text", 1: "Title"},
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.threshold]
        )
        print("[OCR] LayoutParser model loaded successfully.")
    