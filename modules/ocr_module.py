"""
This handles the extraction and detection of content and layout of the
uploaded documents. LayoutParser and Tesseract is used here.
"""
import layoutparser as lp
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import os

