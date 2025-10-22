"""
This handles the validation to confirm system is able to work
with the uploaded files. It checks the format and size of the file.
"""
import os

class DocumentValidator:
    SUPPORTED_FORMATS = [".pdf", ".jpg", ".jpeg", ".png"]
    MAX_FILE_SIZE_MB = 10

    def __init__(self):
        pass

    def check_format(self, file_path: str) -> bool:
        ext = os.path.splitext(file_path)[1].lower() #1. Seperate path e.g file.PDF -> file,.pdf
        if ext not in self.SUPPORTED_FORMATS: #2. Check format from list
            raise ValueError(f"Unsupported file format: {ext}.\nOnly {self.SUPPORTED_FORMATS} is allowed.")
        return True
    
    def check_size(self, file_path: str) -> bool:
        file_size_MB = os.path.getsize(file_path) / (1024*1024) #1. Obtain file size
        if file_size_MB > self.MAX_FILE_SIZE_MB: # 2. Check file size against max size
            raise ValueError(f"File too large: {file_size_MB:.2f}MB. \nMax allowed is {self.MAX_FILE_SIZE_MB}MB.")
        return True
    
    def validate(self, file_path: str) -> bool: # runs both size and format checks on file.
        self.check_format(file_path)
        self.check_size(file_path)
        print(f"File {os.path.basename(file_path)} is compatible with the system.")
        return True
"""
# ========== TEST BLOCK ==========
file_path1 = "datasets/archived/Borang Melapor SUJATI KL-2.pdf"
checker = DocumentValidator()
try:
    checker.validate(file_path1)
except ValueError as e:
    print(f"[ERROR] {e}")
    exit()

--Successful.
"""