import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import cv2

def extract_text(file_path: str) -> str:
    """
    Detects file type and extracts text from a PDF or image file.

    Args:
        file_path (str): Full path to the PDF or image file.

    Returns:
        str: Extracted text content.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                page_text = page.get_text("text")
                # Fallback to OCR if page is empty
                if not page_text.strip():
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    page_text = pytesseract.image_to_string(img)
                text += page_text + "\n"
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"Error reading PDF: {e}")

    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
        try:
            # Preprocess the image
            img_cv = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img_cv = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            img_pil = Image.fromarray(img_cv)
            text = pytesseract.image_to_string(img_pil)
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"Error reading image: {e}")

    else:
        raise ValueError(f"Unsupported file format: {ext}")

