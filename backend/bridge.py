import base64
import difflib
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --- CRITICAL WINDOWS FIX START ---
# This tells Python exactly where your Tesseract "Eyes" are installed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# --- CRITICAL WINDOWS FIX END ---

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("shadowscript")

USE_EASYOCR = False
easy_reader = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global easy_reader
    if USE_EASYOCR:
        import easyocr
        easy_reader = easyocr.Reader(["en"], gpu=False)
        log.info("EasyOCR ready.")
    else:
        log.info("Using Tesseract engine.")
    yield

app = FastAPI(title="ShadowScript OCR Bridge", version="1.0.0", lifespan=lifespan)

last_extracted_text: str = ""
SIMILARITY_THRESHOLD: float = 0.70

class FramePayload(BaseModel):
    image: str

class OCRResult(BaseModel):
    text: str
    duplicate: bool = False

def decode_base64_to_cv2(b64_string: str) -> np.ndarray:
    try:
        raw_bytes = base64.b64decode(b64_string)
        np_array  = np.frombuffer(raw_bytes, dtype=np.uint8)
        image     = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image bytes.")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {e}")

def crop_caption_area(image: np.ndarray) -> np.ndarray:
    """Crop to bottom 30% of frame — where captions always appear."""
    h, w = image.shape[:2]
    return image[int(h * 0.7):h, 0:w]

def preprocess(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 3x upscale for caption strips
    gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

    # Stronger denoise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove small noise blobs (not text)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # White padding
    binary = cv2.copyMakeBorder(binary, 30, 30, 30, 30,
                                cv2.BORDER_CONSTANT, value=255)
    return binary

def run_tesseract(image: np.ndarray) -> str:
    # PSM 6 + char blacklist filters junk symbols while reading full caption block
    config = "--oem 3 --psm 6 -c tessedit_char_blacklist=|}{]["
    return pytesseract.image_to_string(image, config=config).strip()

def run_easyocr(image: np.ndarray) -> str:
    results = easy_reader.readtext(image, detail=0, paragraph=True)
    return " ".join(results).strip()

def extract_text(preprocessed: np.ndarray) -> str:
    return run_easyocr(preprocessed) if USE_EASYOCR else run_tesseract(preprocessed)

def clean_text(text: str) -> str:
    import re
    lines = text.split('\n')
    # Drop lines shorter than 4 chars (noise / stray glyphs)
    lines = [l for l in lines if len(l.strip()) > 4]
    # Drop lines where more than half the chars are non-alphanumeric
    lines = [l for l in lines
             if len(re.sub(r'[^a-zA-Z0-9\s]', '', l)) > len(l) * 0.5]
    return '\n'.join(lines).strip()

def is_duplicate(new_text: str) -> bool:
    global last_extracted_text
    if not last_extracted_text:
        return False
    ratio = difflib.SequenceMatcher(None, last_extracted_text.lower(), new_text.lower()).ratio()
    return ratio >= SIMILARITY_THRESHOLD

@app.post("/process-frame")
async def process_frame(payload: FramePayload):
    global last_extracted_text
    image        = decode_base64_to_cv2(payload.image)
    image        = crop_caption_area(image)
    preprocessed = preprocess(image)
    raw_text     = clean_text(extract_text(preprocessed))
    if not raw_text:
        return JSONResponse(content={"text": "", "duplicate": False})
    if is_duplicate(raw_text):
        log.info(f"Duplicate detected: {repr(raw_text[:30])}...")
        return JSONResponse(content={"text": "", "duplicate": True})
    last_extracted_text = raw_text
    log.info(f"OCR Output: {repr(raw_text)}")
    return JSONResponse(content={"text": raw_text, "duplicate": False})

@app.get("/health")
async def health():
    return {"status": "ok", "engine": "easyocr" if USE_EASYOCR else "tesseract"}

if __name__ == "__main__":
    import uvicorn
    # We use 127.0.0.1 (localhost) so your Java app can find it easily
    uvicorn.run(app, host="127.0.0.1", port=8000)
    