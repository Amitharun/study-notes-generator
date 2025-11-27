import os
import re
import logging
from io import BytesIO
import concurrent.futures

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

import easyocr
from pdf2image import convert_from_bytes
import numpy as np
import cv2

from transformers import pipeline, AutoTokenizer
import torch

# -------------------- LOGGING -------------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr_summary_app")

# -------------------- FLASK APP -------------------- #
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".txt"}

# -------------------- GLOBAL LAZY MODELS -------------------- #
_reader = None
_summarizer = None
_tokenizer = None

DEVICE = 0 if torch.cuda.is_available() else -1


# -------------------- HELPERS -------------------- #
def allowed_file(filename):
    fname = filename.lower()
    return any(fname.endswith(ext) for ext in ALLOWED_EXTENSIONS)


def clean_text(t):
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# -------------------- OCR -------------------- #
def get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False)
        logger.info("EasyOCR reader loaded.")
    return _reader


def fast_easyocr_image(image):
    reader = get_reader()

    h, w = image.shape[:2]
    scale = 0.55
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    try:
        result = reader.readtext(resized, detail=0, paragraph=True)
        return " ".join(result)
    except Exception as e:
        logger.exception("OCR error: %s", e)
        return ""


def pdf_extract_scanned_easyocr(pdf_bytes):
    text_chunks = []

    try:
        pages = convert_from_bytes(pdf_bytes, dpi=180)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for p in pages:
                buf = BytesIO()
                p.save(buf, format="PNG")
                arr = np.frombuffer(buf.getvalue(), np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                futures.append(executor.submit(fast_easyocr_image, img))

            for f in concurrent.futures.as_completed(futures):
                text_chunks.append(f.result())

    except Exception as e:
        logger.exception("PDF OCR error: %s", e)

    return "\n".join(text_chunks)


# -------------------- SUMMARIZER -------------------- #
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        logger.info("Tokenizer loaded.")
    return _tokenizer


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=DEVICE
        )
        logger.info("Summarizer loaded.")
    return _summarizer


def split_into_chunks(text, max_tokens=380):
    tokenizer = get_tokenizer()
    ids = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    for i in range(0, len(ids), max_tokens):
        chunk_ids = ids[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk_ids))
    return chunks


def summarizer_high_accuracy(text):
    text = clean_text(text)
    summarizer = get_summarizer()
    chunks = split_into_chunks(text)

    bullets = []

    for c in chunks:
        try:
            out = summarizer(c, max_length=180, min_length=60, do_sample=False)
            summary = out[0]["summary_text"]

            # Regex sentence split
            sents = re.split(r"(?<=[.!?])\s+", summary)
            bullets.extend([s.strip() for s in sents if len(s.strip()) > 10])
        except Exception as e:
            logger.exception("Summarization error: %s", e)
            continue

    # Deduplicate
    final = []
    seen = set()
    for b in bullets:
        if b not in seen:
            seen.add(b)
            final.append(b)

    return final[:25]


# -------------------- ROUTES -------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def process():
    try:
        text_input = (request.form.get("text") or "").strip()
        file = request.files.get("file")
        extracted = ""

        if text_input:
            extracted = text_input

        elif file:
            filename = secure_filename(file.filename)
            if not allowed_file(filename):
                return jsonify({"error": "Invalid file type"}), 400

            data = file.read()

            if filename.lower().endswith(".pdf"):
                extracted = pdf_extract_scanned_easyocr(data)

            elif filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                arr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                extracted = fast_easyocr_image(img)

            elif filename.lower().endswith(".txt"):
                extracted = data.decode("utf-8", errors="replace")

        extracted = clean_text(extracted)

        if not extracted:
            return jsonify({"error": "No text detected"}), 400

        bullets = summarizer_high_accuracy(extracted)

        return jsonify({
            "summary_bullets": bullets,
            "total_points": len(bullets),
            "preview": extracted[:350]
        })

    except Exception as e:
        logger.exception("Processing error: %s", e)
        return jsonify({"error": "Internal server error"}), 500


# -------------------- RUN -------------------- #
if __name__ == "__main__":
    from waitress import serve
    print("Running on http://127.0.0.1:5000")
    serve(app, host="127.0.0.1", port=5000)
