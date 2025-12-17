import streamlit as st
st.set_page_config(layout="wide", page_title="Dynamic OCR")

# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import cv2
import easyocr
import pytesseract
import re
import io
from PIL import Image
from pdf2image import convert_from_bytes
from pdfminer.high_level import extract_text

# ============================================================
# CONFIG
# ============================================================
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\xd5914\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)
POPPLER_PATH = r"C:\Users\xd5914\source\repos\Poppler\Library\bin"
TESS_CONFIG = "--oem 3 --psm 6 -c preserve_interword_spaces=1"

# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "uploaded": False,
    "processing": False,
    "file_bytes": None,
    "file_type": None,
    "mode": None,
    "final_lines": [],
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ============================================================
# UPLOAD CALLBACK (CRITICAL FIX)
# ============================================================
def handle_upload():
    uploaded = st.session_state.uploader
    if uploaded is None:
        return

    st.session_state.file_bytes = uploaded.getvalue()
    st.session_state.file_type = uploaded.type
    st.session_state.uploaded = True
    st.session_state.processing = True

# ============================================================
# OCR HELPERS
# ============================================================
def enhance_for_ocr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    return cv2.createCLAHE(2.0, (8, 8)).apply(gray)

def normalize_text(text):
    if not text:
        return ""
    text = re.sub(r"[^\x20-\x7E]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def reflow_lines(lines):
    merged, buf = [], ""
    for line in lines:
        line = normalize_text(line)
        if not line:
            continue
        if buf and not buf.endswith((".", ":")) and not line[0].isupper():
            buf += " " + line
        else:
            if buf:
                merged.append(buf)
            buf = line
    if buf:
        merged.append(buf)
    return merged

def run_easyocr(img):
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = reader.readtext(img, detail=1)
    results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))

    lines, buf, last_y = [], [], None
    for bbox, text, _ in results:
        y = bbox[0][1]
        if last_y is None or abs(y - last_y) <= 18:
            buf.append(text)
        else:
            lines.append(normalize_text(" ".join(buf)))
            buf = [text]
        last_y = y

    if buf:
        lines.append(normalize_text(" ".join(buf)))
    return [l for l in lines if l]

def run_tesseract(img):
    txt = pytesseract.image_to_string(
        Image.fromarray(img), config=TESS_CONFIG
    )
    return [normalize_text(l) for l in txt.splitlines() if l.strip()]

# ============================================================
# PDF HANDLING
# ============================================================
def extract_text_pdf(pdf_bytes):
    try:
        return extract_text(io.BytesIO(pdf_bytes))
    except:
        return ""

def is_text_pdf(pdf_bytes, min_chars=200):
    txt = extract_text_pdf(pdf_bytes)
    return len(txt) >= min_chars, txt

def load_input(file_bytes, file_type):
    if file_type == "application/pdf":
        is_text, txt = is_text_pdf(file_bytes)
        if is_text:
            return "text", txt.splitlines(), []

        pages = convert_from_bytes(
            file_bytes, dpi=300, poppler_path=POPPLER_PATH
        )
        imgs = [
            cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)
            for p in pages
        ]
        return "image", None, imgs

    img = cv2.imdecode(
        np.frombuffer(file_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )
    return "image", None, [img]

# ============================================================
# UI
# ============================================================
st.title("📄 Dynamic OCR")

# ============================
# STAGE 1 — UPLOAD (FIXED)
# ============================
if not st.session_state.uploaded:

    st.file_uploader(
        "Upload Image or PDF",
        ["png", "jpg", "jpeg", "tif", "pdf"],
        key="uploader",
        on_change=handle_upload
    )

    st.info("Supported formats: PDF, JPG, PNG, TIFF")

    if st.session_state.uploaded:
        st.experimental_rerun()

    st.stop()

# ============================
# STAGE 2 — OUTPUT UI
# ============================
col_left, col_right = st.columns([1, 1])

# ---------- LEFT: PREVIEW ----------
with col_left:
    st.subheader("📤 Uploaded Document")

    if st.session_state.file_type == "application/pdf":
        preview = convert_from_bytes(
            st.session_state.file_bytes,
            dpi=120,
            first_page=1,
            last_page=1,
            poppler_path=POPPLER_PATH
        )
        st.image(preview[0], use_column_width=True)
    else:
        img = cv2.imdecode(
            np.frombuffer(st.session_state.file_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )
        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            use_column_width=True
        )

# ---------- RIGHT: OUTPUT ----------
with col_right:

    a, b = st.columns(2)
    with a:
        if st.button("⬅ Back"):
            st.session_state.clear()
            st.experimental_rerun()
    with b:
        if st.button("✖ Reset"):
            st.session_state.clear()
            st.experimental_rerun()

    tab_parse, tab_extract = st.tabs(["Parse", "Extract"])
    process_placeholder = st.empty()

    # ---------- PROCESSING ----------
    if st.session_state.processing:

        with process_placeholder:
            st.info("🔍 Processing document…")
            st.spinner("Running OCR…")

        mode, text_lines, images = load_input(
            st.session_state.file_bytes,
            st.session_state.file_type
        )

        final_lines = []
        if mode == "text":
            final_lines = reflow_lines(text_lines)
        else:
            for img in images:
                img = enhance_for_ocr(img)
                final_lines.extend(run_easyocr(img))
                final_lines.extend(run_tesseract(img))

        seen = set()
        final_lines = [
            l for l in final_lines
            if not (l in seen or seen.add(l))
        ]

        st.session_state.final_lines = final_lines
        st.session_state.mode = mode
        st.session_state.processing = False

        process_placeholder.empty()
        st.experimental_rerun()

    # ---------- FINAL OUTPUT ----------
    else:
        with tab_parse:
            st.text_area(
                "Final OCR Output",
                "\n".join(st.session_state.final_lines),
                height=520
            )

        with tab_extract:
            st.json({
                "mode": st.session_state.mode,
                "total_lines": len(st.session_state.final_lines),
                "lines": [
                    {"line_no": i + 1, "text": l}
                    for i, l in enumerate(st.session_state.final_lines)
                ]
            })

st.caption( " " )
