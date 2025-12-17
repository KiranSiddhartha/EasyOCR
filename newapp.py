# import streamlit as st
# st.set_page_config(layout="wide", page_title="Dynamic OCR")

# # ============================================================
# # IMPORTS
# # ============================================================
# import numpy as np
# import cv2
# import easyocr
# import pytesseract
# import re
# import io
# import zipfile
# from PIL import Image
# from pdf2image import convert_from_bytes
# from pdfminer.high_level import extract_text

# # ============================================================
# # CONFIG
# # ============================================================
# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Users\xd5914\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
# )
# POPPLER_PATH = r"C:\Users\xd5914\source\repos\Poppler\Library\bin"
# TESS_CONFIG = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
# ALLOWED_EXT = (".png", ".jpg", ".jpeg", ".tif", ".pdf")

# # ============================================================
# # CACHED OCR MODEL
# # ============================================================
# @st.cache_resource
# def get_easyocr_reader():
#     return easyocr.Reader(["en"], gpu=False)

# EASY_READER = get_easyocr_reader()

# # ============================================================
# # SESSION STATE
# # ============================================================
# defaults = {"uploaded": False, "files": [], "results": {}}
# for k, v in defaults.items():
#     st.session_state.setdefault(k, v)

# # ============================================================
# # IN-MEMORY FILE (ZIP SUPPORT)
# # ============================================================
# class InMemoryFile:
#     def __init__(self, name, data, mime):
#         self.name = name
#         self._data = data
#         self.type = mime

#     def getvalue(self):
#         return self._data

# # ============================================================
# # UPLOAD HANDLER
# # ============================================================
# def handle_upload():
#     uploaded = st.session_state.uploader
#     files = []

#     for f in uploaded:
#         if f.name.lower().endswith(".zip"):
#             with zipfile.ZipFile(io.BytesIO(f.getvalue())) as z:
#                 for info in z.infolist():
#                     if info.filename.lower().endswith(ALLOWED_EXT):
#                         data = z.read(info)
#                         mime = (
#                             "application/pdf"
#                             if info.filename.lower().endswith(".pdf")
#                             else "image"
#                         )
#                         files.append(InMemoryFile(info.filename, data, mime))
#         elif f.name.lower().endswith(ALLOWED_EXT):
#             files.append(f)

#     if files:
#         st.session_state.files = files
#         st.session_state.results = {}
#         st.session_state.uploaded = True

# # ============================================================
# # IMAGE ENHANCEMENT (BLUR SAFE)
# # ============================================================
# def enhance_for_blur_safe(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (3, 3), 0)
#     gray = cv2.createCLAHE(3.0, (8, 8)).apply(gray)
#     return gray

# def downscale_for_ocr(img, max_width=1800):
#     h, w = img.shape[:2]
#     if w > max_width:
#         scale = max_width / w
#         img = cv2.resize(img, (int(w * scale), int(h * scale)))
#     return img

# def normalize_text(t):
#     if not t:
#         return ""
#     t = re.sub(r"[^\x20-\x7E]", "", t)
#     return re.sub(r"\s{2,}", " ", t).strip()

# # ============================================================
# # CONFIDENCE & FILTERING
# # ============================================================
# def classify_conf(c):
#     if c >= 0.90:
#         return "perfect"
#     if c >= 0.70:
#         return "partial"
#     return "poor"

# def filter_low_confidence(words, min_conf=0.55):
#     return [w for w in words if w["confidence"] >= min_conf]

# # ============================================================
# # CONTEXT-AWARE CORRECTIONS
# # ============================================================
# CHAR_CONFUSIONS = {
#     "O": "0",
#     "I": "1",
#     "l": "1",
#     "S": "5",
#     "B": "8",
# }

# def fix_by_context(text):
#     if re.search(r"\d", text):
#         for k, v in CHAR_CONFUSIONS.items():
#             text = text.replace(k, v)
#     return text

# def normalize_fields(line):
#     # ---- FIXED REGEX (NO INVALID GROUPS)
#     line = re.sub(
#         r"(DOB[:\s]*)I(\d{3})[-/]I(\d{2})[-/]I(\d{2})",
#         r"\g<1>1\2-\3-\4",
#         line,
#         flags=re.I
#     )

#     line = re.sub(
#         r"(PREMIUM[:\s]*)(\$?\s*\d+(\.\d{2})?)",
#         r"\g<1>\2",
#         line,
#         flags=re.I
#     )

#     return line

# def ml_correct(text):
#     return text  # future ML hook

# def post_process_output(lines):
#     clean = []
#     for l in lines:
#         l = fix_by_context(l)
#         l = normalize_fields(l)
#         l = ml_correct(l)
#         if len(l.strip()) >= 3:
#             clean.append(l)
#     return clean

# # ============================================================
# # OCR PIPELINE
# # ============================================================
# def run_ocr_pipeline(img):
#     img = downscale_for_ocr(img)
#     gray = enhance_for_blur_safe(img)

#     easy_results = EASY_READER.readtext(gray, detail=1)

#     words, easy_text = [], []
#     for bbox, text, conf in easy_results:
#         text = normalize_text(text)
#         if not text:
#             continue
#         words.append({
#             "text": text,
#             "confidence": round(conf, 2),
#             "status": classify_conf(conf),
#             "bbox": bbox
#         })
#         easy_text.append(text)

#     words = filter_low_confidence(words)

#     tess_text = pytesseract.image_to_string(
#         Image.fromarray(gray), config=TESS_CONFIG
#     )
#     tess_lines = [
#         normalize_text(l) for l in tess_text.splitlines() if l.strip()
#     ]

#     final_lines = list(dict.fromkeys(tess_lines + easy_text))
#     final_lines = post_process_output(final_lines)

#     return final_lines, words

# # ============================================================
# # PDF HANDLING
# # ============================================================
# def extract_text_pdf(pdf_bytes):
#     try:
#         return extract_text(io.BytesIO(pdf_bytes))
#     except:
#         return ""

# def load_input(file_bytes, file_type):
#     images = []

#     if file_type == "application/pdf":
#         pages = convert_from_bytes(
#             file_bytes, dpi=150, poppler_path=POPPLER_PATH
#         )
#         images = [
#             cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)
#             for p in pages
#         ]

#         txt = extract_text_pdf(file_bytes)
#         if len(txt.strip()) > 200:
#             return "text", txt.splitlines(), images

#         return "image", None, images

#     img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
#     return "image", None, [img]

# # ============================================================
# # ACCURACY SCORE
# # ============================================================
# def calculate_document_accuracy(words, lines):
#     if not words:
#         return 0
#     avg_conf = sum(w["confidence"] for w in words) / len(words)
#     meaningful = [l for l in lines if len(l.split()) >= 2]
#     coverage = len(meaningful) / max(len(lines), 1)
#     return round(min(avg_conf * coverage * 100, 99.9), 2)

# # ============================================================
# # OVERLAY
# # ============================================================
# def draw_overlay(img, words):
#     out = img.copy()
#     for w in words:
#         pts = np.array(w["bbox"], dtype=np.int32)
#         color = (0,200,0) if w["status"]=="perfect" else \
#                 (0,165,255) if w["status"]=="partial" else (0,0,255)
#         cv2.polylines(out, [pts], True, color, 2)
#     return out

# # ============================================================
# # UI
# # ============================================================
# st.title("Dynamic OCR ")

# if not st.session_state.uploaded:
#     st.file_uploader(
#         "Upload Images / PDFs / ZIP",
#         ["png","jpg","jpeg","tif","pdf","zip"],
#         accept_multiple_files=True,
#         key="uploader",
#         on_change=handle_upload
#     )
#     st.stop()

# b1, b2 = st.columns(2)
# with b1:
#     if st.button("⬅ Back"):
#         st.session_state.uploaded = False
#         st.session_state.files = []
#         st.session_state.results = {}
#         st.experimental_rerun()

# with b2:
#     if st.button("✖ Reset"):
#         st.session_state.clear()
#         st.experimental_rerun()

# st.markdown("---")

# for idx, file in enumerate(st.session_state.files, start=1):

#     st.markdown(f"## 📄 {idx}. {file.name}")

#     if file.name not in st.session_state.results:
#         with st.spinner(f"OCR processing {file.name}..."):
#             mode, text_lines, images = load_input(
#                 file.getvalue(), file.type
#             )

#             final_lines, words = [], []
#             if mode == "text":
#                 final_lines = post_process_output(
#                     [normalize_text(l) for l in text_lines if l.strip()]
#                 )
#             else:
#                 for img in images:
#                     lines, w = run_ocr_pipeline(img)
#                     final_lines.extend(lines)
#                     words.extend(w)

#             accuracy = calculate_document_accuracy(words, final_lines)

#             st.session_state.results[file.name] = {
#                 "mode": mode,
#                 "lines": final_lines,
#                 "images": images,
#                 "words": words,
#                 "accuracy": accuracy
#             }

#     result = st.session_state.results[file.name]

#     if result["accuracy"] >= 90:
#         st.success(f"✅ Accuracy Score: {result['accuracy']}% (Very High)")
#     elif result["accuracy"] >= 75:
#         st.warning(f"🟠 Accuracy Score: {result['accuracy']}% (Good)")
#     else:
#         st.error(f"🔴 Accuracy Score: {result['accuracy']}% (Review Recommended)")

#     col_left, col_right = st.columns([1,1])

#     with col_left:
#         show_overlay = st.checkbox("Show Confidence Overlay", key=f"ov_{idx}")
#         if result["images"]:
#             img = result["images"][0]
#             if show_overlay:
#                 img = draw_overlay(img, result["words"])
#             st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
#                      use_column_width=True)

#     with col_right:
#         tab_parse, tab_extract = st.tabs(
#             [f"Parse {idx}", f"Extract {idx}"]
#         )

#         with tab_parse:
#             st.text_area(
#                 "Final OCR Output",
#                 "\n".join(result["lines"]),
#                 height=300,
#                 key=f"txt_{idx}"
#             )

#         with tab_extract:
#             st.json({
#                 "file": file.name,
#                 "mode": result["mode"],
#                 "accuracy_score": result["accuracy"],
#                 "total_lines": len(result["lines"]),
#                 "lines": result["lines"]
#             })

#     st.markdown("---")

# st.caption("✔ Regex bug fixed | ✔ Accuracy score added | ✔ No functionality removed")

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
import zipfile
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
ALLOWED_EXT = (".png", ".jpg", ".jpeg", ".tif", ".pdf")

# ============================================================
# CACHED OCR MODEL
# ============================================================
@st.cache_resource
def get_easyocr_reader():
    return easyocr.Reader(["en"], gpu=False)

EASY_READER = get_easyocr_reader()

# ============================================================
# SESSION STATE
# ============================================================
defaults = {"uploaded": False, "files": [], "results": {}}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ============================================================
# IN-MEMORY FILE (ZIP SUPPORT)
# ============================================================
class InMemoryFile:
    def __init__(self, name, data, mime):
        self.name = name
        self._data = data
        self.type = mime

    def getvalue(self):
        return self._data

# ============================================================
# UPLOAD HANDLER
# ============================================================
def handle_upload():
    uploaded = st.session_state.uploader
    files = []

    for f in uploaded:
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(f.getvalue())) as z:
                for info in z.infolist():
                    if info.filename.lower().endswith(ALLOWED_EXT):
                        data = z.read(info)
                        mime = (
                            "application/pdf"
                            if info.filename.lower().endswith(".pdf")
                            else "image"
                        )
                        files.append(InMemoryFile(info.filename, data, mime))
        elif f.name.lower().endswith(ALLOWED_EXT):
            files.append(f)

    if files:
        st.session_state.files = files
        st.session_state.results = {}
        st.session_state.uploaded = True

# ============================================================
# IMAGE ENHANCEMENT
# ============================================================
def enhance_for_blur_safe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.createCLAHE(3.0, (8, 8)).apply(gray)
    return gray

def downscale_for_ocr(img, max_width=1800):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def normalize_text(t):
    if not t:
        return ""
    t = re.sub(r"[^\x20-\x7E]", "", t)
    return re.sub(r"\s{2,}", " ", t).strip()

# ============================================================
# CONFIDENCE & FILTERING
# ============================================================
def classify_conf(c):
    if c >= 0.90:
        return "perfect"
    if c >= 0.70:
        return "partial"
    return "poor"

def filter_low_confidence(words, min_conf=0.55):
    return [w for w in words if w["confidence"] >= min_conf]

# ============================================================
# GARBAGE LINE SUPPRESSION (NEW – FIXES YOUR ISSUE)
# ============================================================
def is_garbage_line(line):
    line = line.strip()

    if len(line) < 4:
        return True

    # Repeated characters like eee, aaa
    if re.fullmatch(r"(.)\1{2,}", line.replace(" ", "")):
        return True

    words = line.split()
    short_words = [w for w in words if len(w) <= 2]

    # Mostly meaningless short words
    if len(words) >= 3 and len(short_words) / len(words) > 0.6:
        return True

    # No strong alphabetic token and no digit
    if not re.search(r"[A-Za-z]{3,}", line) and not re.search(r"\d", line):
        return True

    return False

# ============================================================
# CONTEXT-AWARE CORRECTIONS
# ============================================================
CHAR_CONFUSIONS = {
    "O": "0",
    "I": "1",
    "l": "1",
    "S": "5",
    "B": "8",
}

def fix_by_context(text):
    if re.search(r"\d", text):
        for k, v in CHAR_CONFUSIONS.items():
            text = text.replace(k, v)
    return text

def normalize_fields(line):
    line = re.sub(
        r"(DOB[:\s]*)I(\d{3})[-/]I(\d{2})[-/]I(\d{2})",
        r"\g<1>1\2-\3-\4",
        line,
        flags=re.I
    )

    line = re.sub(
        r"(PREMIUM[:\s]*)(\$?\s*\d+(\.\d{2})?)",
        r"\g<1>\2",
        line,
        flags=re.I
    )

    return line

def ml_correct(text):
    return text

def post_process_output(lines):
    clean = []
    for l in lines:
        l = fix_by_context(l)
        l = normalize_fields(l)
        l = ml_correct(l)

        if not is_garbage_line(l):
            clean.append(l)

    return clean

# ============================================================
# OCR PIPELINE
# ============================================================
def run_ocr_pipeline(img):
    img = downscale_for_ocr(img)
    gray = enhance_for_blur_safe(img)

    easy_results = EASY_READER.readtext(gray, detail=1)

    words, easy_text = [], []
    for bbox, text, conf in easy_results:
        text = normalize_text(text)
        if not text:
            continue
        words.append({
            "text": text,
            "confidence": round(conf, 2),
            "status": classify_conf(conf),
            "bbox": bbox
        })
        easy_text.append(text)

    words = filter_low_confidence(words)

    tess_text = pytesseract.image_to_string(
        Image.fromarray(gray), config=TESS_CONFIG
    )
    tess_lines = [
        normalize_text(l) for l in tess_text.splitlines() if l.strip()
    ]

    final_lines = list(dict.fromkeys(tess_lines + easy_text))
    final_lines = post_process_output(final_lines)

    return final_lines, words

# ============================================================
# PDF HANDLING
# ============================================================
def extract_text_pdf(pdf_bytes):
    try:
        return extract_text(io.BytesIO(pdf_bytes))
    except:
        return ""

def load_input(file_bytes, file_type):
    images = []

    if file_type == "application/pdf":
        pages = convert_from_bytes(
            file_bytes, dpi=150, poppler_path=POPPLER_PATH
        )
        images = [
            cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)
            for p in pages
        ]

        txt = extract_text_pdf(file_bytes)
        if len(txt.strip()) > 200:
            return "text", txt.splitlines(), images

        return "image", None, images

    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    return "image", None, [img]

# ============================================================
# ACCURACY SCORE
# ============================================================
def calculate_document_accuracy(words, lines):
    if not words:
        return 0
    avg_conf = sum(w["confidence"] for w in words) / len(words)
    meaningful = [l for l in lines if len(l.split()) >= 2]
    coverage = len(meaningful) / max(len(lines), 1)
    return round(min(avg_conf * coverage * 100, 99.9), 2)

# ============================================================
# OVERLAY
# ============================================================
def draw_overlay(img, words):
    out = img.copy()
    for w in words:
        pts = np.array(w["bbox"], dtype=np.int32)
        color = (0,200,0) if w["status"]=="perfect" else \
                (0,165,255) if w["status"]=="partial" else (0,0,255)
        cv2.polylines(out, [pts], True, color, 2)
    return out

# ============================================================
# UI
# ============================================================
st.title("Dynamic OCR")

if not st.session_state.uploaded:
    st.file_uploader(
        "Upload Images / PDFs / ZIP",
        ["png","jpg","jpeg","tif","pdf","zip"],
        accept_multiple_files=True,
        key="uploader",
        on_change=handle_upload
    )
    st.stop()

b1, b2 = st.columns(2)
with b1:
    if st.button("⬅ Back"):
        st.session_state.uploaded = False
        st.session_state.files = []
        st.session_state.results = {}
        st.experimental_rerun()

with b2:
    if st.button("✖ Reset"):
        st.session_state.clear()
        st.experimental_rerun()

st.markdown("---")

for idx, file in enumerate(st.session_state.files, start=1):

    st.markdown(f"## 📄 {idx}. {file.name}")

    if file.name not in st.session_state.results:
        with st.spinner(f"OCR processing {file.name}..."):
            mode, text_lines, images = load_input(
                file.getvalue(), file.type
            )

            final_lines, words = [], []
            if mode == "text":
                final_lines = post_process_output(
                    [normalize_text(l) for l in text_lines if l.strip()]
                )
            else:
                for img in images:
                    lines, w = run_ocr_pipeline(img)
                    final_lines.extend(lines)
                    words.extend(w)

            accuracy = calculate_document_accuracy(words, final_lines)

            st.session_state.results[file.name] = {
                "mode": mode,
                "lines": final_lines,
                "images": images,
                "words": words,
                "accuracy": accuracy
            }

    result = st.session_state.results[file.name]

    if result["accuracy"] >= 90:
        st.success(f"✅ Accuracy Score: {result['accuracy']}% (Very High)")
    elif result["accuracy"] >= 75:
        st.warning(f"🟠 Accuracy Score: {result['accuracy']}% (Good)")
    else:
        st.error(f"🔴 Accuracy Score: {result['accuracy']}% (Review Recommended)")

    col_left, col_right = st.columns([1,1])

    with col_left:
        show_overlay = st.checkbox("Show Confidence Overlay", key=f"ov_{idx}")
        if result["images"]:
            img = result["images"][0]
            if show_overlay:
                img = draw_overlay(img, result["words"])
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                     use_column_width=True)

    with col_right:
        tab_parse, tab_extract = st.tabs(
            [f"Parse {idx}", f"Extract {idx}"]
        )

        with tab_parse:
            st.text_area(
                "Final OCR Output",
                "\n".join(result["lines"]),
                height=300,
                key=f"txt_{idx}"
            )

        with tab_extract:
            st.json({
                "file": file.name,
                "mode": result["mode"],
                "accuracy_score": result["accuracy"],
                "total_lines": len(result["lines"]),
                "lines": result["lines"]
            })

    st.markdown("---")

st.caption("✔ Garbage lines removed | ✔ Accuracy improved | ✔ No functionality removed")
