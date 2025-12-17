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

# # ============================================================
# # SESSION STATE
# # ============================================================
# defaults = {
#     "uploaded": False,
#     "files": [],
#     "results": {},
#     "active_index": 0,
# }
# for k, v in defaults.items():
#     st.session_state.setdefault(k, v)

# # ============================================================
# # UPLOAD CALLBACK
# # ============================================================
# def handle_upload():
#     files = st.session_state.uploader
#     if not files:
#         return
#     st.session_state.files = files
#     st.session_state.active_index = 0
#     st.session_state.uploaded = True

# # ============================================================
# # OCR HELPERS
# # ============================================================
# def enhance_for_ocr(bgr):
#     gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
#     gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#     gray = cv2.bilateralFilter(gray, 9, 75, 75)
#     return cv2.createCLAHE(2.0, (8, 8)).apply(gray)

# def normalize_text(text):
#     if not text:
#         return ""
#     text = re.sub(r"[^\x20-\x7E]", "", text)
#     text = re.sub(r"\s{2,}", " ", text)
#     return text.strip()

# def reflow_lines(lines):
#     merged, buf = [], ""
#     for line in lines:
#         line = normalize_text(line)
#         if not line:
#             continue
#         if buf and not buf.endswith((".", ":")) and not line[0].isupper():
#             buf += " " + line
#         else:
#             if buf:
#                 merged.append(buf)
#             buf = line
#     if buf:
#         merged.append(buf)
#     return merged

# # ============================================================
# # WORD CONFIDENCE
# # ============================================================
# def classify_conf(conf):
#     if conf >= 0.90:
#         return "perfect"
#     elif conf >= 0.70:
#         return "partial"
#     return "poor"

# def run_easyocr_with_conf(img):
#     reader = easyocr.Reader(["en"], gpu=False, verbose=False)
#     results = reader.readtext(img, detail=1)
#     return [{
#         "text": t,
#         "confidence": round(c, 2),
#         "status": classify_conf(c),
#         "bbox": b
#     } for b, t, c in results]

# def draw_overlay(img, words):
#     overlay = img.copy()
#     for w in words:
#         pts = np.array(w["bbox"], dtype=np.int32)
#         color = (0,200,0) if w["status"]=="perfect" else \
#                 (0,165,255) if w["status"]=="partial" else (0,0,255)
#         cv2.polylines(overlay, [pts], True, color, 2)
#     return overlay

# # ============================================================
# # OCR ENGINES
# # ============================================================
# def run_easyocr(img):
#     reader = easyocr.Reader(["en"], gpu=False, verbose=False)
#     return [normalize_text(r) for r in reader.readtext(img, detail=0)]

# def run_tesseract(img):
#     txt = pytesseract.image_to_string(Image.fromarray(img), config=TESS_CONFIG)
#     return [normalize_text(l) for l in txt.splitlines() if l.strip()]

# # ============================================================
# # PDF HANDLING (FIXED)
# # ============================================================
# def extract_text_pdf(pdf_bytes):
#     try:
#         return extract_text(io.BytesIO(pdf_bytes))
#     except:
#         return ""

# def is_text_pdf(pdf_bytes, min_chars=200):
#     txt = extract_text_pdf(pdf_bytes)
#     return len(txt) >= min_chars, txt

# def load_input(file_bytes, file_type):
#     images = []
#     if file_type == "application/pdf":
#         pages = convert_from_bytes(
#             file_bytes, dpi=150, poppler_path=POPPLER_PATH
#         )
#         images = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages]

#         is_text, txt = is_text_pdf(file_bytes)
#         if is_text:
#             return "text", txt.splitlines(), images

#         return "image", None, images

#     img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
#     return "image", None, [img]

# # ============================================================
# # EXTRACTION SUMMARY
# # ============================================================
# def extraction_summary(lines):
#     perfect = partial = 0
#     for l in lines:
#         if ":" in l:
#             _, v = l.split(":", 1)
#             if v.strip():
#                 perfect += 1
#             else:
#                 partial += 1
#     return perfect, partial

# # ============================================================
# # UI
# # ============================================================
# st.title("📄 Dynamic OCR (Multi-File)")

# # ============================
# # UPLOAD
# # ============================
# if not st.session_state.uploaded:
#     st.file_uploader(
#         "Upload Image or PDF",
#         ["png","jpg","jpeg","tif","pdf"],
#         key="uploader",
#         accept_multiple_files=True,
#         on_change=handle_upload
#     )
#     st.stop()

# # ============================
# # FILE LIST
# # ============================
# file_labels = [f"📄 {f.name}" for f in st.session_state.files]
# st.radio(
#     "Uploaded Files",
#     range(len(file_labels)),
#     format_func=lambda i: file_labels[i],
#     key="active_index"
# )

# active_file = st.session_state.files[st.session_state.active_index]

# # ============================
# # OCR PROCESSING
# # ============================
# if active_file.name not in st.session_state.results:
#     with st.spinner(f"Processing {active_file.name}..."):
#         mode, text_lines, images = load_input(
#             active_file.getvalue(), active_file.type
#         )

#         final_lines, words = [], []
#         if mode == "text":
#             final_lines = reflow_lines(text_lines)
#         else:
#             for img in images:
#                 img = enhance_for_ocr(img)
#                 final_lines += run_easyocr(img)
#                 final_lines += run_tesseract(img)
#                 words += run_easyocr_with_conf(img)

#         st.session_state.results[active_file.name] = {
#             "mode": mode,
#             "lines": final_lines,
#             "images": images,
#             "words": words
#         }

# result = st.session_state.results[active_file.name]

# # ============================
# # LAYOUT
# # ============================
# left, right = st.columns([1,1])

# # ---------- LEFT ----------
# with left:
#     st.subheader("📤 Preview")
#     show_overlay = st.checkbox("Show Confidence Overlay")

#     if result["images"]:
#         img = result["images"][0]
#         if show_overlay and result["words"]:
#             img = draw_overlay(img, result["words"])
#         st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
#     else:
#         st.info("Preview not available")

# # ---------- RIGHT ----------
# with right:
#     p, q = extraction_summary(result["lines"])
#     st.markdown("### 📊 Extraction Summary")
#     c1, c2 = st.columns(2)
#     c1.success(f"Perfectly Extracted: {p}")
#     c2.warning(f"Partially Extracted: {q}")

#     tab_parse, tab_extract = st.tabs(["Parse", "Extract"])

#     with tab_parse:
#         st.text_area("Final OCR Output", "\n".join(result["lines"]), height=520)

#     with tab_extract:
#         st.json({
#             "file": active_file.name,
#             "mode": result["mode"],
#             "total_lines": len(result["lines"]),
#             "lines": result["lines"]
#         })

# st.caption("✔ PDF preview fixed | ✔ Text-based PDFs supported | ✔ No regression")



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
# CACHED OCR MODELS
# ============================================================
@st.cache_resource
def get_easyocr_reader():
    return easyocr.Reader(["en"], gpu=False)

EASY_READER = get_easyocr_reader()

# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "uploaded": False,
    "files": [],
    "results": {},
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ============================================================
# UPLOAD CALLBACK
# ============================================================
def handle_upload():
    files = st.session_state.uploader
    if files:
        st.session_state.files = files
        st.session_state.results = {}
        st.session_state.uploaded = True

# ============================================================
# IMAGE HELPERS
# ============================================================
def downscale_for_ocr(img, max_width=1800):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def enhance_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(2.0, (8, 8)).apply(gray)

def normalize_text(t):
    if not t:
        return ""
    t = re.sub(r"[^\x20-\x7E]", "", t)
    return re.sub(r"\s{2,}", " ", t).strip()

# ============================================================
# CONFIDENCE
# ============================================================
def classify_conf(c):
    if c >= 0.90:
        return "perfect"
    if c >= 0.70:
        return "partial"
    return "poor"

# ============================================================
# OCR PIPELINE
# ============================================================
def run_ocr_pipeline(img):
    img = downscale_for_ocr(img)
    gray = enhance_for_ocr(img)

    easy_results = EASY_READER.readtext(gray, detail=1)

    words, easy_text = [], []
    for bbox, text, conf in easy_results:
        text = normalize_text(text)
        if not text:
            continue
        easy_text.append(text)
        words.append({
            "text": text,
            "confidence": round(conf, 2),
            "status": classify_conf(conf),
            "bbox": bbox
        })

    tess_text = pytesseract.image_to_string(
        Image.fromarray(gray), config=TESS_CONFIG
    )
    tess_lines = [
        normalize_text(l)
        for l in tess_text.splitlines() if l.strip()
    ]

    final_lines = list(dict.fromkeys(tess_lines + easy_text))
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

    img = cv2.imdecode(
        np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR
    )
    return "image", None, [img]

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
st.title("Dynamic OCR – Batch View")

# ============================
# UPLOAD
# ============================
if not st.session_state.uploaded:
    st.file_uploader(
        "Upload Image or PDF",
        ["png","jpg","jpeg","tif","pdf"],
        accept_multiple_files=True,
        key="uploader",
        on_change=handle_upload
    )
    st.stop()

# ============================
# BACK / RESET
# ============================
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

# ============================
# BATCH RENDER (SCROLLABLE)
# ============================
for idx, file in enumerate(st.session_state.files, start=1):

    st.markdown(f"## 📄 {idx}. {file.name}")

    if file.name not in st.session_state.results:
        with st.spinner(f"OCR processing {file.name}..."):
            mode, text_lines, images = load_input(
                file.getvalue(), file.type
            )

            final_lines, words = [], []
            if mode == "text":
                final_lines = [
                    normalize_text(l) for l in text_lines if l.strip()
                ]
            else:
                for img in images:
                    lines, w = run_ocr_pipeline(img)
                    final_lines.extend(lines)
                    words.extend(w)

            st.session_state.results[file.name] = {
                "mode": mode,
                "lines": final_lines,
                "images": images,
                "words": words
            }

    result = st.session_state.results[file.name]

    col_left, col_right = st.columns([1,1])

    with col_left:
        show_overlay = st.checkbox(
            "Show Confidence Overlay",
            key=f"overlay_{idx}"
        )
        if result["images"]:
            img = result["images"][0]
            if show_overlay:
                img = draw_overlay(img, result["words"])
            st.image(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                use_column_width=True
            )

    with col_right:
        tab_parse, tab_extract = st.tabs(
            [f"Parse {idx}", f"Extract {idx}"]
        )

        with tab_parse:
            st.text_area(
                "Final OCR Output",
                "\n".join(result["lines"]),
                height=300,
                key=f"text_{idx}"
            )

        with tab_extract:
            st.json({
                "file": file.name,
                "mode": result["mode"],
                "total_lines": len(result["lines"]),
                "lines": result["lines"]
            })

    st.markdown("---")

st.caption("")
