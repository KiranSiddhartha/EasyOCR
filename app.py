# # app.py
# import streamlit as st
# import numpy as np
# import cv2
# import pytesseract
# import easyocr
# import json
# from PIL import Image
# from preprocess import preprocess_candidates   # updated function name

# # ---------------------- TESSERACT CONFIG ----------------------
# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Users\xd5914\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
# )
# TESS_CONFIG = "--oem 1 --psm 6 -c preserve_interword_spaces=1"

# # ---------------------- FILTERS ----------------------
# def bbox_is_valid(bbox, min_w=10, min_h=8, max_ratio=5.0):
#     xs = [p[0] for p in bbox]
#     ys = [p[1] for p in bbox]
#     w = max(xs) - min(xs)
#     h = max(ys) - min(ys)
#     if w < min_w or h < min_h:
#         return False
#     if h / w > max_ratio:
#         return False
#     return True

# def scale_bbox(bbox, src_shape, dst_shape):
#     src_h, src_w = src_shape[:2]
#     dst_h, dst_w = dst_shape[:2]
#     sx = dst_w / src_w
#     sy = dst_h / src_h
#     return [[int(x * sx), int(y * sy)] for x, y in bbox]

# def draw_overlay_scaled(orig_bgr, word_json, src_shape):
#     img = orig_bgr.copy()
#     for w in word_json:
#         bbox = scale_bbox(w["bbox"], src_shape, img.shape)
#         pts = np.array(bbox, dtype=np.int32)
#         x1 = int(np.min(pts[:, 0]))
#         y1 = int(np.min(pts[:, 1]))
#         x2 = int(np.max(pts[:, 0]))
#         y2 = int(np.max(pts[:, 1]))
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         # display short label for readability (limit length)
#         label = w["text"] if len(w["text"]) <= 20 else (w["text"][:17] + "...")
#         cv2.putText(img, label, (x1, max(0, y1 - 6)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
#     return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # Merge easyocr words to lines based on y coordinate
# def merge_easyocr_lines(words):
#     if not words:
#         return []
#     sorted_words = sorted(words, key=lambda w: sum(p[1] for p in w["bbox"]) / 4)
#     lines = []
#     acc = []
#     prev_y = None
#     for w in sorted_words:
#         y = sum(p[1] for p in w["bbox"]) / 4
#         if prev_y is None or abs(y - prev_y) < 22:
#             acc.append(w["text"])
#         else:
#             lines.append(" ".join(acc))
#             acc = [w["text"]]
#         prev_y = y
#     if acc:
#         lines.append(" ".join(acc))
#     return lines

# # Smart merge lines (clean EasyOCR + filtered Tesseract)
# def smart_merge_lines(tess_lines, easy_lines):
#     merged = []

#     # Normalize lines (strip empty)
#     easy_clean = [ln.strip() for ln in easy_lines if ln.strip()]
#     tess_clean = [ln.strip() for ln in tess_lines if ln.strip()]

#     # Filter garbage lines
#     def is_clean(line):
#         if not line or line.strip() == "":
#             return False

#         # too many symbols → noise
#         symbols = sum(not ch.isalnum() and ch not in " .:-/" for ch in line)
#         if symbols > len(line) * 0.40:
#             return False

#         # low alpha ratio → distorted text
#         alpha_ratio = sum(ch.isalpha() for ch in line) / max(1, len(line))
#         if alpha_ratio < 0.40:
#             return False

#         return True

#     # Step 1 — EasyOCR as primary
#     final_easy = [ln for ln in easy_clean if is_clean(ln)]
#     merged.extend(final_easy)

#     # Step 2 — Tesseract fallback (only clean + non-duplicate lines)
#     final_tess = []
#     for ln in tess_clean:
#         if not is_clean(ln):
#             continue

#         # Skip if EasyOCR has similar line
#         if any(ln.lower() in e.lower() or e.lower() in ln.lower() for e in final_easy):
#             continue

#         final_tess.append(ln)

#     # Step 3 — Merge unique lines
#     for ln in final_tess:
#         if ln not in merged:
#             merged.append(ln)

#     return merged


# # Score candidate using pytesseract confidence and char count
# def score_candidate_for_ocr(pil_img):
#     """
#     Returns a score for the image candidate using Tesseract image_to_data.
#     Higher is better. Score = mean_confidence * log(1 + num_chars)
#     """
#     try:
#         # run tesseract in simple config to get words/conf
#         data = pytesseract.image_to_data(pil_img, config="--psm 6", output_type=pytesseract.Output.DICT)
#         confs = []
#         num_chars = 0
#         for txt, conf in zip(data.get('text', []), data.get('conf', [])):
#             if txt and txt.strip():
#                 num_chars += len(txt.strip())
#             # conf may be -1 for non-text; include only >=0
#             try:
#                 c = float(conf)
#             except Exception:
#                 continue
#             if c >= 0:
#                 confs.append(c)
#         if len(confs) == 0:
#             return 0.0
#         mean_conf = float(np.mean(confs))
#         score = mean_conf * np.log1p(num_chars)
#         return score
#     except Exception:
#         return 0.0

# # ---------------------- APP UI & Flow ----------------------
# st.set_page_config(layout="wide", page_title="Enhanced OCR Engine (Max Accuracy)")
# st.title("📄 Enhanced OCR — Max-Accuracy (deblur + RL candidates)")

# uploaded = st.file_uploader("Upload document image", ["png", "jpg", "jpeg", "tif", "tiff"])
# if not uploaded:
#     st.info("Upload a document image to begin.")
#     st.stop()

# file_bytes = uploaded.read()

# # PREPROCESS: generate candidates (upscaled color + several enhanced grayscale candidates)
# orig_color, candidates = preprocess_candidates(file_bytes)
# # candidates: list of uint8 grayscale images (CLAHE/sharpened/restored)

# # Evaluate each candidate using a fast Tesseract score
# best_score = -1.0
# best_idx = 0
# best_pil = None
# for i, cand in enumerate(candidates):
#     pil = Image.fromarray(cand)
#     sc = score_candidate_for_ocr(pil)
#     # small penalty for extremely small mean intensity (all black) or too noisy
#     if sc > best_score:
#         best_score = sc
#         best_idx = i
#         best_pil = pil

# # If none produced any score, fall back to the first candidate
# if best_pil is None:
#     best_pil = Image.fromarray(candidates[0])
#     best_idx = 0

# # Use best candidate (grayscale) for OCR
# clahe_img = np.array(best_pil) if isinstance(best_pil, Image.Image) else candidates[best_idx]

# # Tesseract OCR (multiline)
# tess_text = pytesseract.image_to_string(Image.fromarray(clahe_img), config=TESS_CONFIG)
# tess_lines = [ln.strip() for ln in tess_text.splitlines() if ln.strip()]

# # EasyOCR (word-level) on the selected candidate
# reader = easyocr.Reader(['en'], gpu=False)
# easy_raw = reader.readtext(clahe_img, detail=1)

# # Relaxed confidence to keep more text in tough images
# easy_raw = [x for x in easy_raw if x[2] >= 0.40]

# # Filter by bbox geometry
# easy_raw = [x for x in easy_raw if bbox_is_valid(x[0], min_w=8, min_h=6, max_ratio=6.0)]

# word_json = [
#     {"text": t, "confidence": float(c), "bbox": [[int(a), int(b)] for a, b in bbox]}
#     for (bbox, t, c) in easy_raw
# ]

# easy_lines = merge_easyocr_lines(word_json)
# merged_text = "\n".join(tess_lines + easy_lines)

# # final_lines = smart_merge_lines(tess_lines, easy_lines)
# # merged_text = "\n".join(final_lines)

# # UI: two columns (image left, outputs right)
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Image with Bounding Boxes (best candidate selected)")
#     overlay = draw_overlay_scaled(orig_color, word_json, src_shape=candidates[best_idx].shape)
#     st.image(overlay, caption=f"Selected candidate #{best_idx} (score={best_score:.2f})")

# with col2:
#     st.subheader("Extracted Text (Merged Tesseract + EasyOCR)")
#     st.text_area("OCR Output", merged_text, height=350)

#     st.subheader("Word-Level JSON")
#     st.json(word_json)

#     st.subheader("Tesseract Lines")
#     st.json(tess_lines)

#     st.subheader("EasyOCR Lines")
#     st.json(easy_lines)

#     # allow saving selected candidate image and JSON
#     if st.button("Save chosen candidate + JSON"):
#         import os, datetime
#         os.makedirs("output", exist_ok=True)
#         ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         # save candidate image
#         cv2.imwrite(f"output/candidate_{ts}_{best_idx}.png", candidates[best_idx])
#         # save overlay image
#         overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(f"output/overlay_{ts}.png", overlay_bgr)
#         # save json
#         out = {
#             "merged_text": merged_text,
#             "tesseract_lines": tess_lines,
#             "easyocr_lines": easy_lines,
#             "words": word_json,
#             "selected_candidate_index": best_idx,
#             "score": best_score
#         }
#         with open(f"output/ocr_{ts}.json", "w", encoding="utf8") as f:
#             json.dump(out, f, indent=2, ensure_ascii=False)
#         st.success(f"Saved candidate_{ts}_{best_idx}.png, overlay_{ts}.png, ocr_{ts}.json in output/")


 
# import streamlit as st
# import numpy as np
# import cv2
# import pytesseract
# import easyocr
# import re
# from PIL import Image
# from preprocess import preprocess_candidates

# # ------------------------------------------------------------
# # TESSERACT CONFIG
# # ------------------------------------------------------------
# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Users\xd5914\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
# )

# # Sparse text mode → best recall for faint / broken text
# TESS_CONFIG = "--oem 1 --psm 11"

# # ------------------------------------------------------------
# # NORMALIZE OCR LINES (POST-OCR CLEANUP)
# # ------------------------------------------------------------
# def normalize_line(line: str):
#     """
#     Light OCR cleanup.
#     Removes garbage but NEVER deletes valid information.
#     """
#     line = line.strip()

#     # collapse spaces
#     line = re.sub(r"\s{2,}", " ", line)

#     # common OCR symbol fixes
#     line = line.replace(")", ":")
#     line = line.replace("Oo0", "00")
#     line = line.replace("O0", "00")
#     line = line.replace("i0u", "100")

#     # remove repeated junk symbols
#     line = re.sub(r"[><]{2,}", "", line)

#     # keep only readable lines
#     alpha_ratio = sum(c.isalnum() for c in line) / max(1, len(line))
#     if alpha_ratio < 0.45:
#         return None

#     if len(line) < 4:
#         return None

#     return line


# # ------------------------------------------------------------
# # MERGE EASYOCR WORDS → MULTILINE TEXT
# # ------------------------------------------------------------
# def merge_easyocr_lines(words, y_tol=18):
#     if not words:
#         return []

#     words = sorted(
#         words,
#         key=lambda w: (
#             sum(p[1] for p in w["bbox"]) / 4,  # avg Y
#             sum(p[0] for p in w["bbox"]) / 4   # avg X
#         )
#     )

#     lines = []
#     current = []
#     last_y = None

#     for w in words:
#         y = sum(p[1] for p in w["bbox"]) / 4

#         if last_y is None or abs(y - last_y) <= y_tol:
#             current.append(w["text"])
#         else:
#             lines.append(" ".join(current))
#             current = [w["text"]]

#         last_y = y

#     if current:
#         lines.append(" ".join(current))

#     return lines


# # ------------------------------------------------------------
# # STREAMLIT UI
# # ------------------------------------------------------------
# st.set_page_config(layout="wide", page_title="OCR POC – Dynamic Multiline Extraction")
# st.title("📄 OCR POC — Dynamic Multiline Text Extraction")

# uploaded = st.file_uploader(
#     "Upload an image",
#     ["png", "jpg", "jpeg", "tif", "tiff"]
# )

# if not uploaded:
#     st.stop()

# file_bytes = uploaded.read()

# # ------------------------------------------------------------
# # PREPROCESS IMAGE
# # ------------------------------------------------------------
# orig_color, candidates = preprocess_candidates(file_bytes)
# display_img = cv2.cvtColor(orig_color, cv2.COLOR_BGR2RGB)

# col_img, col_out = st.columns(2)

# with col_img:
#     st.subheader("Uploaded Image")
#     st.image(display_img, use_column_width=True)

# with col_out:
#     status = st.empty()
#     output = st.empty()
#     status.info("Running OCR… please wait")

# # ------------------------------------------------------------
# # SELECT BEST CANDIDATE (FAST SCORE)
# # ------------------------------------------------------------
# def score_candidate(img):
#     try:
#         data = pytesseract.image_to_data(
#             Image.fromarray(img),
#             config="--psm 6",
#             output_type=pytesseract.Output.DICT
#         )
#         confs = [float(c) for c in data["conf"] if c != "-1"]
#         return np.mean(confs) if confs else 0.0
#     except:
#         return 0.0

# best_img = max(candidates, key=score_candidate)

# # ------------------------------------------------------------
# # OCR – TESSERACT
# # ------------------------------------------------------------
# tess_text = pytesseract.image_to_string(
#     Image.fromarray(best_img),
#     config=TESS_CONFIG
# )

# tess_lines = [
#     ln.strip()
#     for ln in tess_text.splitlines()
#     if ln.strip()
# ]

# # ------------------------------------------------------------
# # OCR – EASYOCR (PARAGRAPH MODE)
# # ------------------------------------------------------------
# reader = easyocr.Reader(["en"], gpu=False)

# easy_raw = reader.readtext(
#     best_img,
#     detail=1,
#     paragraph=True
# )

# word_json = []

# for item in easy_raw:
#     if len(item) == 3:
#         bbox, text, conf = item
#     else:
#         bbox, text = item
#         conf = 0.0

#     word_json.append({
#         "text": text.strip(),
#         "confidence": float(conf),
#         "bbox": [[int(a), int(b)] for a, b in bbox]
#     })

# easy_lines = merge_easyocr_lines(word_json)

# # ------------------------------------------------------------
# # FINAL MERGE (MAX RECALL → CLEAN)
# # ------------------------------------------------------------
# final_lines = []
# final_lines.extend(tess_lines)
# final_lines.extend(easy_lines)

# # de-duplicate (preserve order)
# seen = set()
# final_lines = [l for l in final_lines if not (l in seen or seen.add(l))]

# # normalize lines
# clean_lines = []
# for ln in final_lines:
#     cleaned = normalize_line(ln)
#     if cleaned:
#         clean_lines.append(cleaned)

# # final de-duplication
# seen = set()
# clean_lines = [l for l in clean_lines if not (l in seen or seen.add(l))]

# merged_text = "\n".join(clean_lines)

# # ------------------------------------------------------------
# # OUTPUT
# # ------------------------------------------------------------
# status.empty()

# output.subheader("Extracted Text (Dynamic Multiline)")
# output.text_area("", merged_text, height=450)

# st.markdown("---")
# st.markdown(f"**Total clean lines extracted:** {len(clean_lines)}")


import streamlit as st
import numpy as np
import cv2
import easyocr
import pytesseract
import re
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

# ------------------------------------------------------------
# PYTESSERACT CONFIG
# ------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\xd5914\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)
TESS_CONFIG = "--oem 3 --psm 6 -c preserve_interword_spaces=1"

# ------------------------------------------------------------
# IMAGE ENHANCEMENT FOR OCR (CRITICAL)
# ------------------------------------------------------------
def enhance_for_ocr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # upscale (helps blur)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # edge-preserving denoise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # mild sharpening
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.addWeighted(gray, 1.3, blur, -0.3, 0)

    return gray

# ------------------------------------------------------------
# OCR TEXT NORMALIZATION (DYNAMIC)
# ------------------------------------------------------------
def normalize_ocr_text(text):
    t = text.strip()
    t = re.sub(r"[^\x20-\x7E]", "", t)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"[;']", ":", t)
    t = re.sub(r"^[#@>{}\[\]]+", "", t)

    # digit fixes only in numeric context
    t = re.sub(r"(?<=\d)[Oo](?=\d)", "0", t)
    t = re.sub(r"(?<=\d)[Il](?=\d)", "1", t)

    alpha_ratio = sum(c.isalnum() for c in t) / max(1, len(t))
    if alpha_ratio < 0.45 or len(t) < 3:
        return None

    return t

# ------------------------------------------------------------
# LINE QUALITY CLASSIFICATION (SUMMARY)
# ------------------------------------------------------------
def classify_quality(line):
    alpha = sum(c.isalnum() for c in line)
    ratio = alpha / max(1, len(line))
    return "perfect" if ratio >= 0.75 else "partial"

# ------------------------------------------------------------
# EASYOCR LINE MERGING
# ------------------------------------------------------------
def merge_easyocr_lines(results, y_tol=18):
    results = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))
    lines, current, last_y = [], [], None

    for bbox, text, conf in results:
        y = bbox[0][1]
        if last_y is None or abs(y - last_y) <= y_tol:
            current.append(text)
        else:
            lines.append(" ".join(current))
            current = [text]
        last_y = y

    if current:
        lines.append(" ".join(current))

    return lines

# ------------------------------------------------------------
# SIMILARITY + ARBITRATION
# ------------------------------------------------------------
def normalize_for_compare(text):
    return re.sub(r"[^\w\s:-]", "", text.lower())

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def line_score(line):
    return sum(c.isalnum() for c in line) + len(line) * 0.2

def merge_best_lines(tess, easy, threshold=0.60):
    final, used_easy = [], set()

    for t in tess:
        best, best_score = t, line_score(t)
        for i, e in enumerate(easy):
            if i in used_easy:
                continue
            if similarity(normalize_for_compare(t), normalize_for_compare(e)) >= threshold:
                if line_score(e) > best_score:
                    best, best_score = e, line_score(e)
                used_easy.add(i)
        final.append(best)

    for i, e in enumerate(easy):
        if i not in used_easy:
            final.append(e)

    return final

# ------------------------------------------------------------
# OCR RUNNERS
# ------------------------------------------------------------
def run_easyocr(gray):
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return reader.readtext(
        gray,
        detail=1,
        paragraph=False,
        text_threshold=0.5,
        low_text=0.3,
        link_threshold=0.4
    )

def run_tesseract(gray):
    txt = pytesseract.image_to_string(Image.fromarray(gray), config=TESS_CONFIG)
    return [l.strip() for l in txt.splitlines() if l.strip()]

# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Dynamic OCR – Improved Accuracy")
st.title("📄 Dynamic OCR – Accuracy Improved")

uploaded = st.file_uploader("Upload an image", ["png", "jpg", "jpeg", "tif"])

if not uploaded:
    st.stop()

file_bytes = np.frombuffer(uploaded.read(), np.uint8)
orig = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if orig is None:
    st.error("Image load failed")
    st.stop()

enhanced = enhance_for_ocr(orig)

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📤 Uploaded Image")
    st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), use_column_width=True)

with col_right:
    st.subheader("📊 Extraction Summary & Output")

    with st.spinner("Running OCR engines…"):
        with ThreadPoolExecutor(max_workers=2) as ex:
            easy_res = ex.submit(run_easyocr, enhanced).result()
            tess_res = ex.submit(run_tesseract, enhanced).result()

        easy_lines = [
            normalize_ocr_text(l)
            for l in merge_easyocr_lines(easy_res)
            if normalize_ocr_text(l)
        ]

        tess_lines = [
            normalize_ocr_text(l)
            for l in tess_res
            if normalize_ocr_text(l)
        ]

        final_lines = merge_best_lines(tess_lines, easy_lines)

    # -------- Extraction Summary --------
    perfect = sum(1 for l in final_lines if classify_quality(l) == "perfect")
    partial = len(final_lines) - perfect
    progress = perfect / max(1, len(final_lines))

    st.markdown("### 📈 Extraction Summary")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Perfectly Extracted**")
        st.markdown(f"<h2 style='color:green'>{perfect}</h2>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"**Partially Extracted**")
        st.markdown(f"<h2 style='color:orange'>{partial}</h2>", unsafe_allow_html=True)

    st.progress(progress)

    # -------- Final Output --------
    st.markdown("### 📄 Processed Document")
    st.text_area("Final OCR Output", "\n".join(final_lines), height=320)

st.caption("✔ Image enhancement | ✔ Parallel OCR | ✔ Dynamic | ✔ Accuracy arbitration")
