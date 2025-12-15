# preprocess.py
import cv2
import numpy as np

def upscale(img, scale=2.0):
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

def bilateral_and_nlmeans(gray):
    # bilateral preserves edges; nl-means removes texture noise
    b = cv2.bilateralFilter(gray, 9, 75, 75)
    # fastNlMeansDenoising requires 8-bit gray
    nl = cv2.fastNlMeansDenoising(b, h=10, templateWindowSize=7, searchWindowSize=21)
    return nl

def unsharp_mask(img, kernel_size=(5,5), sigma=1.0, amount=1.0):
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

def clahe_enhance(gray, clipLimit=2.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(gray)

# --- Richardson-Lucy deconvolution (simple implementation) ---
def gaussian_psf(ksize, sigma):
    """Return normalized 2D Gaussian kernel as PSF."""
    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel

def rl_deconvolution(image, psf, iterations=20):
    """
    Richardson-Lucy deconvolution on a single-channel float image.
    image: float image (0..1)
    psf: 2D kernel, sums to 1
    """
    # pad to avoid wrap-around, use same shape
    image = image.astype(np.float32)
    eps = 1e-8
    estimate = np.full(image.shape, 0.5, dtype=np.float32)
    psf_mirror = psf[::-1, ::-1]
    for i in range(iterations):
        conv_est = cv2.filter2D(estimate, -1, psf, borderType=cv2.BORDER_REPLICATE)
        relative_blur = image / (conv_est + eps)
        error_est = cv2.filter2D(relative_blur, -1, psf_mirror, borderType=cv2.BORDER_REPLICATE)
        estimate = estimate * error_est
        # clamp to valid range to avoid numeric blowup
        estimate = np.clip(estimate, 0, 1)
    return estimate

def prepare_gray(img_bgr):
    """Return uint8 grayscale"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = (255 * (gray - gray.min()) / (gray.ptp() + 1e-9)).astype(np.uint8)
    return gray

def preprocess_candidates(file_bytes):
    """
    Accepts file bytes (as read from uploaded file).
    Returns:
      orig_color (upscaled color image),
      candidates_gray (list of uint8 grayscale images prepped for OCR)
    This function does NOT call OCR; selection happens in app.py.
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError("Could not decode image bytes in preprocess_candidates.")

    # 1) Gentle upscale color for display/overlay (we will scale bbox back accordingly)
    up_color = upscale(color, scale=2.0)

    # convert to grayscale and do light denoising
    gray0 = prepare_gray(up_color)
    den = bilateral_and_nlmeans(gray0)

    # Candidate images list
    candidates = []

    # Candidate A: denoised + CLAHE + slight unsharp (safe baseline)
    claheA = clahe_enhance(den, clipLimit=2.0, tileGridSize=(8,8))
    sharpA = unsharp_mask(claheA, kernel_size=(5,5), sigma=1.0, amount=0.7)
    candidates.append(sharpA)

    # Candidate B..D: RL deconvolution with varying gaussian PSF sigmas
    # Convert den to float normalized (0..1) for RL
    den_f = den.astype(np.float32) / 255.0

    for sigma, iters in [(0.8, 15), (1.2, 25), (1.8, 30)]:
        ksize = max(3, int(6 * sigma + 1))  # ensure odd, reasonable
        if ksize % 2 == 0:
            ksize += 1
        psf = gaussian_psf(ksize, sigma)
        try:
            restored_f = rl_deconvolution(den_f, psf, iterations=iters)
            # convert back to uint8
            restored = (np.clip(restored_f, 0.0, 1.0) * 255.0).astype(np.uint8)
            # small denoise to remove ringing
            restored = cv2.fastNlMeansDenoising(restored, h=7, templateWindowSize=7, searchWindowSize=21)
            # sharpen lightly then CLAHE
            restored = unsharp_mask(restored, kernel_size=(5,5), sigma=1.0, amount=0.9)
            restored = clahe_enhance(restored, clipLimit=2.0, tileGridSize=(8,8))
            candidates.append(restored)
        except Exception:
            # If RL fails, skip this candidate
            pass

    # Candidate E: morphological stroke width emphasis (helps faint strokes)
    # Use blackhat/top-hat to emphasize strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    blackhat = cv2.morphologyEx(den, cv2.MORPH_BLACKHAT, kernel)
    top = cv2.morphologyEx(den, cv2.MORPH_TOPHAT, kernel)
    combined = cv2.addWeighted(den, 0.8, blackhat, 0.6, 0)
    combined = cv2.addWeighted(combined, 0.9, top, 0.4, 0)
    combined = clahe_enhance(combined, clipLimit=2.0, tileGridSize=(8,8))
    combined = unsharp_mask(combined, kernel_size=(3,3), sigma=0.8, amount=0.8)
    candidates.append(combined)

    # keep candidates unique in size/type and return
    # ensure all candidates are uint8 grayscale
    uniq = []
    seen = set()
    for c in candidates:
        key = (c.shape, c.dtype, int(np.mean(c)))
        if key in seen:
            continue
        seen.add(key)
        if c.dtype != np.uint8:
            c = np.clip(c, 0, 255).astype(np.uint8)
        uniq.append(c)

    return up_color, uniq
 