# superres.py
import cv2
import numpy as np


def upscale(img, scale=1.8):
    """Gentle upscale (keeps stroke shape)"""
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


def light_sharpen(img):
    """Gentle unsharp mask"""
    blur = cv2.GaussianBlur(img, (5,5), 0)
    return cv2.addWeighted(img, 1.3, blur, -0.3, 0)


def light_denoise(img):
    """Keeps text edges intact"""
    return cv2.bilateralFilter(img, 7, 50, 50)


def enhance_for_ocr(img_bgr):
    """
    Safe enhancement that does NOT destroy thin strokes.
    """
    # 1. Gentle upscale
    up = upscale(img_bgr, scale=1.8)

    # 2. Denoise
    den = light_denoise(up)

    # 3. Sharpen edges very lightly
    sharp = light_sharpen(den)

    # 4. CLAHE contrast
    gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)

    # DO NOT hard-threshold (kills strokes) → return soft grayscale
    return up, clahe_img
