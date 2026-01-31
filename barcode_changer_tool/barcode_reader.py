import os
import re
import threading
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Tuple, Set

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import zxingcpp

WEIGHTS = hf_hub_download(
    repo_id="Piero2411/YOLOV8s-Barcode-Detection",
    filename="YOLOV8s_Barcode_Detection.pt",
)
YOLO_MODEL = YOLO(WEIGHTS)
YOLO_LOCK = threading.Lock()

YOLO_CONF = 0.15
MAX_BOXES = 6
ANGLES = [0, 90, -90]
CROP_PAD_X = 0.60
CROP_PAD_Y = 0.20
UPSCALE = 3.0

ALLOWED_ZX_FORMATS: Set[zxingcpp.BarcodeFormat] = {
    zxingcpp.BarcodeFormat.EAN13,
    zxingcpp.BarcodeFormat.EAN8,
    zxingcpp.BarcodeFormat.UPCA,
    zxingcpp.BarcodeFormat.ITF,
}


class BarcodeStatus(str, Enum):
    BARCODE = "BARCODE"
    NONBARCODE = "NONBARCODE"
    UNSURE = "UNSURE"


def extract_upc_candidate(text: str) -> Optional[str]:
    if not text:
        return None
    raw = text.strip()
    if not raw:
        return None

    m = re.search(r"\(01\)\s*(\d{14})", raw)
    if m:
        return m.group(1)

    m = re.search(r"(?:\]C1|\]d2|\]Q3)?\s*01\s*(\d{14})", raw)
    if m:
        return m.group(1)

    cleaned = raw.replace(" ", "").replace("-", "")
    if cleaned.isdigit() and len(cleaned) in (8, 12, 13, 14):
        return cleaned

    return None


def _upc_a_checksum_ok(code: str) -> bool:
    if len(code) != 12 or not code.isdigit():
        return False
    digits = [int(c) for c in code]
    odd_sum = sum(digits[0:11:2])
    even_sum = sum(digits[1:11:2])
    total = odd_sum * 3 + even_sum
    check = (10 - (total % 10)) % 10
    return check == digits[11]


def _ean13_checksum_ok(code: str) -> bool:
    if len(code) != 13 or not code.isdigit():
        return False
    digits = [int(c) for c in code]
    odd_sum = sum(digits[0:12:2])
    even_sum = sum(digits[1:12:2])
    total = odd_sum + even_sum * 3
    check = (10 - (total % 10)) % 10
    return check == digits[12]


def _ean8_checksum_ok(code: str) -> bool:
    if len(code) != 8 or not code.isdigit():
        return False
    digits = [int(c) for c in code]
    odd_sum = digits[0] + digits[2] + digits[4] + digits[6]
    even_sum = digits[1] + digits[3] + digits[5]
    total = odd_sum * 3 + even_sum
    check = (10 - (total % 10)) % 10
    return check == digits[7]


def _gtin14_checksum_ok(code: str) -> bool:
    if len(code) != 14 or not code.isdigit():
        return False

    digits = [int(c) for c in code]
    body = digits[:-1]
    check_digit = digits[-1]

    total = 0
    for i, d in enumerate(reversed(body), start=1):
        weight = 3 if (i % 2 == 1) else 1
        total += d * weight

    check = (10 - (total % 10)) % 10
    return check == check_digit


def is_valid_upc_ean(code: str) -> bool:
    if not code or not code.isdigit():
        return False
    if len(code) == 12:
        return _upc_a_checksum_ok(code)
    if len(code) == 13:
        return _ean13_checksum_ok(code)
    if len(code) == 8:
        return _ean8_checksum_ok(code)
    if len(code) == 14:
        return _gtin14_checksum_ok(code)
    return False


def _rotate(img_bgr: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h))


def _iter_preprocessed_grays_fast(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    yield gray

    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)
        yield gray_clahe
    except Exception:
        gray_clahe = gray

    _t, otsu = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield otsu

    try:
        adap = cv2.adaptiveThreshold(
            gray_clahe,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )
        yield adap
    except Exception:
        pass

    blur = cv2.GaussianBlur(gray_clahe, (3, 3), 0)
    _t, otsu_blur = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield otsu_blur


def _zxing_decode(img_bgr: np.ndarray) -> Optional[str]:
    if img_bgr is None or img_bgr.size == 0:
        return None

    for gray in _iter_preprocessed_grays_fast(img_bgr):
        try:
            results = zxingcpp.read_barcodes(gray) or []
        except Exception:
            continue

        for res in results:
            if not getattr(res, "valid", True):
                continue

            fmt = getattr(res, "format", None)
            if fmt is not None and fmt not in ALLOWED_ZX_FORMATS:
                continue

            raw = (res.text or "").strip()
            if not raw:
                continue

            cand = extract_upc_candidate(raw)
            if cand and is_valid_upc_ean(cand):
                return cand

    return None


def yolo_detect_boxes(
    source: Union[str, Path, np.ndarray], conf: float = YOLO_CONF
) -> Tuple[np.ndarray, np.ndarray]:
    with YOLO_LOCK:
        res = YOLO_MODEL.predict(source=source, conf=conf, verbose=False)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=int), np.zeros((0,), dtype=float)

    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    order = confs.argsort()[::-1]
    return xyxy[order], confs[order]


def _decode_with_angles(img_bgr: np.ndarray) -> Optional[str]:
    for a in ANGLES:
        code = _zxing_decode(_rotate(img_bgr, a))
        if code:
            return code
    return None


def readBarcode_hf_status(
    image_path: Union[str, Path],
    *,
    require_large_region: bool = False,
    min_box_area_ratio: float = 0.0,
    min_box_max_dim_ratio: float = 0.0,
) -> tuple[BarcodeStatus, Optional[str]]:
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        return BarcodeStatus.NONBARCODE, None

    code = _decode_with_angles(img)
    if code:
        return BarcodeStatus.BARCODE, code

    xyxy, confs = yolo_detect_boxes(img, conf=YOLO_CONF)
    if xyxy.shape[0] == 0:
        return BarcodeStatus.NONBARCODE, None

    h, w = img.shape[:2]
    tried = 0

    for (x1, y1, x2, y2), c in zip(xyxy, confs):
        if tried >= MAX_BOXES:
            break

        bw = max(1, int(x2) - int(x1))
        bh = max(1, int(y2) - int(y1))

        pad_x = int(CROP_PAD_X * bw)
        pad_y = int(CROP_PAD_Y * bh)

        x1p = max(0, int(x1) - pad_x)
        y1p = max(0, int(y1) - pad_y)
        x2p = min(w, int(x2) + pad_x)
        y2p = min(h, int(y2) + pad_y)

        crop = img[y1p:y2p, x1p:x2p]
        if crop is None or crop.size == 0:
            continue

        tried += 1

        crop_big = cv2.resize(
            crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC
        )

        code = _decode_with_angles(crop_big)
        if code:
            return BarcodeStatus.BARCODE, code

    return BarcodeStatus.UNSURE, None


def readBarcode_hf(
    image_path: Union[str, Path],
    *,
    require_large_region: bool = False,
    min_box_area_ratio: float = 0.0,
    min_box_max_dim_ratio: float = 0.0,
) -> Optional[str]:
    status, code = readBarcode_hf_status(
        image_path,
        require_large_region=require_large_region,
        min_box_area_ratio=min_box_area_ratio,
        min_box_max_dim_ratio=min_box_max_dim_ratio,
    )
    return code if status == BarcodeStatus.BARCODE else None
