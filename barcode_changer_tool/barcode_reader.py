import os
import tempfile
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from pyrxing import read_barcode
from ultralytics import YOLO


weights = hf_hub_download(
    repo_id="Piero2411/YOLOV8s-Barcode-Detection",
    filename="YOLOV8s_Barcode_Detection.pt",
)
YOLO_MODEL = YOLO(weights)

try:
    BARCODE_DETECTOR = cv2.barcode.BarcodeDetector()
except Exception:
    BARCODE_DETECTOR = None


def decode_with_pyrxing_from_array(img_bgr: np.ndarray) -> Optional[str]:
    if img_bgr is None or img_bgr.size == 0:
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        cv2.imwrite(tmp_path, img_bgr)
        res = read_barcode(tmp_path)
        if res is not None and getattr(res, "text", None):
            return res.text.strip()
        return None
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _try_pyrxing_with_rotations_and_preproc(img_bgr: np.ndarray) -> Optional[str]:
    if img_bgr is None or img_bgr.size == 0:
        return None

    angles = [0, 90, 180, 270]
    for angle in angles:
        if angle == 0:
            rotated = img_bgr
        elif angle == 90:
            rotated = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(img_bgr, cv2.ROTATE_180)
        else:
            rotated = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

        code = decode_with_pyrxing_from_array(rotated)
        if code:
            print(
                f"[BARCODE] accepted (loose) '{code}' "
                f"from enhanced-full raw rot={angle}deg"
            )
            return code

        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        clahe = _apply_clahe(gray)
        clahe_bgr = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
        code = decode_with_pyrxing_from_array(clahe_bgr)
        if code:
            print(
                f"[BARCODE] accepted (clahe) '{code}' "
                f"from enhanced-full rot={angle}deg"
            )
            return code

    return None


def _refine_and_rectify_barcode_region(roi_bgr: np.ndarray) -> Optional[np.ndarray]:
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    abs_sobelx = np.abs(sobelx)

    col_profile = abs_sobelx.mean(axis=0)
    col_min = float(np.min(col_profile))
    col_range = float(np.ptp(col_profile))
    col_profile_norm = (col_profile - col_min) / (col_range + 1e-6)

    mask = col_profile_norm > 0.3
    if not np.any(mask):
        return None

    x_indices = np.where(mask)[0]
    x_min = int(max(x_indices.min() - 3, 0))
    x_max = int(min(x_indices.max() + 3, roi_bgr.shape[1] - 1))

    refined = roi_bgr[:, x_min:x_max]
    if refined.size == 0:
        return None

    rgray = cv2.cvtColor(refined, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        rgray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        35,
        10,
    )
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def _ean13_valid(code: str) -> bool:
    if len(code) != 13 or not code.isdigit():
        return False
    digits = [int(c) for c in code]
    check = digits[-1]
    body = digits[:-1]
    s_odd = sum(body[-1::-2])
    s_even = sum(body[-2::-2])
    calc = (10 - ((s_odd * 3 + s_even) % 10)) % 10
    return check == calc


def _upca_valid(code: str) -> bool:
    if len(code) != 12 or not code.isdigit():
        return False
    digits = [int(c) for c in code]
    check = digits[-1]
    body = digits[:-1]
    s_odd = sum(body[0::2])
    s_even = sum(body[1::2])
    calc = (10 - ((s_odd * 3 + s_even) % 10)) % 10
    return check == calc


def _ean8_valid(code: str) -> bool:
    if len(code) != 8 or not code.isdigit():
        return False
    digits = [int(c) for c in code]
    check = digits[-1]
    body = digits[:-1]
    s_odd = body[0] + body[2] + body[4]
    s_even = body[1] + body[3] + body[5]
    calc = (10 - ((s_odd * 3 + s_even) % 10)) % 10
    return check == calc


def _normalize_and_validate_numeric(text: str) -> Optional[str]:
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 13 and _ean13_valid(digits):
        return digits
    if len(digits) == 12 and _upca_valid(digits):
        return digits
    if len(digits) == 8 and _ean8_valid(digits):
        return digits
    return None


def _as_string_list(obj) -> list[str]:
    if obj is None:
        return []

    if isinstance(obj, str):
        return [obj]

    if isinstance(obj, (list, tuple)):
        out = []
        for x in obj:
            if x is None:
                continue
            if isinstance(x, str):
                out.append(x)
            else:
                out.append(str(x))
        return out

    if isinstance(obj, np.ndarray):
        return [str(x) for x in obj.ravel().tolist()]

    return [str(obj)]


def _decode_with_opencv_barcode(
    img_bgr: np.ndarray, context: str = ""
) -> Optional[str]:
    if BARCODE_DETECTOR is None:
        return None
    if img_bgr is None or img_bgr.size == 0:
        return None

    try:
        result = BARCODE_DETECTOR.detectAndDecode(img_bgr)
    except Exception as e:
        print(
            f"[BARCODE] OpenCV BarcodeDetector error in "
            f"{context or 'full image'}: {e}"
        )
        return None

    decoded_infos = []

    if isinstance(result, tuple):
        if len(result) == 4:
            _, infos, _, _ = result
            decoded_infos = _as_string_list(infos)
        elif len(result) == 3:
            _, infos, _ = result
            decoded_infos = _as_string_list(infos)
        elif len(result) == 2:
            infos, _ = result
            decoded_infos = _as_string_list(infos)
        elif len(result) == 1:
            decoded_infos = _as_string_list(result[0])
        else:
            decoded_infos = []
    else:
        decoded_infos = _as_string_list(result)

    for raw in decoded_infos:
        code = _normalize_and_validate_numeric(raw or "")
        if code:
            print(
                f"[BARCODE] OpenCV backend decoded"
                f"{(' from ' + context) if context else ''}: {code}"
            )
            return code

    return None


def readBarcode_hf(image_path: Union[str, Path]) -> Optional[str]:
    image_path = str(image_path)

    try:
        res = read_barcode(image_path)
    except Exception as e:
        print(f"[BARCODE] pyrxing error on full image: {e}")
        res = None

    if res is not None and getattr(res, "text", None):
        txt = res.text.strip()
        if txt:
            print(f"[BARCODE] accepted '{txt}' from plain pyrxing")
            return txt

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[BARCODE] cannot read image: {image_path}")
        return None

    enhanced_code = _try_pyrxing_with_rotations_and_preproc(img)
    if enhanced_code:
        return enhanced_code

    print("[BARCODE] enhanced pipeline failed; trying yolo detector...")

    orientations = [
        ("0deg", img),
        ("90deg", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
        ("180deg", cv2.rotate(img, cv2.ROTATE_180)),
        ("270deg", cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    for orient_label, orient_img in orientations:
        try:
            results = YOLO_MODEL(orient_img)[0]
        except Exception as e:
            print(f"[BARCODE] YOLO error on {orient_label}: {e}")
            continue

        boxes = getattr(results, "boxes", None)
        if boxes is None or len(boxes) == 0:
            if orient_label == "0deg":
                print("[BARCODE] Yolo found no barcode/QR regions")
            continue

        h, w = orient_img.shape[:2]
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        idx_order = np.argsort(-confs)
        for idx in idx_order:
            x1, y1, x2, y2 = xyxy[idx].astype(int)
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            roi = orient_img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            scale = 2.0
            roi_big = cv2.resize(
                roi,
                (int(roi.shape[1] * scale), int(roi.shape[0] * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

            print(
                f"[BARCODE] Trying backends on YOLO crop idx={idx}, "
                f"orient={orient_label}, conf={confs[idx]:.3f}, "
                f"size={roi_big.shape[1]}x{roi_big.shape[0]}"
            )

            rectified = _refine_and_rectify_barcode_region(roi_big)
            if rectified is not None:
                rect_code = decode_with_pyrxing_from_array(rectified)
                if rect_code:
                    print(
                        "[BARCODE] decoded from rectified YOLO crop "
                        f"idx={idx}, orient={orient_label}: {rect_code}"
                    )
                    return rect_code

            code = decode_with_pyrxing_from_array(roi_big)
            if code:
                print(
                    f"[BARCODE] decoded from YOLO crop idx={idx}, "
                    f"orient={orient_label}: {code}"
                )
                return code

            cv_code = _decode_with_opencv_barcode(
                roi_big, context=f"YOLO crop idx={idx}, {orient_label}"
            )
            if cv_code:
                return cv_code

    for orient_label, orient_img in orientations:
        cv_code = _decode_with_opencv_barcode(
            orient_img, context=f"full image {orient_label}"
        )
        if cv_code:
            return cv_code

    print("[BARCODE] No barcode decoded after YOLO + backends")
    return None
