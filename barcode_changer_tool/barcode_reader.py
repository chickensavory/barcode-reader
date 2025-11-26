import os
import tempfile
from collections import defaultdict
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

STRICT_BARCODE_VALIDATION = True
EXPECT_NUMERIC_ONLY = True
ALLOWED_BARCODE_LENGTHS = {8, 12, 13}
MIN_VOTES_FOR_AGGRESSIVE_DECODE = 2

try:
    BARCODE_DETECTOR = cv2.barcode_BarcodeDetector()
    print("[BARCODE] OpenCV BarcodeDetector available")
except AttributeError:
    BARCODE_DETECTOR = None
    print("[BARCODE] OpenCV BarcodeDetector NOT available in this OpenCV build")


def decode_with_pyrxing_from_array(img_bgr) -> Optional[str]:
    if img_bgr is None or img_bgr.size == 0:
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        cv2.imwrite(tmp_path, img_bgr)
        res = read_barcode(tmp_path)
        if res is not None and res.text:
            return res.text.strip()
        return None
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _strict_is_plausible_barcode(text: str) -> bool:
    if not text:
        return False
    text = text.strip()

    if EXPECT_NUMERIC_ONLY and not text.isdigit():
        return False

    if ALLOWED_BARCODE_LENGTHS and len(text) not in ALLOWED_BARCODE_LENGTHS:
        return False

    return True


def _ean13_checksum_is_valid(text: str) -> bool:
    if len(text) != 13 or not text.isdigit():
        return False

    digits = [int(c) for c in text]
    check = digits[-1]
    data = digits[:-1]

    s_odd = sum(data[0::2])
    s_even = sum(data[1::2])
    total = s_odd + 3 * s_even
    calc = (10 - (total % 10)) % 10

    return calc == check


def accept_or_reject(
    text: Optional[str], where: str, strict_checksum: bool = False
) -> Optional[str]:
    if not text:
        return None

    text = text.strip()
    if not text:
        return None

    if not STRICT_BARCODE_VALIDATION:
        print(f"[BARCODE] accepted (loose) '{text}' from {where}")
        return text

    if not _strict_is_plausible_barcode(text):
        print(f"[BARCODE] rejected implausible '{text}' from {where}")
        return None

    if strict_checksum and len(text) == 13:
        if not _ean13_checksum_is_valid(text):
            print(f"[BARCODE] rejected invalid EAN-13 checksum '{text}' from {where}")
            return None

    print(f"[BARCODE] accepted (strict) '{text}' from {where}")
    return text


def decode_with_opencv_barcode(img_bgr) -> Optional[str]:
    if BARCODE_DETECTOR is None:
        return None
    if img_bgr is None or img_bgr.size == 0:
        return None

    try:
        res = BARCODE_DETECTOR.detectAndDecode(img_bgr)
    except Exception as e:
        print(f"[BARCODE] OpenCV barcode detector error: {e}")
        return None

    if not isinstance(res, tuple):
        return None

    if len(res) == 4:
        ok, decoded_info, decoded_type, points = res
    elif len(res) == 3:
        decoded_info, decoded_type, points = res
        ok = bool(decoded_info)
    else:
        print(
            f"[BARCODE] OpenCV barcode detector returned unexpected tuple length: {len(res)}"
        )
        return None

    if not ok or not decoded_info:
        return None

    if isinstance(decoded_info, (list, tuple)):
        for s in decoded_info:
            if s and s.strip():
                return s.strip()
    elif isinstance(decoded_info, str):
        if decoded_info.strip():
            return decoded_info.strip()

    return None


def decode_with_all_backends(
    img_bgr, where: str, strict_checksum: bool = False
) -> Optional[str]:
    txt = decode_with_opencv_barcode(img_bgr)
    code = accept_or_reject(txt, where + " (opencv)", strict_checksum=strict_checksum)
    if code:
        return code

    txt = decode_with_pyrxing_from_array(img_bgr)
    code = accept_or_reject(txt, where + " (pyrxing)", strict_checksum=strict_checksum)
    if code:
        return code

    return None


def _apply_contrast_enhancement(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    lab_enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0, sigmaY=1.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

    return sharpened


def _try_pyrxing_with_rotations_and_preproc(img_bgr) -> Optional[str]:
    if img_bgr is None or img_bgr.size == 0:
        return None

    candidate_votes = defaultdict(int)

    rotations = [
        ("0deg", img_bgr),
        ("90deg", cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)),
        ("180deg", cv2.rotate(img_bgr, cv2.ROTATE_180)),
        ("270deg", cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    for label, rot_img in rotations:
        code = decode_with_all_backends(
            rot_img,
            f"enhanced-full raw rot={label}",
            strict_checksum=True,
        )
        if code:
            candidate_votes[code] += 1

        enhanced = _apply_contrast_enhancement(rot_img)
        code = decode_with_all_backends(
            enhanced,
            f"enhanced-full clahe+sharp rot={label}",
            strict_checksum=True,
        )
        if code:
            candidate_votes[code] += 1

    if not candidate_votes:
        return None

    best_code, votes = max(candidate_votes.items(), key=lambda kv: kv[1])
    print(f"[BARCODE] enhanced-full best candidate '{best_code}' with {votes} vote(s)")

    if votes >= MIN_VOTES_FOR_AGGRESSIVE_DECODE:
        return best_code

    print("[BARCODE] enhanced-full candidate below vote threshold; discarded")
    return None


def readBarcode_hf(image_path: Union[str, Path]) -> Optional[str]:
    image_path = str(image_path)

    try:
        full_res = read_barcode(image_path)
        if full_res is not None and full_res.text:
            raw_text = full_res.text.strip()
            code = accept_or_reject(raw_text, "full-image", strict_checksum=True)
            if code:
                return code
    except Exception as e:
        print(f"[BARCODE] pyrxing full image error {e}")

    print(
        "[BARCODE] pyrxing failed or implausible on full image; "
        "trying enhanced full-image pipeline..."
    )

    img = cv2.imread(image_path)
    if img is not None:
        enhanced_code = _try_pyrxing_with_rotations_and_preproc(img)
        if enhanced_code:
            return enhanced_code
    else:
        print(f"[BARCODE] cv2 cannot read image for enhanced pipeline: {image_path}")

    print("[BARCODE] enhanced pipeline failed; trying yolo detector...")

    results = YOLO_MODEL(image_path)[0]
    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        print("[BARCODE] Yolo found no barcode/QR regions")
        return None

    if img is None:
        img = cv2.imread(image_path)
    if img is None:
        print(f"[BARCODE] cannot read image: {image_path}")
        return None

    h, w = img.shape[:2]

    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy().astype(int)

    indices = list(range(len(confs)))

    def _is_likely_1d(idx: int) -> bool:
        x1, y1, x2, y2 = xyxy[idx]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        aspect = max(bw, bh) / max(1, min(bw, bh))
        return aspect >= 2.5

    indices.sort(key=lambda i: (_is_likely_1d(i), confs[i]), reverse=True)

    for idx in indices:
        conf = float(confs[idx])
        if conf < 0.35:
            print(f"[BARCODE] skipping low-conf YOLO box idx={idx}, conf={conf:.3f}")
            continue

        x1, y1, x2, y2 = xyxy[idx]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        roi_big = cv2.resize(roi, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        aspect = max(bw, bh) / max(1, min(bw, bh))

        print(
            f"[BARCODE] Trying pyrxing/OpenCV on YOLO crop idx={idx}, "
            f"conf={conf:.3f}, aspect={aspect:.2f}, "
            f"size={roi_big.shape[1]}x{roi_big.shape[0]}"
        )

        code = decode_with_all_backends(
            roi_big, f"yolo-crop idx={idx}", strict_checksum=True
        )
        if code:
            print(f"[BARCODE] decoded from YOLO crop: {code}")
            return code

    print("[BARCODE] No barcode decoded after YOLO + backends")
    return None
