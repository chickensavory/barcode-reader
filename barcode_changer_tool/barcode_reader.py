import os, tempfile, cv2, numpy as np, zxingcpp
from enum import Enum
from pathlib import Path
from typing import Optional, Union
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from pyrxing import read_barcode


weights = hf_hub_download(
    repo_id="Piero2411/YOLOV8s-Barcode-Detection",
    filename="YOLOV8s_Barcode_Detection.pt",
)
YOLO_MODEL = YOLO(weights)


class BarcodeStatus(str, Enum):
    BARCODE = "BARCODE"
    NONBARCODE = "NONBARCODE"
    UNSURE = "UNSURE"


def extract_upc_candidate(text: str) -> Optional[str]:
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) in (8, 12, 13, 14):
        return digits
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


def is_valid_upc_ean(code: str) -> bool:
    if not code or not code.isdigit():
        return False
    if len(code) == 12:
        return _upc_a_checksum_ok(code)
    if len(code) == 13:
        return _ean13_checksum_ok(code)
    if len(code) == 8:
        return _ean8_checksum_ok(code)
    return len(code) == 14


def enhance_for_pyrxing(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(gray, 1.7, blur, -0.7, 0)
    _, bw = cv2.threshold(sharp, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    return bw


def decode_with_pyrxing_from_array(img_bgr: np.ndarray) -> Optional[str]:
    if img_bgr is None or img_bgr.size == 0:
        return None
    enhanced = enhance_for_pyrxing(img_bgr)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        cv2.imwrite(tmp_path, enhanced)
        result = read_barcode(tmp_path)
        if result and result.text:
            raw = result.text.strip()
            candidate = extract_upc_candidate(raw)
            if candidate:
                if is_valid_upc_ean(candidate):
                    print(f"[BARCODE] pyrxing decoded candidate: {raw} -> {candidate}")
                    return candidate
                else:
                    print(
                        f"[BARCODE] pyrxing candidate failed checksum: {raw} -> {candidate}"
                    )
        return None
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def decode_with_zxing(img_bgr: np.ndarray) -> Optional[str]:
    if img_bgr is None or img_bgr.size == 0:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    results = zxingcpp.read_barcodes(gray)
    if not results:
        return None

    for res in results:
        raw = (res.text or "").strip()
        if not raw:
            continue
        candidate = extract_upc_candidate(raw)
        if candidate:
            if is_valid_upc_ean(candidate):
                print(f"[BARCODE] ZXing decoded candidate: {raw} -> {candidate}")
                return candidate
            else:
                print(
                    f"[BARCODE] ZXing candidate failed checksum: {raw} -> {candidate}"
                )
    return None


def brute_force_full_image(img_bgr: np.ndarray) -> Optional[str]:
    if img_bgr is None:
        return None

    h, w = img_bgr.shape[:2]
    center = (w // 2, h // 2)

    zx_votes: dict[str, int] = {}
    pyr_votes: dict[str, int] = {}

    for angle in [0, 90, -90, 45, -45]:
        if angle == 0:
            rotated = img_bgr
        else:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img_bgr, M, (w, h))

        code_pyr = decode_with_pyrxing_from_array(rotated)
        code_zx = decode_with_zxing(rotated)

        if code_pyr and code_zx and code_pyr == code_zx:
            print(f"[BARCODE] STRICT full-image match at {angle}°: {code_pyr}")
            return code_pyr

        if code_zx:
            zx_votes[code_zx] = zx_votes.get(code_zx, 0) + 1
        if code_pyr:
            pyr_votes[code_pyr] = pyr_votes.get(code_pyr, 0) + 1

        if code_pyr or code_zx:
            print(
                f"[BARCODE] Full-image partial at {angle}°: pyr={code_pyr}, zxing={code_zx}"
            )

    if not zx_votes and not pyr_votes:
        return None

    if zx_votes:
        candidate, count = max(zx_votes.items(), key=lambda kv: kv[1])
        pyr_conflicts = [c for c in pyr_votes.keys() if c != candidate]
        if (count >= 2 or not pyr_conflicts) and is_valid_upc_ean(candidate):
            print(f"[BARCODE] Accepting ZXing winner: {candidate} ({count} vote(s))")
            return candidate

    print(f"[BARCODE] Full-image votes inconclusive: zx={zx_votes}, pyr={pyr_votes}")
    return None


def readBarcode_hf_status(
    image_path: Union[str, Path],
) -> tuple[BarcodeStatus, Optional[str]]:
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[BARCODE] Cannot read {image_path}")
        return BarcodeStatus.NONBARCODE, None

    code = brute_force_full_image(img)
    if code:
        return BarcodeStatus.BARCODE, code

    print("[BARCODE] Full-image strict read failed, trying YOLO crops...")

    try:
        results = YOLO_MODEL.predict(image_path, conf=0.05, verbose=False)[0]
    except Exception as e:
        print("[BARCODE] YOLO error:", e)
        return BarcodeStatus.UNSURE, None

    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        print("[BARCODE] YOLO found no regions.")
        return BarcodeStatus.NONBARCODE, None

    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    order = confs.argsort()[::-1]

    h, w = img.shape[:2]
    for idx in order:
        x1, y1, x2, y2 = xyxy[idx]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        pad_x = max(30, int(0.6 * bw))
        pad_y = max(10, int(0.2 * bh))

        x1p = max(0, x1 - pad_x)
        y1p = max(0, y1 - pad_y)
        x2p = min(w, x2 + pad_x)
        y2p = min(h, y2 + pad_y)

        crop = img[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            continue

        crop_big = cv2.resize(crop, None, fx=3.5, fy=3.5, interpolation=cv2.INTER_CUBIC)
        print(
            f"[BARCODE] Trying YOLO crop {idx} conf={confs[idx]:.3f}, size={crop_big.shape[1]}x{crop_big.shape[0]}"
        )

        ch, cw = crop_big.shape[:2]
        center = (cw // 2, ch // 2)

        for angle in [0, 90, -90, 45, -45]:
            if angle == 0:
                rotated = crop_big
            else:
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(crop_big, M, (cw, ch))

            code_pyr = decode_with_pyrxing_from_array(rotated)
            code_zx = decode_with_zxing(rotated)

            if code_pyr and code_zx and code_pyr == code_zx:
                print(
                    f"[BARCODE] STRICT YOLO match (box {idx}, angle {angle}°): {code_pyr}"
                )
                return BarcodeStatus.BARCODE, code_pyr
            if code_zx and is_valid_upc_ean(code_zx):
                print(
                    f"[BARCODE] YOLO accept ZXing (box {idx}, angle {angle}°): {code_zx}"
                )
                return BarcodeStatus.BARCODE, code_zx
            if code_pyr and is_valid_upc_ean(code_pyr):
                print(
                    f"[BARCODE] YOLO accept pyrxing (box {idx}, angle {angle}°): {code_pyr}"
                )
                return BarcodeStatus.BARCODE, code_pyr

            if code_pyr or code_zx:
                print(
                    f"[BARCODE] YOLO box {idx} angle {angle}° mismatch: pyr={code_pyr}, zxing={code_zx}"
                )

    print("[BARCODE] Barcode-like regions found but no valid decode.")
    return BarcodeStatus.UNSURE, None


def readBarcode_hf(image_path: Union[str, Path]) -> Optional[str]:
    status, code = readBarcode_hf_status(image_path)
    return code if status == BarcodeStatus.BARCODE else None
