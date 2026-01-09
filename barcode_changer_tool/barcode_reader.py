import os
import re
import tempfile
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Dict, Set, Tuple, Iterable

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from pyrxing import read_barcode
import zxingcpp


weights = hf_hub_download(
    repo_id="Piero2411/YOLOV8s-Barcode-Detection",
    filename="YOLOV8s_Barcode_Detection.pt",
)
YOLO_MODEL = YOLO(weights)


class BarcodeStatus(str, Enum):
    BARCODE = "BARCODE"
    NONBARCODE = "NONBARCODE"
    UNSURE = "UNSURE"


YOLO_PRESENCE_CONF = 0.15

MIN_MARGIN_OVER_RUNNER_UP = 0
MIN_EVIDENCE_TO_ACCEPT = 2
MIN_VOTES_STRICT = 3
MIN_VOTES_LENIENT = 2

MAX_YOLO_BOXES = 6
ANGLES = [0, 90, -90, 45, -45]

MIN_CROP_AREA_PX = 2_000
MIN_ASPECT_RATIO_1D = 1.25

ALLOWED_ZX_FORMATS = {
    zxingcpp.BarcodeFormat.EAN13,
    zxingcpp.BarcodeFormat.EAN8,
    zxingcpp.BarcodeFormat.UPCA,
    zxingcpp.BarcodeFormat.ITF,
}

DUAL_DECODER_MATCH_BONUS_VOTES = 3

DEFAULT_MIN_BOX_AREA_RATIO = 0.012
DEFAULT_MIN_BOX_MAX_DIM_RATIO = 0.18


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


def enhance_for_pyrxing(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(gray, 1.7, blur, -0.7, 0)
    _, bw = cv2.threshold(sharp, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    return bw


def enhance_for_zxing_mild(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.8)
    gray = cv2.convertScaleAbs(gray, alpha=1.15, beta=0)
    return gray


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
            if candidate and is_valid_upc_ean(candidate):
                print(f"[BARCODE] pyrxing decoded: {raw} -> {candidate}")
                return candidate
        return None
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def _zxing_results_from_gray(gray: np.ndarray) -> Iterable[zxingcpp.Barcode]:
    try:
        return zxingcpp.read_barcodes(gray) or []
    except Exception:
        return []


def _zxing_geometry_plausible(res: zxingcpp.Barcode) -> bool:
    try:
        pos = getattr(res, "position", None)
        if not pos:
            return True

        xs = [p.x for p in pos]
        ys = [p.y for p in pos]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        if w <= 0 or h <= 0:
            return False

        area = w * h
        ar = (w / h) if w >= h else (h / w)

        if area < 400:
            return False
        if ar < 1.05:
            return False
        return True
    except Exception:
        return True


def decode_with_zxing(
    img_bgr: np.ndarray,
    allow_secondary_pass_only_if_matches: Optional[Set[str]] = None,
) -> Optional[str]:
    if img_bgr is None or img_bgr.size == 0:
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    for res in _zxing_results_from_gray(gray):
        if not getattr(res, "valid", True):
            continue
        fmt = getattr(res, "format", None)
        if fmt is not None and fmt not in ALLOWED_ZX_FORMATS:
            continue
        if not _zxing_geometry_plausible(res):
            continue

        raw = (res.text or "").strip()
        if not raw:
            continue
        candidate = extract_upc_candidate(raw)
        if candidate and is_valid_upc_ean(candidate):
            print(f"[BARCODE] ZXing decoded: {raw} -> {candidate}")
            return candidate

    if allow_secondary_pass_only_if_matches:
        gray2 = enhance_for_zxing_mild(img_bgr)
        for res in _zxing_results_from_gray(gray2):
            if not getattr(res, "valid", True):
                continue
            fmt = getattr(res, "format", None)
            if fmt is not None and fmt not in ALLOWED_ZX_FORMATS:
                continue
            if not _zxing_geometry_plausible(res):
                continue

            raw = (res.text or "").strip()
            if not raw:
                continue
            candidate = extract_upc_candidate(raw)
            if (
                candidate
                and is_valid_upc_ean(candidate)
                and candidate in allow_secondary_pass_only_if_matches
            ):
                print(f"[BARCODE] ZXing (mild) agrees: {raw} -> {candidate}")
                return candidate

    return None


Evidence = Tuple[str, int, str]


def _rotate(img_bgr: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h))


def _add_vote(
    votes: Dict[str, int],
    evidence: Dict[str, Set[Evidence]],
    code: str,
    decoder: str,
    angle: int,
    region_label: str,
    n: int = 1,
):
    votes[code] = votes.get(code, 0) + n
    evidence.setdefault(code, set()).add((decoder, angle, region_label))


def _pick_winner(
    votes: Dict[str, int],
    evidence: Dict[str, Set[Evidence]],
    min_votes: int,
    min_margin: int,
    min_evidence: int,
) -> Tuple[Optional[str], str]:
    if not votes:
        return None, "no_votes"

    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    winner, wv = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0

    if wv < min_votes:
        return None, f"winner_has_{wv}_votes_lt_{min_votes}"

    if (wv - runner_up) < min_margin:
        return (
            None,
            f"too_close_winner_{wv}_runnerup_{runner_up}_lt_margin_{min_margin}",
        )

    evc = len(evidence.get(winner, set()))
    if evc < min_evidence:
        return None, f"winner_has_{evc}_evidence_lt_{min_evidence}"

    return winner, "accepted"


def _vote_from_image(
    img_bgr: np.ndarray,
    region_label: str,
) -> Tuple[Dict[str, int], Dict[str, Set[Evidence]]]:
    votes: Dict[str, int] = {}
    evidence: Dict[str, Set[Evidence]] = {}

    if img_bgr is None or img_bgr.size == 0:
        return votes, evidence

    for angle in ANGLES:
        rotated = _rotate(img_bgr, angle)

        code_pyr = decode_with_pyrxing_from_array(rotated)
        already = set([code_pyr]) if code_pyr else set()
        code_zx = decode_with_zxing(
            rotated, allow_secondary_pass_only_if_matches=already
        )

        if code_pyr and code_zx and code_pyr == code_zx:
            print(
                f"[BARCODE] {region_label}: dual-decoder match at {angle}° => {code_zx} (+{DUAL_DECODER_MATCH_BONUS_VOTES} votes)"
            )
            _add_vote(
                votes,
                evidence,
                code_zx,
                "zx",
                angle,
                region_label,
                n=DUAL_DECODER_MATCH_BONUS_VOTES,
            )
            _add_vote(
                votes,
                evidence,
                code_pyr,
                "pyr",
                angle,
                region_label,
                n=DUAL_DECODER_MATCH_BONUS_VOTES,
            )
            continue

        if code_zx:
            print(f"[BARCODE] {region_label}: zx at {angle}° => {code_zx}")
            _add_vote(votes, evidence, code_zx, "zx", angle, region_label, n=1)

        if code_pyr:
            print(f"[BARCODE] {region_label}: pyr at {angle}° => {code_pyr}")
            _add_vote(votes, evidence, code_pyr, "pyr", angle, region_label, n=1)

    return votes, evidence


def yolo_detect_boxes(
    image_path: Union[str, Path],
    conf: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    res = YOLO_MODEL.predict(str(image_path), conf=conf, verbose=False)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=int), np.zeros((0,), dtype=float)

    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    order = confs.argsort()[::-1]
    return xyxy[order], confs[order]


def _crop_plausible(x1: int, y1: int, x2: int, y2: int) -> bool:
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    if bw <= 0 or bh <= 0:
        return False
    area = bw * bh
    if area < MIN_CROP_AREA_PX:
        return False
    ar = bw / bh if bh else 999.0
    ar = ar if ar >= 1.0 else (1.0 / ar)
    if ar < MIN_ASPECT_RATIO_1D:
        return False
    return True


def yolo_has_large_enough_barcode_region(
    image_path: Union[str, Path],
    min_area_ratio: float = DEFAULT_MIN_BOX_AREA_RATIO,
    min_max_dim_ratio: float = DEFAULT_MIN_BOX_MAX_DIM_RATIO,
) -> bool:
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    img_area = float(h * w)
    max_dim = float(max(h, w))

    xyxy, confs = yolo_detect_boxes(image_path, conf=0.05)
    if xyxy.shape[0] == 0:
        return False

    for (x1, y1, x2, y2), c in zip(xyxy, confs):
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        if bw <= 0 or bh <= 0:
            continue
        area_ratio = (bw * bh) / img_area
        max_dim_ratio = max(bw, bh) / max_dim
        if area_ratio >= min_area_ratio and max_dim_ratio >= min_max_dim_ratio:
            return True

    return False


def readBarcode_hf_status(
    image_path: Union[str, Path],
    *,
    require_large_region: bool = False,
    min_box_area_ratio: float = DEFAULT_MIN_BOX_AREA_RATIO,
    min_box_max_dim_ratio: float = DEFAULT_MIN_BOX_MAX_DIM_RATIO,
) -> tuple[BarcodeStatus, Optional[str]]:
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[BARCODE] Cannot read {image_path}")
        return BarcodeStatus.NONBARCODE, None

    if require_large_region:
        ok = yolo_has_large_enough_barcode_region(
            image_path,
            min_area_ratio=min_box_area_ratio,
            min_max_dim_ratio=min_box_max_dim_ratio,
        )
        if not ok:
            print(
                "[BARCODE] Rejecting decode attempt: barcode region too small (hero-shot defense)."
            )
            return BarcodeStatus.NONBARCODE, None

    xyxy, _ = yolo_detect_boxes(image_path, conf=YOLO_PRESENCE_CONF)
    yolo_present = xyxy.shape[0] > 0
    min_votes = MIN_VOTES_LENIENT if yolo_present else MIN_VOTES_STRICT
    min_margin = MIN_MARGIN_OVER_RUNNER_UP
    min_evidence = MIN_EVIDENCE_TO_ACCEPT

    votes_all: Dict[str, int] = {}
    evidence_all: Dict[str, Set[Evidence]] = {}

    votes, evidence = _vote_from_image(img, region_label="full")
    for c, n in votes.items():
        votes_all[c] = votes_all.get(c, 0) + n
        evidence_all.setdefault(c, set()).update(evidence.get(c, set()))

    winner, reason = _pick_winner(
        votes_all, evidence_all, min_votes, min_margin, min_evidence
    )
    if winner:
        print(
            f"[BARCODE] ACCEPT full-image winner: {winner} "
            f"({votes_all[winner]} votes, {len(evidence_all[winner])} evidence) reason={reason}"
        )
        return BarcodeStatus.BARCODE, winner

    print("[BARCODE] Full-image not accepted; trying YOLO crops...")

    try:
        xyxy, confs = yolo_detect_boxes(image_path, conf=0.05)
    except Exception as e:
        print("[BARCODE] YOLO error:", e)
        return (BarcodeStatus.UNSURE if votes_all else BarcodeStatus.NONBARCODE), None

    if xyxy.shape[0] == 0:
        print("[BARCODE] YOLO found no regions.")
        return (BarcodeStatus.UNSURE if votes_all else BarcodeStatus.NONBARCODE), None

    h, w = img.shape[:2]
    tried = 0

    for x1, y1, x2, y2 in xyxy:
        if tried >= MAX_YOLO_BOXES:
            break

        if not _crop_plausible(int(x1), int(y1), int(x2), int(y2)):
            continue

        bw = max(1, int(x2) - int(x1))
        bh = max(1, int(y2) - int(y1))

        pad_x = max(30, int(0.6 * bw))
        pad_y = max(10, int(0.2 * bh))

        x1p = max(0, int(x1) - pad_x)
        y1p = max(0, int(y1) - pad_y)
        x2p = min(w, int(x2) + pad_x)
        y2p = min(h, int(y2) + pad_y)

        crop = img[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            continue

        crop_big = cv2.resize(crop, None, fx=3.5, fy=3.5, interpolation=cv2.INTER_CUBIC)

        tried += 1
        votes_c, evidence_c = _vote_from_image(crop_big, region_label=f"crop{tried}")
        for c, n in votes_c.items():
            votes_all[c] = votes_all.get(c, 0) + n
            evidence_all.setdefault(c, set()).update(evidence_c.get(c, set()))

        winner, reason = _pick_winner(
            votes_all, evidence_all, min_votes, min_margin, min_evidence
        )
        if winner:
            print(
                f"[BARCODE] ACCEPT after YOLO crops: {winner} "
                f"({votes_all[winner]} votes, {len(evidence_all[winner])} evidence) reason={reason}"
            )
            return BarcodeStatus.BARCODE, winner

    if votes_all:
        print(f"[BARCODE] NOT ACCEPTED. votes={votes_all}")
        return BarcodeStatus.UNSURE, None

    return BarcodeStatus.NONBARCODE, None


def readBarcode_hf(
    image_path: Union[str, Path],
    *,
    require_large_region: bool = False,
    min_box_area_ratio: float = DEFAULT_MIN_BOX_AREA_RATIO,
    min_box_max_dim_ratio: float = DEFAULT_MIN_BOX_MAX_DIM_RATIO,
) -> Optional[str]:
    status, code = readBarcode_hf_status(
        image_path,
        require_large_region=require_large_region,
        min_box_area_ratio=min_box_area_ratio,
        min_box_max_dim_ratio=min_box_max_dim_ratio,
    )
    return code if status == BarcodeStatus.BARCODE else None
