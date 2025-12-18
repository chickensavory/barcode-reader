import os, re, tempfile, cv2, numpy as np, zxingcpp
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Dict, Set, Tuple
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


MIN_VOTES_TO_ACCEPT = 4
REQUIRE_BOTH_DECODERS = True
MIN_MARGIN_OVER_RUNNER_UP = 2
MAX_YOLO_BOXES = 6
ANGLES = [0, 90, -90, 45, -45]
REQUIRE_YOLO_PRESENCE = True
YOLO_PRESENCE_CONF = 0.15


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
        if candidate and is_valid_upc_ean(candidate):
            print(f"[BARCODE] ZXing decoded: {raw} -> {candidate}")
            return candidate

    return None


def _rotate(img_bgr: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h))


def _add_vote(
    votes: Dict[str, int],
    sources: Dict[str, Set[str]],
    code: str,
    src: str,
    n: int = 1,
):
    votes[code] = votes.get(code, 0) + n
    if code not in sources:
        sources[code] = set()
    sources[code].add(src)


def _pick_winner(
    votes: Dict[str, int],
    sources: Dict[str, Set[str]],
) -> Tuple[Optional[str], str]:
    if not votes:
        return None, "no_votes"

    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    winner, wv = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0

    if wv < MIN_VOTES_TO_ACCEPT:
        return None, f"winner_has_{wv}_votes_lt_{MIN_VOTES_TO_ACCEPT}"

    if (wv - runner_up) < MIN_MARGIN_OVER_RUNNER_UP:
        return None, f"too_close_winner_{wv}_runnerup_{runner_up}"

    if REQUIRE_BOTH_DECODERS:
        srcs = sources.get(winner, set())
        if not ("zx" in srcs and "pyr" in srcs):
            return None, f"winner_missing_both_decoders_sources={sorted(list(srcs))}"

    return winner, "accepted"


def _vote_from_image(
    img_bgr: np.ndarray, label: str
) -> Tuple[Dict[str, int], Dict[str, Set[str]]]:
    votes: Dict[str, int] = {}
    sources: Dict[str, Set[str]] = {}

    if img_bgr is None or img_bgr.size == 0:
        return votes, sources

    for angle in ANGLES:
        rotated = _rotate(img_bgr, angle)

        code_pyr = decode_with_pyrxing_from_array(rotated)
        code_zx = decode_with_zxing(rotated)

        if code_pyr and code_zx and code_pyr == code_zx:
            print(f"[BARCODE] {label}: match at {angle}° => {code_zx} (2 votes)")
            _add_vote(votes, sources, code_zx, "zx", n=1)
            _add_vote(votes, sources, code_pyr, "pyr", n=1)
            continue

        if code_zx:
            print(f"[BARCODE] {label}: zx at {angle}° => {code_zx}")
            _add_vote(votes, sources, code_zx, "zx", n=1)

        if code_pyr:
            print(f"[BARCODE] {label}: pyr at {angle}° => {code_pyr}")
            _add_vote(votes, sources, code_pyr, "pyr", n=1)

    return votes, sources


def yolo_barcode_present(
    image_path: Union[str, Path], conf: float = YOLO_PRESENCE_CONF
) -> bool:
    try:
        res = YOLO_MODEL.predict(str(image_path), conf=conf, verbose=False)[0]
        boxes = res.boxes
        return boxes is not None and len(boxes) > 0
    except Exception:
        return False


def readBarcode_hf_status(
    image_path: Union[str, Path],
) -> tuple[BarcodeStatus, Optional[str]]:
    image_path = str(image_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[BARCODE] Cannot read {image_path}")
        return BarcodeStatus.NONBARCODE, None

    if REQUIRE_YOLO_PRESENCE:
        present = yolo_barcode_present(image_path, conf=YOLO_PRESENCE_CONF)
        if not present:
            return BarcodeStatus.NONBARCODE, None

    votes_all: Dict[str, int] = {}
    sources_all: Dict[str, Set[str]] = {}

    votes, sources = _vote_from_image(img, label="full")
    for c, n in votes.items():
        _add_vote(votes_all, sources_all, c, src="seed", n=0)
        votes_all[c] = votes_all.get(c, 0) + n
        sources_all[c] |= sources.get(c, set())

    winner, reason = _pick_winner(votes_all, sources_all)
    if winner:
        print(
            f"[BARCODE] ACCEPT full-image winner: {winner} ({votes_all[winner]} votes) reason={reason}"
        )
        return BarcodeStatus.BARCODE, winner

    if votes_all:
        print(
            f"[BARCODE] Full-image votes present but not accepted: reason={reason}, votes={votes_all}"
        )

    print("[BARCODE] Full-image strict read not accepted, trying YOLO crops...")

    try:
        results = YOLO_MODEL.predict(image_path, conf=0.05, verbose=False)[0]
    except Exception as e:
        print("[BARCODE] YOLO error:", e)
        return BarcodeStatus.UNSURE, None

    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        print("[BARCODE] YOLO found no regions.")
        return BarcodeStatus.UNSURE, None

    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    order = confs.argsort()[::-1]

    h, w = img.shape[:2]
    tried = 0

    for idx in order:
        if tried >= MAX_YOLO_BOXES:
            break

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

        tried += 1
        votes_c, sources_c = _vote_from_image(crop_big, label=f"crop{tried}")

        for c, n in votes_c.items():
            _add_vote(votes_all, sources_all, c, src="seed", n=0)
            votes_all[c] = votes_all.get(c, 0) + n
            sources_all[c] |= sources_c.get(c, set())

        winner, reason = _pick_winner(votes_all, sources_all)
        if winner:
            print(
                f"[BARCODE] ACCEPT after YOLO crops: {winner} ({votes_all[winner]} votes) reason={reason}"
            )
            return BarcodeStatus.BARCODE, winner

    if votes_all:
        print(
            f"[BARCODE] NOT ACCEPTED. votes={votes_all}, sources={ {k: sorted(list(v)) for k,v in sources_all.items()} }"
        )
        return BarcodeStatus.UNSURE, None

    return BarcodeStatus.NONBARCODE, None


def readBarcode_hf(image_path: Union[str, Path]) -> Optional[str]:
    status, code = readBarcode_hf_status(image_path)
    return code if status == BarcodeStatus.BARCODE else None
