import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import threading
YOLO_LOCK = threading.Lock()

import re, subprocess, time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from barcode_changer_tool.barcode_reader import readBarcode_hf_status, BarcodeStatus
from barcode_changer_tool.barcode_rename import read_xmp_rating_and_label, role_from_xmp

DEFAULT_RATING_TO_ROLE: Dict[int, str] = {
    1: "hero",
    2: "packaging",
    3: "nutritional",
    4: "upc",
}

for var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(var, "1")

try:
    from PIL import Image
except ImportError:
    Image = None

INPUT_DIR = Path("input")
GOOD_DIR = Path("good_LS2B")
BAD_DIR = Path("bad_LS2B")

SUPPORTED_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".dng",
    ".bmp",
    ".nef",
    ".arw",
    ".cr2",
    ".cr3",
}

RAW_EXTS = {".nef", ".arw", ".cr2", ".cr3"}

RESET_GAP_SEC = 120.0
ALLOWED_RATINGS = {1, 2, 3, 4}

BARCODE_SCAN_PRIORITY = [4, 1]

MAX_WORKERS = min(8, (os.cpu_count() or 4))


@dataclass
class Photo:
    path: Path
    timestamp: datetime
    index: int
    rating: Optional[int]
    label: Optional[str]


@dataclass
class ProductSet:
    photos_by_rating: Dict[int, Photo]
    starter_rating: int
    barcode: Optional[str] = None
    barcode_source_rating: Optional[int] = None


def extractIndex(path: Path) -> int:
    m = re.search(r"\((\d+)\)", path.stem)
    return int(m.group(1)) if m else 999999


def sanitizeBarcodeForFileName(barcode_text: str) -> str:
    sanitized = re.sub(r"[^\dA-Za-z]+", "_", (barcode_text or "")).strip("_")
    return sanitized or "barcode"


def getPhotoTimestamp(path: Path) -> datetime:
    ts = datetime.fromtimestamp(path.stat().st_mtime)
    if Image is None:
        return ts

    try:
        with Image.open(path) as im:
            exif = getattr(im, "_getexif", lambda: None)()
            if not exif:
                return ts
            for tag in (36867, 406):
                value = exif.get(tag)
                if isinstance(value, str):
                    try:
                        return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                    except Exception:
                        continue
    except Exception:
        return ts
    return ts


def listInputFiles() -> List[Path]:
    if not INPUT_DIR.exists():
        print("[ERROR] input folder missing")
        return []
    return [
        p
        for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]


def convertRawToTempPNG(raw_path: Path) -> Optional[Path]:
    if raw_path.suffix.lower() not in RAW_EXTS:
        return None

    tmp_png = raw_path.with_suffix(".barcode_tmp.png")
    print(f"[RAW] Converting {raw_path.name} -> {tmp_png.name} for barcode scan")
    try:
        result = subprocess.run(
            ["sips", "-s", "format", "png", str(raw_path), "--out", str(tmp_png)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[RAW] sips failed on {raw_path.name}: {result.stderr.strip()}")
            return None
    except FileNotFoundError:
        print(
            "[RAW] ERROR: 'sips' not found (macOS-only). RAW barcode scan will be skipped."
        )
        return None

    if not tmp_png.exists():
        print(f"[RAW] ERROR: temp PNG missing after conversion of {raw_path.name}")
        return None

    return tmp_png


def _prepare_scan_path(path: Path) -> Tuple[Path, Optional[Path]]:
    if path.suffix.lower() not in RAW_EXTS:
        return path, None

    tmp_png = convertRawToTempPNG(path)
    if tmp_png is None:
        return path, None
    return tmp_png, tmp_png


def _read_xmp_rating(path: Path) -> Tuple[Optional[int], Optional[str]]:
    try:
        meta = read_xmp_rating_and_label(path)
        return meta.rating, meta.label
    except Exception:
        return None, None


def loadPhotos() -> List[Photo]:
    files = listInputFiles()
    if not files:
        print("[INFO] No image found.")
        return []

    photos: List[Photo] = []
    for p in files:
        ts = getPhotoTimestamp(p)
        idx = extractIndex(p)
        rating, label = _read_xmp_rating(p)
        photos.append(
            Photo(path=p, timestamp=ts, index=idx, rating=rating, label=label)
        )

    photos.sort(key=lambda ph: (ph.timestamp, ph.index))
    return photos


def chunk_by_reset_gap(photos: List[Photo]) -> List[List[Photo]]:
    if not photos:
        return []
    chunks: List[List[Photo]] = []
    cur = [photos[0]]
    for prev, nxt in zip(photos, photos[1:]):
        gap = (nxt.timestamp - prev.timestamp).total_seconds()
        if gap > RESET_GAP_SEC:
            chunks.append(cur)
            cur = [nxt]
        else:
            cur.append(nxt)
    chunks.append(cur)
    return chunks


def moveToBad(path: Path, reason: str = ""):
    BAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = BAD_DIR / path.name
    i = 1
    while dest.exists():
        dest = BAD_DIR / f"{path.stem}_{i}{path.suffix}"
        i += 1
    msg = f"[MOVE] {path.name} -> bad/{dest.name}"
    if reason:
        msg += f" ({reason})"
    print(msg)
    path.rename(dest)


def move_set_to_bad(ps: ProductSet, reason: str):
    for ph in ps.photos_by_rating.values():
        moveToBad(ph.path, reason=reason)


def _safe_rename(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    candidate = dest
    n = 1
    while candidate.exists():
        candidate = dest.parent / f"{dest.stem}_{n}{dest.suffix}"
        n += 1
    print(f"[RENAME] {src.name} -> {candidate.name}")
    src.rename(candidate)
    return candidate


def build_product_sets(block: List[Photo]) -> List[ProductSet]:
    sets: List[ProductSet] = []
    cur: Dict[int, Photo] = {}
    starter_rating: Optional[int] = None
    in_set = False

    def flush_current():
        nonlocal cur, starter_rating, in_set
        if not cur:
            starter_rating = None
            in_set = False
            return
        sets.append(
            ProductSet(photos_by_rating=cur, starter_rating=starter_rating or 4)
        )
        cur = {}
        starter_rating = None
        in_set = False

    for ph in block:
        r = ph.rating

        if r not in ALLOWED_RATINGS:
            moveToBad(ph.path, reason=f"unexpected_or_missing_rating={r}")
            continue

        ri = int(r)

        if not in_set:
            if ri in (4, 1):
                in_set = True
                starter_rating = ri
                cur[ri] = ph
                continue

            moveToBad(ph.path, reason=f"rating_{ri}_seen_before_any_starter")
            continue

        if ri == 4 and starter_rating == 1 and 4 not in cur:
            if set(cur.keys()) == {1}:
                cur[4] = ph
                starter_rating = 4
                continue
            else:
                flush_current()
                in_set = True
                starter_rating = 4
                cur[4] = ph
                continue

        if ri in (1, 4) and ri in cur:
            flush_current()
            in_set = True
            starter_rating = ri
            cur[ri] = ph
            continue

        if ri in cur:
            moveToBad(ph.path, reason=f"duplicate_rating_in_set={ri}")
            continue

        cur[ri] = ph

    flush_current()
    return sets


def _try_decode_barcode(path: Path, *, require_large_region: bool) -> Optional[str]:
    scan_path, tmp_png = _prepare_scan_path(path)
    try:
        with YOLO_LOCK:
            status, code = readBarcode_hf_status(
                str(scan_path),
                require_large_region=require_large_region,
            )
        return code if status == BarcodeStatus.BARCODE and code else None
    finally:
        if tmp_png is not None and tmp_png.exists():
            try:
                tmp_png.unlink()
            except Exception:
                pass


def decode_barcode_for_set(ps: ProductSet) -> None:
    ph4 = ps.photos_by_rating.get(4)
    if ph4 is not None:
        code = _try_decode_barcode(ph4.path, require_large_region=True)
        if code:
            ps.barcode = code
            ps.barcode_source_rating = 4
            print(f"[BARCODE] decoded from rating 4 => {code}")
            return
        print("[BARCODE] rating 4 present but decode failed; trying rating 1...")

    ph1 = ps.photos_by_rating.get(1)
    if ph1 is None:
        ps.barcode = None
        ps.barcode_source_rating = None
        print("[BARCODE] no rating 1 available => barcode fail")
        return

    code = _try_decode_barcode(ph1.path, require_large_region=True)
    if code:
        ps.barcode = code
        ps.barcode_source_rating = 1
        print(f"[BARCODE] decoded from rating 1 => {code}")
        return

    ps.barcode = None
    ps.barcode_source_rating = None
    print("[BARCODE] rating 1 decode failed => barcode fail")


def decode_barcodes_for_sets(product_sets: List[ProductSet]) -> None:
    if not product_sets:
        return

    def _worker(i: int) -> int:
        try:
            decode_barcode_for_set(product_sets[i])
        except Exception as e:
            ps = product_sets[i]
            ps.barcode = None
            ps.barcode_source_rating = None
            print(f"[BARCODE] ERROR decoding set {i}: {type(e).__name__}: {e}")
        return i

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_worker, i) for i in range(len(product_sets))]
        for fut in as_completed(futures):
            _ = fut.result()


def role_from_rating_fallback(rating: Optional[int]) -> Optional[str]:
    if rating is None:
        return None
    return DEFAULT_RATING_TO_ROLE.get(int(rating))


def rename_product_set(ps: ProductSet):
    if not ps.barcode:
        has4 = 4 in ps.photos_by_rating
        has1 = 1 in ps.photos_by_rating

        if has4 and has1:
            move_set_to_bad(ps, reason="barcode_failed_after_try_4_then_1")
        elif has4 and not has1:
            move_set_to_bad(ps, reason="barcode_failed_on_4_and_no_1_fallback")
        elif (not has4) and has1:
            move_set_to_bad(ps, reason="barcode_failed_on_1_only_no_4_available")
        else:
            move_set_to_bad(ps, reason="no_4_or_1_available_for_barcode_scan")
        return

    barcode = sanitizeBarcodeForFileName(ps.barcode)

    for rating, ph in sorted(ps.photos_by_rating.items()):
        role, r2, label, reason = role_from_xmp(ph.path)

        if not role:
            role = role_from_rating_fallback(rating)
            if role:
                print(
                    f"[ROLE] Fallback role from rating={rating} => {role} (xmp_reason={reason})"
                )
            else:
                moveToBad(
                    ph.path,
                    reason=f"no_role_from_xmp_and_no_rating_fallback ({reason}, rating={r2}, label={label})",
                )
                continue

        dest = GOOD_DIR / f"{barcode}_{role}{ph.path.suffix.lower()}"
        _safe_rename(ph.path, dest)


def main():
    t0 = time.time()

    photos = loadPhotos()
    if not photos:
        return

    blocks = chunk_by_reset_gap(photos)
    print(f"[INFO] Found {len(photos)} file(s) across {len(blocks)} time block(s).")
    print(
        "[INFO] Building product sets: each set starts with rating 4 (upc) or 1 (hero); 2/3 are optional."
    )

    all_sets: List[ProductSet] = []
    for bi, block in enumerate(blocks, start=1):
        sets = build_product_sets(block)
        print(f"[INFO] Block {bi}: {len(sets)} product set(s).")
        all_sets.extend(sets)

    if not all_sets:
        print("[INFO] No product sets found.")
        return

    valid_sets: List[ProductSet] = []
    for ps in all_sets:
        if 1 not in ps.photos_by_rating:
            move_set_to_bad(ps, reason="missing_required_hero_shot_rating_1")
        else:
            valid_sets.append(ps)

    all_sets = valid_sets

    if not all_sets:
        print(
            "[INFO] All detected sets were missing hero shots; nothing left to process."
        )
        return

    print(
        "[INFO] Decoding barcodes: try rating 4, else try rating 1 (with large-region requirement)."
    )
    decode_barcodes_for_sets(all_sets)

    good = 0
    bad = 0
    for ps in all_sets:
        if ps.barcode:
            good += 1
        else:
            bad += 1
        rename_product_set(ps)

    print(f"[INFO] Completed. Sets with decoded barcode: {good}, without: {bad}")
    print(f"[TOTAL] Finished in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
