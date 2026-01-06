import os, re, subprocess, time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from barcode_changer_tool.barcode_reader import readBarcode_hf_status, BarcodeStatus
from barcode_changer_tool.barcode_rename import read_xmp_rating_and_label, role_from_xmp

# Keep CPU usage sane for OpenCV / BLAS stacks
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
GOOD_DIR = Path("good")
BAD_DIR = Path("bad")

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

RAW_EXTS = {
    ".nef",
    ".arw",
    ".cr2",
    ".cr3",
}

# Time gap that starts a new set
RESET_GAP_SEC = 120.0

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
    photos: List[Photo]
    barcode: Optional[str] = None


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
    """Convert RAW to a temporary PNG for barcode scanning (macOS sips)."""
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
    """Returns (scan_path, tmp_file_to_cleanup)."""
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

    # Stable ordering within time blocks
    photos.sort(key=lambda ph: (ph.timestamp, ph.index))
    return photos


def chunk_by_reset_gap(photos: List[Photo]) -> List[List[Photo]]:
    """Split by time gaps only. No star/rating logic here."""
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
    """Loose mode: a time block = one product set, any number of photos."""
    if not block:
        return []
    return [ProductSet(photos=list(block))]


def _try_decode_barcode(path: Path) -> Optional[str]:
    scan_path, tmp_png = _prepare_scan_path(path)
    try:
        status, code = readBarcode_hf_status(str(scan_path))
        return code if status == BarcodeStatus.BARCODE and code else None
    finally:
        if tmp_png is not None and tmp_png.exists():
            try:
                tmp_png.unlink()
            except Exception:
                pass


def decode_barcodes_for_sets(product_sets: List[ProductSet]) -> None:
    """
    Loose mode: try to decode from ANY photo in the set.
    For speed, decode sets in parallel, but within a set we scan in order.
    """

    def _decode_one_set(si: int, ps: ProductSet) -> Tuple[int, Optional[str]]:
        for ph in ps.photos:
            code = _try_decode_barcode(ph.path)
            if code:
                print(f"[BARCODE] set#{si+1}: decoded from {ph.path.name} => {code}")
                return si, code
        return si, None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [
            ex.submit(_decode_one_set, i, ps) for i, ps in enumerate(product_sets)
        ]
        for fut in as_completed(futures):
            si, code = fut.result()
            product_sets[si].barcode = code


def rename_product_set(ps: ProductSet):
    """
    If barcode exists: rename all photos to good/<barcode>_<role>.<ext>
    Role is read from XMP when possible; if missing, fall back to image_<n>.
    """
    if not ps.barcode:
        for ph in ps.photos:
            moveToBad(ph.path, reason="barcode_not_decoded")
        return

    barcode = sanitizeBarcodeForFileName(ps.barcode)

    for idx, ph in enumerate(ps.photos, start=1):
        role, r2, label, reason = role_from_xmp(ph.path)

        if not role:
            role = f"image_{idx}"

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
        "[INFO] Loose mode: no strict star/rating requirements; scanning any photo for barcode..."
    )

    all_sets: List[ProductSet] = []
    for bi, block in enumerate(blocks, start=1):
        sets = build_product_sets(block)
        print(f"[INFO] Block {bi}: {len(sets)} set(s), {len(block)} photo(s).")
        all_sets.extend(sets)

    if not all_sets:
        print("[INFO] No sets found.")
        return

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
