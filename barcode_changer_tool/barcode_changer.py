import re, os, subprocess, time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from bisect import bisect_left

from barcode_reader import readBarcode_hf_status, BarcodeStatus

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

RAW_EXTS = {".nef", ".arw", ".cr2", ".cr3"}

RESET_GAP_SEC = 120.0
MAX_ASSIGN_SEC = 90.0

MAX_WORKERS = min(8, (os.cpu_count() or 4))


@dataclass
class Photo:
    path: Path
    timestamp: datetime
    index: int
    code: Optional[str]
    is_barcode: bool
    is_unsure: bool = False


@dataclass
class Session:
    code: str
    barcode_photos: List[Photo]
    other_photos: List[Photo]


def extractIndex(path: Path) -> int:
    m = re.search(r"\((\d+)\)", path.stem)
    return int(m.group(1)) if m else 999999


def sanitizeBarcodeForFileName(barcode_text: str) -> str:
    sanitized = re.sub(r"[^\dA-Za-z]+", "_", barcode_text).strip("_")
    return sanitized or "barcode"


def convertRawToTempPNG(raw_path: Path) -> Optional[Path]:
    if raw_path.suffix.lower() not in RAW_EXTS:
        return None

    tmp_png = raw_path.with_suffix(".barcode_tmp.png")
    print(f"[RAW] Converting to {raw_path.name} -> {tmp_png.name} for barcode scan")
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
        print("[RAW] ERROR: 'sips' not found")
        return None

    if not tmp_png.exists():
        print(f"[RAW] ERROR: temp PNG missing after conversion of {raw_path.name}")
        return None

    return tmp_png


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


def scanPhoto(path: Path) -> Photo:
    index = extractIndex(path)
    timestamp = getPhotoTimestamp(path)

    scan_path = path
    tmp_png: Optional[Path] = None

    if path.suffix.lower() in RAW_EXTS:
        tmp_png = convertRawToTempPNG(path)
        if tmp_png is None:
            print(f"[SCAN] Skipping RAW {path.name} (no temp PNG)")
            return Photo(
                path=path,
                timestamp=timestamp,
                index=index,
                code=None,
                is_barcode=False,
                is_unsure=False,
            )
        scan_path = tmp_png

    status, code = readBarcode_hf_status(str(scan_path))

    if tmp_png is not None:
        try:
            tmp_png.unlink()
        except Exception:
            pass

    if status == BarcodeStatus.BARCODE and code:
        print(f"[SCAN] {path.name}: BARCODE {code}")
        return Photo(
            path=path,
            timestamp=timestamp,
            index=index,
            code=code,
            is_barcode=True,
            is_unsure=False,
        )

    if status == BarcodeStatus.UNSURE:
        print(f"[SCAN] {path.name}: UNSURE")
        return Photo(
            path=path,
            timestamp=timestamp,
            index=index,
            code=None,
            is_barcode=False,
            is_unsure=True,
        )

    return Photo(
        path=path,
        timestamp=timestamp,
        index=index,
        code=None,
        is_barcode=False,
        is_unsure=False,
    )


def scanAndClassifyPhotos(max_workers: int = MAX_WORKERS) -> List[Photo]:
    files = listInputFiles()
    if not files:
        print("[INFO] No image found.")
        return []

    print(f"[INFO] Found {len(files)} file(s). Scanning for barcodes...")
    photos: List[Photo] = []

    def _worker(p: Path) -> Photo:
        try:
            return scanPhoto(p)
        except Exception as e:
            print(f"[ERROR] Unexpected error scanning {p.name}: {e}")
            ts = getPhotoTimestamp(p)
            return Photo(
                path=p,
                timestamp=ts,
                index=extractIndex(p),
                code=None,
                is_barcode=False,
                is_unsure=False,
            )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, p): p for p in files}
        for i, fut in enumerate(as_completed(futures), start=1):
            photos.append(fut.result())
            if i % 10 == 0 or i == len(futures):
                print(f"[INFO] Scanned {i}/{len(futures)}")

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


def _nearest_anchor_index(anchor_times: List[float], t: float) -> Optional[int]:
    if not anchor_times:
        return None
    j = bisect_left(anchor_times, t)
    if j == 0:
        return 0
    if j >= len(anchor_times):
        return len(anchor_times) - 1
    if abs(t - anchor_times[j - 1]) <= abs(anchor_times[j] - t):
        return j - 1
    return j


def buildSessions_nearest(photos: List[Photo]) -> List[Session]:
    sessions: List[Session] = []

    for block in chunk_by_reset_gap(photos):
        anchors = [ph for ph in block if ph.is_barcode and ph.code]
        if not anchors:
            for ph in block:
                moveToBad(ph.path)
            continue

        anchor_times = [a.timestamp.timestamp() for a in anchors]
        anchor_to_session: Dict[int, Session] = {
            i: Session(
                code=anchors[i].code or "barcode",
                barcode_photos=[anchors[i]],
                other_photos=[],
            )
            for i in range(len(anchors))
        }

        for ph in block:
            if ph.is_barcode and ph.code:
                continue

            t = ph.timestamp.timestamp()
            k = _nearest_anchor_index(anchor_times, t)
            if k is None:
                moveToBad(ph.path)
                continue

            dt = abs(t - anchor_times[k])
            if dt <= MAX_ASSIGN_SEC:
                anchor_to_session[k].other_photos.append(ph)
            else:
                moveToBad(ph.path)

        sessions.extend(anchor_to_session.values())

    return sessions


def moveToBad(path: Path):
    BAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = BAD_DIR / path.name
    i = 1
    while dest.exists():
        dest = BAD_DIR / f"{path.stem}_{i}{path.suffix}"
        i += 1
    print(f"[MOVE] {path.name} -> {dest}")
    path.rename(dest)


def processSession(session: Session):
    GOOD_DIR.mkdir(parents=True, exist_ok=True)

    sanitized = sanitizeBarcodeForFileName(session.code)

    for idx, ph in enumerate(session.barcode_photos, start=1):
        suffix = ph.path.suffix.lower()
        base = (
            f"{sanitized}_barcode"
            if len(session.barcode_photos) == 1
            else f"{sanitized}_barcode_{idx}"
        )
        dest = GOOD_DIR / f"{base}{suffix}"
        n = 1
        while dest.exists():
            dest = GOOD_DIR / f"{base}_{n}{suffix}"
            n += 1
        print(f"[RENAME] {ph.path.name} -> {dest.name}")
        ph.path.rename(dest)

    for idx, ph in enumerate(session.other_photos, start=1):
        suffix = ph.path.suffix.lower()
        base = f"{sanitized}_product_{idx}"
        dest = GOOD_DIR / f"{base}{suffix}"
        n = 1
        while dest.exists():
            dest = GOOD_DIR / f"{base}_{n}{suffix}"
            n += 1
        print(f"[RENAME] {ph.path.name} -> {dest.name}")
        ph.path.rename(dest)


def main():
    t0 = time.time()
    photos = scanAndClassifyPhotos()
    if not photos:
        return

    sessions = buildSessions_nearest(photos)
    print(f"[INFO] Built {len(sessions)} session(s).")

    for s in sessions:
        print(
            f"[SESSION] {s.code}: {len(s.barcode_photos)} barcode, {len(s.other_photos)} other"
        )
        processSession(s)

    print(f"[TOTAL] Completed in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
