import re, os, subprocess, time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from barcode_changer_tool.barcode_reader import readBarcode_hf

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

CHUNK_GAP_SEC = 80.0

MAX_WORKERS = min(8, (os.cpu_count() or 4))


@dataclass
class Photo:
    path: Path
    timestamp: datetime
    index: int
    code: Optional[str]
    is_barcode: bool


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
    suffix = raw_path.suffix.lower()
    if suffix not in RAW_EXTS:
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

    print(f"[RAW] Temp PNG created: {tmp_png.name}")
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
    files: List[Path] = []
    for p in INPUT_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    return files


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
            )
        scan_path = tmp_png

    code: Optional[str] = readBarcode_hf(str(scan_path))

    if tmp_png is not None:
        try:
            tmp_png.unlink()
            print(f"[RAW] Deleted temp PNG {tmp_png.name}")
        except Exception:
            pass

    if code:
        print(f"[SCAN] {path.name}: BARCODE {code}")
        return Photo(
            path=path,
            timestamp=timestamp,
            index=index,
            code=code,
            is_barcode=True,
        )
    else:
        print(f"[SCAN] {path.name}: no barcode")
        return Photo(
            path=path,
            timestamp=timestamp,
            index=index,
            code=None,
            is_barcode=False,
        )


def scanAndClassifyPhotos(max_workers: int = MAX_WORKERS) -> List[Photo]:
    files = listInputFiles()
    if not files:
        print("[INFO] No image found.")
        return []

    print(f"[INFO] Found {len(files)} file(s). Scanning for barcodes...")
    print(f"[INFO] Using up to {max_workers} worker thread(s).")

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
            )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(_worker, p): p for p in files}
        total = len(future_to_path)
        for i, future in enumerate(as_completed(future_to_path), start=1):
            p = future_to_path[future]
            photo = future.result()
            photos.append(photo)
            print(f"[INFO] Finished {i}/{total}: {p.name}")

    photos.sort(key=lambda ph: (ph.timestamp, ph.index))
    return photos


def chunk_photos_by_time(photos: List[Photo]) -> List[List[Photo]]:
    if not photos:
        return []

    chunks: List[List[Photo]] = []
    current_chunk: List[Photo] = [photos[0]]

    for prev, curr in zip(photos, photos[1:]):
        gap = (curr.timestamp - prev.timestamp).total_seconds()
        if gap > CHUNK_GAP_SEC:
            chunks.append(current_chunk)
            current_chunk = [curr]
        else:
            current_chunk.append(curr)

    chunks.append(current_chunk)
    return chunks


def buildSessionsFromChunk(chunk: List[Photo]) -> List[Session]:
    sessions: List[Session] = []
    current: Optional[Session] = None
    seen_barcode_in_chunk = False

    for ph in chunk:
        if ph.is_barcode and ph.code:
            if current is None or ph.code != current.code:
                current = Session(code=ph.code, barcode_photos=[], other_photos=[])
                sessions.append(current)
            current.barcode_photos.append(ph)
            seen_barcode_in_chunk = True
        else:
            if not seen_barcode_in_chunk:
                continue
            if current is not None:
                current.other_photos.append(ph)

    return sessions


def buildSessions(photos: List[Photo]) -> List[Session]:
    chunks = chunk_photos_by_time(photos)
    sessions: List[Session] = []
    for chunk in chunks:
        sessions.extend(buildSessionsFromChunk(chunk))
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
        if len(session.barcode_photos) == 1:
            base = f"{sanitized}_barcode"
        else:
            base = f"{sanitized}_barcode_{idx}"
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


def processUnassignedPhotos(photos: List[Photo], sessions: List[Session]):
    session_paths = {
        ph.path for s in sessions for ph in (s.barcode_photos + s.other_photos)
    }
    for ph in photos:
        if ph.path not in session_paths:
            print(f"[ORPHAN] {ph.path.name} has no associated barcode; moving to bad/")
            moveToBad(ph.path)


def main():
    t0 = time.time()
    photos = scanAndClassifyPhotos()
    if not photos:
        return

    sessions = buildSessions(photos)
    print(f"[INFO] Built {len(sessions)} session(s).")

    for s in sessions:
        print(
            f"[SESSION] {s.code}: "
            f"{len(s.barcode_photos)} barcode image(s), "
            f"{len(s.other_photos)} other image(s)"
        )
        processSession(s)

    processUnassignedPhotos(photos, sessions)
    print(f"[TOTAL] Completed in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()

# TODO skip images with no product and too blurry images
