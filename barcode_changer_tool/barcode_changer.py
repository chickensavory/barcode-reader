import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import threading

YOLO_LOCK = threading.Lock()

import re, subprocess, time, struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from barcode_changer_tool.barcode_reader import readBarcode_hf_status, BarcodeStatus
from barcode_changer_tool.barcode_rename import role_from_xmp

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
GOOD_DIR = Path("good_test")
BAD_DIR = Path("bad_test")

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
ANCHOR_RATING = 1
MAX_WORKERS = min(8, (os.cpu_count() or 4))

SOI = b"\xff\xd8"
XMP_ID = b"http://ns.adobe.com/xap/1.0/\x00"
XMP_NS = "http://ns.adobe.com/xap/1.0/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def _iter_jpeg_segments(data: bytes):
    if not data.startswith(SOI):
        return

    i = 2
    n = len(data)

    while i < n:
        if data[i] != 0xFF:
            i += 1
            continue

        j = i
        while j < n and data[j] == 0xFF:
            j += 1
        if j >= n:
            break

        marker = data[j]
        i = j + 1

        if marker == 0xD9:
            yield (marker, j - 1, i, i)
            break

        if marker == 0xDA:
            if i + 2 > n:
                break
            seglen = struct.unpack(">H", data[i : i + 2])[0]
            seg_start = j - 1
            seg_end = i + seglen
            payload_start = i + 2
            yield (marker, seg_start, seg_end, payload_start)
            break

        if i + 2 > n:
            break

        seglen = struct.unpack(">H", data[i : i + 2])[0]
        seg_start = j - 1
        seg_end = i + seglen
        payload_start = i + 2
        yield (marker, seg_start, seg_end, payload_start)
        i = seg_end


def _extract_xmp_packet_from_jpeg(jpeg_path: Path) -> Optional[bytes]:
    try:
        data = jpeg_path.read_bytes()
    except Exception:
        return None

    if not data.startswith(SOI):
        return None

    for marker, seg_start, seg_end, payload_start in _iter_jpeg_segments(data):
        if marker != 0xE1:
            continue
        payload = data[payload_start:seg_end]
        if payload.startswith(XMP_ID):
            return payload[len(XMP_ID) :]
    return None


def _parse_xmp_rating_and_label_from_xml(
    xmp_xml_bytes: bytes,
) -> Tuple[Optional[int], Optional[str]]:
    if not xmp_xml_bytes:
        return None, None

    txt = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            txt = xmp_xml_bytes.decode(enc, errors="replace")
            break
        except Exception:
            continue
    if txt is None:
        return None, None

    try:
        root = ET.fromstring(txt)
    except ET.ParseError:
        return None, None

    desc_tag = f"{{{RDF_NS}}}Description"
    rating_attr = f"{{{XMP_NS}}}Rating"
    rating_tag = f"{{{XMP_NS}}}Rating"
    label_attr = f"{{{XMP_NS}}}Label"
    label_tag = f"{{{XMP_NS}}}Label"

    rating: Optional[int] = None
    label: Optional[str] = None

    for e in root.iter():
        if e.tag == desc_tag:
            if rating is None:
                v = e.attrib.get(rating_attr)
                if v is not None:
                    try:
                        rating = int(str(v).strip())
                    except Exception:
                        pass
            if label is None:
                lv = e.attrib.get(label_attr)
                if lv is not None:
                    lv2 = str(lv).strip()
                    if lv2:
                        label = lv2

    if rating is None:
        for e in root.iter():
            if e.tag == rating_tag and e.text is not None:
                t = e.text.strip()
                if t:
                    try:
                        rating = int(t)
                        break
                    except Exception:
                        pass

    if label is None:
        for e in root.iter():
            if e.tag == label_tag and e.text is not None:
                t = e.text.strip()
                if t:
                    label = t
                    break

    return rating, label


def robust_read_xmp_rating_and_label(path: Path) -> Tuple[Optional[int], Optional[str]]:
    ext = path.suffix.lower()
    if ext not in (".jpg", ".jpeg"):
        return None, None

    xmp_xml = _extract_xmp_packet_from_jpeg(path)
    return _parse_xmp_rating_and_label_from_xml(xmp_xml)


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
    anchor: Photo
    barcode: Optional[str] = None
    barcode_source_path: Optional[Path] = None


def extractIndex(path: Path) -> int:
    m = re.search(r"\((\d+)\)", path.stem)
    return int(m.group(1)) if m else 999999


def sanitizeBarcodeForFileName(barcode_text: str) -> str:
    sanitized = re.sub(r"[^\dA-Za-z]+", "_", (barcode_text or "")).strip("_")
    return sanitized or "barcode"


def sanitizeStemToken(token: str) -> str:
    token = (token or "").strip()
    token = re.sub(r"[^\dA-Za-z]+", "_", token).strip("_")
    return token or "img"


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
        return robust_read_xmp_rating_and_label(path)
    except Exception as e:
        print(
            f"[XMP] ERROR reading rating/label from {path.name}: {type(e).__name__}: {e}"
        )
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
    for ph in ps.photos:
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
    cur_photos: List[Photo] = []
    cur_anchor: Optional[Photo] = None

    def flush_current():
        nonlocal cur_photos, cur_anchor
        if cur_anchor is None or not cur_photos:
            cur_photos = []
            cur_anchor = None
            return
        sets.append(ProductSet(photos=cur_photos, anchor=cur_anchor))
        cur_photos = []
        cur_anchor = None

    for ph in block:
        r = ph.rating

        if cur_anchor is None:
            if r == ANCHOR_RATING:
                cur_anchor = ph
                cur_photos = [ph]
            else:
                moveToBad(ph.path, reason="seen_before_any_anchor_rating_1")
            continue

        if r == ANCHOR_RATING:
            flush_current()
            cur_anchor = ph
            cur_photos = [ph]
            continue

        cur_photos.append(ph)

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
    ordered = [ps.anchor] + [ph for ph in ps.photos if ph.path != ps.anchor.path]

    for i, ph in enumerate(ordered, start=1):
        code = _try_decode_barcode(ph.path, require_large_region=True)
        if code:
            ps.barcode = code
            ps.barcode_source_path = ph.path
            where = "anchor" if ph.path == ps.anchor.path else f"image_{i:02d}"
            print(f"[BARCODE] decoded from {where} ({ph.path.name}) => {code}")
            return

    ps.barcode = None
    ps.barcode_source_path = None
    print("[BARCODE] failed on all images in set => barcode fail")


def decode_barcodes_for_sets(product_sets: List[ProductSet]) -> None:
    if not product_sets:
        return

    def _worker(i: int) -> int:
        try:
            decode_barcode_for_set(product_sets[i])
        except Exception as e:
            ps = product_sets[i]
            ps.barcode = None
            ps.barcode_source_path = None
            print(f"[BARCODE] ERROR decoding set {i}: {type(e).__name__}: {e}")
        return i

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_worker, i) for i in range(len(product_sets))]
        for fut in as_completed(futures):
            _ = fut.result()


def rename_product_set(ps: ProductSet):
    if not ps.barcode:
        move_set_to_bad(ps, reason="barcode_failed_for_entire_set")
        return

    barcode = sanitizeBarcodeForFileName(ps.barcode)

    for idx, ph in enumerate(ps.photos, start=1):
        role, r2, label, reason = role_from_xmp(ph.path)

        if role:
            token = sanitizeStemToken(role)
        else:
            if label:
                token = sanitizeStemToken(label)
                print(f"[ROLE] No xmp role; using label fallback => {token} ({reason})")
            else:
                token = f"img{idx:02d}"
                print(
                    f"[ROLE] No xmp role/label; using ordinal fallback => {token} ({reason})"
                )

        dest = GOOD_DIR / f"{barcode}_{token}{ph.path.suffix.lower()}"
        _safe_rename(ph.path, dest)


def main():
    t0 = time.time()

    photos = loadPhotos()
    if not photos:
        return

    blocks = chunk_by_reset_gap(photos)
    print(f"[INFO] Found {len(photos)} file(s) across {len(blocks)} time block(s).")
    print(
        "[INFO] Building product sets: each set starts at rating 1 (anchor). "
        "All following images belong to the set until the next rating 1."
    )

    all_sets: List[ProductSet] = []
    for bi, block in enumerate(blocks, start=1):
        sets = build_product_sets(block)
        print(f"[INFO] Block {bi}: {len(sets)} product set(s).")
        all_sets.extend(sets)

    if not all_sets:
        print("[INFO] No product sets found (no rating-1 anchors).")
        return

    print(
        "[INFO] Decoding barcodes per set: try anchor first, then all images in set order."
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
