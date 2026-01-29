import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import threading
import re
import subprocess
import time
import json
import getpass
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

from barcode_changer_tool.barcode_reader import readBarcode_hf_status, BarcodeStatus
from barcode_changer_tool.barcode_rename import (
    read_xmp_label,
    token_from_color_label,
    role_from_xmp,
    write_processed_tags,
)

TRACK_ENDPOINT = os.environ.get(
    "BARCODE_TRACK_ENDPOINT", "https://sofiakris-barcodereader.hf.space/track"
)

KEYCHAIN_SERVICE = "barcode-changer-tracker"
KEYCHAIN_ACCOUNT_HF = "hf_token"
KEYCHAIN_ACCOUNT_TRACKER = "tracker_token"


def _keychain_get(account: str) -> Optional[str]:
    try:
        res = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                account,
                "-w",
            ],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            return None
        return (res.stdout or "").strip() or None
    except Exception:
        return None


def _keychain_set(account: str, secret: str) -> bool:
    try:
        subprocess.run(
            [
                "security",
                "delete-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                account,
            ],
            capture_output=True,
            text=True,
        )
        res = subprocess.run(
            [
                "security",
                "add-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                account,
                "-w",
                secret,
            ],
            capture_output=True,
            text=True,
        )
        return res.returncode == 0
    except Exception:
        return False


def _ensure_tokens_or_prompt_once() -> Tuple[Optional[str], Optional[str]]:
    hf_token = _keychain_get(KEYCHAIN_ACCOUNT_HF)
    tracker_token = _keychain_get(KEYCHAIN_ACCOUNT_TRACKER)

    env_hf = os.environ.get("HF_ACCESS_TOKEN")
    env_tracker = os.environ.get("TRACKER_TOKEN")

    if not hf_token and env_hf:
        hf_token = env_hf.strip()
        _keychain_set(KEYCHAIN_ACCOUNT_HF, hf_token)

    if not tracker_token and env_tracker:
        tracker_token = env_tracker.strip()
        _keychain_set(KEYCHAIN_ACCOUNT_TRACKER, tracker_token)

    if not hf_token:
        print("[TRACK] Hugging Face access token not found in Keychain.")
        hf_token = getpass.getpass("Enter Hugging Face access token (hf_...): ").strip()
        if hf_token:
            _keychain_set(KEYCHAIN_ACCOUNT_HF, hf_token)

    if not tracker_token:
        print("[TRACK] Tracker token not found in Keychain.")
        tracker_token = getpass.getpass(
            "Enter tracker token (hex / random string): "
        ).strip()
        if tracker_token:
            _keychain_set(KEYCHAIN_ACCOUNT_TRACKER, tracker_token)

    if not hf_token or not tracker_token:
        print("[TRACK] Tracking disabled: missing token(s).")
        return None, None

    return hf_token, tracker_token


def _post_run_counts(
    *, hf_token: str, tracker_token: str, processed: int, unprocessed: int
) -> None:
    payload = json.dumps(
        {"processed": int(processed), "unprocessed": int(unprocessed)}
    ).encode("utf-8")
    req = urllib.request.Request(
        TRACK_ENDPOINT,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {hf_token}",
            "X-Tracker-Token": tracker_token,
            "User-Agent": "barcode-changer/1.0",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            _ = resp.read()
        print(
            f"[TRACK] Sent counts to API: processed={processed}, unprocessed={unprocessed}"
        )
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        print(f"[TRACK] FAILED ({e.code}): {body[:120].strip()}")
    except Exception as e:
        print(f"[TRACK] FAILED: {type(e).__name__}: {e}")


YOLO_LOCK = threading.Lock()

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

ANCHOR_RATING = 1
MAX_WORKERS = min(8, (os.cpu_count() or 4))

PROCESS_TOOL = "barcode-changer"


@dataclass
class Photo:
    path: Path
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
    stem = path.stem

    m = re.search(r"\((\d+)\)", stem)
    if m:
        return int(m.group(1))

    m = re.match(r"^\s*(\d+)", stem)
    if m:
        return int(m.group(1))

    nums = re.findall(r"(\d+)", stem)
    if nums:
        return int(nums[-1])

    return 999999


def sanitizeBarcodeForFileName(barcode_text: str) -> str:
    sanitized = re.sub(r"[^\dA-Za-z]+", "_", (barcode_text or "")).strip("_")
    return sanitized or "barcode"


def sanitizeStemToken(token: str) -> str:
    token = (token or "").strip()
    token = re.sub(r"[^\dA-Za-z]+", "_", token).strip("_")
    return token or "img"


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
        print("[RAW] ERROR: 'sips' not found. RAW barcode scan will be skipped.")
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


def loadPhotos() -> List[Photo]:
    files = listInputFiles()
    if not files:
        print("[INFO] No image found.")
        return []

    photos: List[Photo] = []
    for p in files:
        meta = read_xmp_label(p)
        rating, label = meta.rating, meta.label

        photos.append(
            Photo(
                path=p,
                index=extractIndex(p),
                rating=rating,
                label=label,
            )
        )

    photos.sort(key=lambda ph: (ph.index, ph.path.name.lower()))
    return photos


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


def build_product_sets(photos_in_order: List[Photo]) -> List[ProductSet]:
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

    for ph in photos_in_order:
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


def _add_processed_tags_after_successful_rename(
    image_path_in_good: Path,
    *,
    tool: str = PROCESS_TOOL,
    processed_date: Optional[str] = None,
) -> bool:
    processed_date = processed_date or date.today().isoformat()

    ok = write_processed_tags(
        image_path_in_good,
        tool=tool,
        processed_date=processed_date,
        write_sidecar=True,
        embed_jpeg=True,
    )

    if ok:
        print(
            f"[XMP] tagged: {image_path_in_good.name} (tool={tool} date={processed_date})"
        )
    else:
        print(f"[XMP] FAILED to tag: {image_path_in_good.name}")

    return ok


def rename_product_set(ps: ProductSet) -> Tuple[int, int]:
    processed = 0
    unprocessed = 0

    if not ps.barcode:
        move_set_to_bad(ps, reason="barcode_failed_for_entire_set")
        unprocessed += len(ps.photos)
        return processed, unprocessed

    barcode = sanitizeBarcodeForFileName(ps.barcode)
    processed_date = date.today().isoformat()

    for idx, ph in enumerate(ps.photos, start=1):
        meta = read_xmp_label(ph.path)
        _rating, label = meta.rating, meta.label

        role, _r2, _l2, reason = role_from_xmp(ph.path)
        token: Optional[str] = None

        mapped = token_from_color_label(label)
        if mapped:
            token = mapped

        if not token and role:
            token = str(role).strip().lower()

        if not token and label:
            token = str(label).strip().lower()
            print(
                f"[ROLE] No mapped color/role; using label fallback => {token} ({reason})"
            )

        if not token:
            token = f"img{idx:02d}"
            print(
                f"[ROLE] No xmp role/label; using ordinal fallback => {token} ({reason})"
            )

        token = sanitizeStemToken(token).lower()
        dest = GOOD_DIR / f"{barcode}_{token}{ph.path.suffix.lower()}"

        try:
            new_path = _safe_rename(ph.path, dest)
            processed += 1
        except Exception as e:
            print(
                f"[RENAME] ERROR: could not rename {ph.path.name}: {type(e).__name__}: {e}"
            )
            moveToBad(ph.path, reason="rename_failed")
            unprocessed += 1
            continue

        _add_processed_tags_after_successful_rename(
            new_path,
            tool=PROCESS_TOOL,
            processed_date=processed_date,
        )

    return processed, unprocessed


def main():
    hf_token, tracker_token = _ensure_tokens_or_prompt_once()

    t0 = time.time()

    photos = loadPhotos()
    if not photos:
        return

    print(f"[INFO] Found {len(photos)} file(s).")
    print("[INFO] Ordering: filename/sequence ONLY. Timestamps are ignored completely.")
    print(
        "[INFO] Building product sets: each set starts at rating 1 (anchor). "
        "All following images belong to the set until the next rating 1."
    )

    all_sets = build_product_sets(photos)
    print(f"[INFO] Built {len(all_sets)} product set(s).")

    if not all_sets:
        print("[INFO] No product sets found (no rating-1 anchors).")
        return

    print(
        "[INFO] Decoding barcodes per set: try anchor first, then all images in set order."
    )
    decode_barcodes_for_sets(all_sets)

    sets_with_barcode = 0
    sets_without_barcode = 0

    processed_images = 0
    unprocessed_images = 0

    for ps in all_sets:
        if ps.barcode:
            sets_with_barcode += 1
        else:
            sets_without_barcode += 1

        p, u = rename_product_set(ps)
        processed_images += p
        unprocessed_images += u

    print(
        f"[INFO] Completed. Sets with decoded barcode: {sets_with_barcode}, without: {sets_without_barcode}"
    )
    print(
        f"[COUNT] Images processed: {processed_images}, unprocessed: {unprocessed_images}"
    )
    print(f"[TOTAL] Finished in {time.time() - t0:.2f}s")

    if hf_token and tracker_token:
        _post_run_counts(
            hf_token=hf_token,
            tracker_token=tracker_token,
            processed=processed_images,
            unprocessed=unprocessed_images,
        )


if __name__ == "__main__":
    main()
