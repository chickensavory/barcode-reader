import re
import subprocess
import time
from pathlib import Path
from typing import Optional, List

from barcode_reader import readBarcode_hf

INPUT_DIR = Path("input")
GOOD_DIR = Path("good")
BAD_DIR = Path("bad")


def extract_index(path: Path) -> int:
    m = re.search(r"\((\d+)\)", path.stem)
    return int(m.group(1)) if m else 999999


def convertNEFToTempPNG(nef_path: Path) -> Optional[Path]:
    if nef_path.suffix.lower() != ".nef":
        return None
    tmp_png = nef_path.with_suffix(".barcode_tmp.png")
    print(f"[NEF] Converting {nef_path.name} -> {tmp_png.name} for barcode scan")
    try:
        result = subprocess.run(
            ["sips", "-s", "format", "png", str(nef_path), "--out", str(tmp_png)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[NEF] sips failed: {result.stderr.strip()}")
            return None
    except FileNotFoundError:
        print("[NEF] ERROR: 'sips' not found (requires macOS)")
        return None
    if not tmp_png.exists():
        print("[NEF] ERROR: temp PNG missing after conversion")
        return None
    print(f"[NEF] Temp PNG created: {tmp_png.name}")
    return tmp_png


def sanitizeBarcodeForFileName(barcode_text: str) -> str:
    sanitized = re.sub(r"[^\dA-Za-z]+", "_", barcode_text).strip("_")
    return sanitized or "barcode"


def renameWithBarcodeForPair(barcode_path: Path, product_path: Path, barcode_text: str):
    sanitized = sanitizeBarcodeForFileName(barcode_text)
    b_suffix = barcode_path.suffix.lower()
    p_suffix = product_path.suffix.lower()
    new_barcode = barcode_path.with_name(f"{sanitized}_barcode{b_suffix}")
    new_product = product_path.with_name(f"{sanitized}_product{p_suffix}")
    i = 1
    while (new_barcode.exists() and new_barcode != barcode_path) or (
        new_product.exists() and new_product != product_path
    ):
        new_barcode = barcode_path.with_name(f"{sanitized}_barcode_{i}{b_suffix}")
        new_product = product_path.with_name(f"{sanitized}_product_{i}{p_suffix}")
        i += 1
    if new_barcode != barcode_path:
        barcode_path.rename(new_barcode)
        print(f"[RENAME] {barcode_path.name} -> {new_barcode.name}")
    if new_product != product_path:
        product_path.rename(new_product)
        print(f"[RENAME] {product_path.name} -> {new_product.name}")
    return new_barcode, new_product


def getSupportedFilesSorted() -> List[Path]:
    if not INPUT_DIR.exists():
        print("[ERROR] input folder missing")
        return []
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng", ".bmp", ".nef"}
    files = [p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=extract_index)
    return files


def move_pair(barcode_path: Path, product_path: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    dest_barcode = target_dir / barcode_path.name
    dest_product = target_dir / product_path.name
    i = 1
    while dest_barcode.exists():
        dest_barcode = target_dir / f"{barcode_path.stem}_{i}{barcode_path.suffix}"
        i += 1
    j = 1
    while dest_product.exists():
        dest_product = target_dir / f"{product_path.stem}_{j}{product_path.suffix}"
        j += 1
    print(f"[MOVE] {barcode_path.name} -> {dest_barcode}")
    barcode_path.rename(dest_barcode)
    print(f"[MOVE] {product_path.name} -> {dest_product}")
    product_path.rename(dest_product)


def processPair(barcode_img: Path, product_img: Path):
    print(f"\n== Pair ==\n [1] {barcode_img.name}\n [2] {product_img.name}")
    barcode_scan_path = barcode_img
    tmp_png = None
    if barcode_img.suffix.lower() == ".nef":
        tmp_png = convertNEFToTempPNG(barcode_img)
        if tmp_png is None:
            print("[PAIR] Could not convert NEF for barcode; marking pair as bad")
            move_pair(barcode_img, product_img, BAD_DIR)
            return
        barcode_scan_path = tmp_png

    code = readBarcode_hf(barcode_scan_path)

    if tmp_png is not None:
        try:
            tmp_png.unlink()
            print(f"[NEF] Deleted temp PNG {tmp_png.name}")
        except Exception:
            pass

    if not code:
        print("[BARCODE] No barcode detected (STRICT mode)")
        move_pair(barcode_img, product_img, BAD_DIR)
        return

    print(f"[BARCODE] Detected (STRICT): {code}")
    renamed_barcode, renamed_product = renameWithBarcodeForPair(
        barcode_img, product_img, code
    )
    move_pair(renamed_barcode, renamed_product, GOOD_DIR)


def processInputFolderAsPairs():
    files = getSupportedFilesSorted()
    if not files:
        print("[INFO] No images found.")
        return
    print(f"[INFO] Found {len(files)} file(s).")
    for i in range(0, len(files), 2):
        pair = files[i : i + 2]
        if len(pair) < 2:
            print(f"[WARN] Unpaired leftover: {pair[0].name}")
            break
        processPair(pair[0], pair[1])

def main():
    t0 = time.time()
    processInputFolderAsPairs()
    print(f"[TOTAL] Completed in {time.time()-t0:.2f}s")
    
if __name__ == "__main__":
    main()