import re
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

from barcode_reader import readBarcode_hf

try:
    import rawpy
    import imageio.v2 as imageio
except ImportError:
    rawpy = None
    imageio = None


SUPPORTED_EXTS = {".nef", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic"}
INPUT_DIR = Path("input")


def getSupportedFilesSorted(folder: Optional[Path] = None) -> List[Path]:
    if folder is None:
        folder = INPUT_DIR

    if not folder.exists():
        print(f"[ERROR] Input folder does not exist: {folder}")
        return []

    files: List[Path] = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)

    def natural_key(p: Path):
        m = re.search(r"\((\d+)\)", p.stem)
        if m:
            return (0, int(m.group(1)))
        m2 = re.search(r"(\d+)", p.stem)
        if m2:
            return (1, int(m2.group(1)))
        return (2, p.name)

    files.sort(key=natural_key)
    print(f"[INFO] Found {len(files)} file(s).")
    return files


def convertNEFToTempPNG(nef_path: Path) -> Optional[Path]:
    if rawpy is None or imageio is None:
        print("[NEF] rawpy/imageio not available; cannot convert NEF.")
        return None

    tmp_png = nef_path.with_name(nef_path.stem + ".barcode_tmp.png")
    print(f"[NEF] Converting {nef_path.name} -> {tmp_png.name} for barcode scan")

    try:
        with rawpy.imread(str(nef_path)) as raw:
            rgb = raw.postprocess(
                output_bps=8,
                use_auto_wb=True,
                no_auto_bright=True,
                gamma=(2.2, 4.5),
            )
        imageio.imwrite(str(tmp_png), rgb)
        print(f"[NEF] Temp PNG created: {tmp_png.name}")
        return tmp_png
    except Exception as e:
        print(f"[NEF] Conversion failed for {nef_path.name}: {e}")
        try:
            if tmp_png.exists():
                tmp_png.unlink()
        except Exception:
            pass
        return None


def normalize_barcode_text(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None

    raw = raw.strip()
    if raw.isdigit():
        return raw

    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None

    return digits


def sanitize_code_for_filename(code: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]", "_", code)
    safe = safe.lstrip("./\\")
    if not safe:
        safe = "barcode"
    return safe


def renameWithBarcodeForPair(
    barcode_img: Path, product_img: Path, code: str
) -> Tuple[Path, Path]:
    ext = barcode_img.suffix.lower()
    parent = barcode_img.parent

    safe_code = sanitize_code_for_filename(code)

    new_barcode = parent / f"{safe_code}_barcode{ext}"
    new_product = parent / f"{safe_code}_product{ext}"

    print(f"[RENAME] {barcode_img.name} -> {new_barcode.name}")
    barcode_img = barcode_img.rename(new_barcode)

    print(f"[RENAME] {product_img.name} -> {new_product.name}")
    product_img = product_img.rename(new_product)

    return barcode_img, product_img


def move_pair_to_folder(barcode_img: Path, product_img: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)

    for img in (barcode_img, product_img):
        dest = dest_dir / img.name
        print(f"[MOVE] {img.name} -> {dest}")
        img.rename(dest)


def processPair(barcode_img: Path, product_img: Path):
    print(f"\n== Pair ==\n [1] {barcode_img.name}\n [2] {product_img.name}")

    base_dir = barcode_img.parent
    good_dir = base_dir / "good"
    bad_dir = base_dir / "bad"

    if barcode_img.suffix.lower() == ".nef":
        tmp_png = convertNEFToTempPNG(barcode_img)
        if tmp_png is None:
            print("[PAIR] Could not convert NEF for barcode; marking as BAD.")
            move_pair_to_folder(barcode_img, product_img, bad_dir)
            return

        try:
            raw_code = readBarcode_hf(tmp_png)
        finally:
            try:
                tmp_png.unlink()
                print(f"[NEF] Deleted temp PNG {tmp_png.name}")
            except Exception:
                pass
    else:
        raw_code = readBarcode_hf(barcode_img)

    code = normalize_barcode_text(raw_code)
    if not code:
        print(
            "[BARCODE] Decoded text is not a valid numeric barcode "
            f"(raw={raw_code!r}); marking pair as BAD"
        )
        move_pair_to_folder(barcode_img, product_img, bad_dir)
        return

    print(f"[BARCODE] Detected: {code}")
    barcode_img, product_img = renameWithBarcodeForPair(barcode_img, product_img, code)
    move_pair_to_folder(barcode_img, product_img, good_dir)


def processInputFolderAsPairs():
    files = getSupportedFilesSorted()
    if not files:
        print("[INFO] No images found.")
        return

    for i in range(0, len(files), 2):
        pair = files[i : i + 2]
        if len(pair) < 2:
            print(f"[WARN] Unpaired leftover: {pair[0].name}")
            break

        processPair(pair[0], pair[1])


def main():
    t0 = time.time()
    processInputFolderAsPairs()
    print(f"[TOTAL] Completed in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
