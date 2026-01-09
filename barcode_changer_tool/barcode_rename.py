from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union


DEFAULT_RATING_TO_ROLE: Dict[int, str] = {
    1: "hero",
    2: "packaging",
    3: "nutritional",
    4: "upc",
}


@dataclass(frozen=True)
class XmpMeta:
    rating: Optional[int]
    label: Optional[str]


@dataclass(frozen=True)
class RenameResult:
    original: Path
    renamed: Optional[Path]
    rating: Optional[int]
    label: Optional[str]
    role: Optional[str]
    reason: str


_slug_rx = re.compile(r"[^a-z0-9]+", re.IGNORECASE)


def slugify(stem: str) -> str:
    s = (stem or "").strip().lower()
    s = _slug_rx.sub("_", s).strip("_")
    return s or "image"


def pick_nonconflicting_path(target: Path) -> Path:
    if not target.exists():
        return target
    base = target.with_suffix("")
    ext = target.suffix
    i = 2
    while True:
        cand = Path(f"{base}_{i}{ext}")
        if not cand.exists():
            return cand
        i += 1


def extract_xmp_packet(file_bytes: bytes) -> Optional[bytes]:
    start = file_bytes.find(b"<x:xmpmeta")
    if start == -1:
        low = file_bytes.lower()
        start = low.find(b"<x:xmpmeta")
        if start == -1:
            return None
    end = file_bytes.find(b"</x:xmpmeta>", start)
    if end == -1:
        return None
    end += len(b"</x:xmpmeta>")
    return file_bytes[start:end]


def localname(tag: str) -> str:
    if not isinstance(tag, str):
        return ""
    if "}" in tag:
        tag = tag.split("}", 1)[1]
    if ":" in tag:
        tag = tag.split(":", 1)[1]
    return tag.lower().strip()


def read_xmp_rating_and_label(image_path: Union[str, Path]) -> XmpMeta:
    p = Path(image_path)

    try:
        data = p.read_bytes()
    except Exception:
        return XmpMeta(None, None)

    xmp = extract_xmp_packet(data)
    if not xmp:
        return XmpMeta(None, None)

    try:
        root = ET.fromstring(xmp.decode("utf-8", errors="ignore"))
    except Exception:
        return XmpMeta(None, None)

    rating: Optional[int] = None
    label: Optional[str] = None

    for elem in root.iter():
        for k, v in elem.attrib.items():
            kname = localname(str(k))

            if rating is None and kname == "rating":
                try:
                    rating = int(str(v).strip())
                except Exception:
                    pass

            if label is None and kname == "label":
                txt = str(v).strip()
                if txt:
                    label = txt

        if rating is not None and label is not None:
            break

    if rating is None or label is None:
        for elem in root.iter():
            lname = localname(elem.tag)

            if rating is None and lname == "rating":
                txt = (elem.text or "").strip()
                if txt:
                    try:
                        rating = int(txt)
                    except Exception:
                        pass

            if label is None and lname == "label":
                txt = (elem.text or "").strip()
                if txt:
                    label = txt

            if rating is not None and label is not None:
                break

    return XmpMeta(rating=rating, label=label)


def role_from_xmp(
    image_path: Union[str, Path],
    rating_to_role: Optional[Dict[int, str]] = None,
) -> tuple[Optional[str], Optional[int], Optional[str], str]:
    rating_to_role = rating_to_role or DEFAULT_RATING_TO_ROLE
    p = Path(image_path)

    meta = read_xmp_rating_and_label(p)
    rating, label = meta.rating, meta.label

    if rating is None and (label is None or label == ""):
        return None, rating, label, "no_xmp_rating_or_label_found"

    role = rating_to_role.get(rating) if rating is not None else None
    if role is None:
        return None, rating, label, "unmapped_rating"

    return role, rating, label, "ok"


def rename_with_role_barcode(
    image_path: Union[str, Path],
    *,
    role: str,
    barcode: str,
    kind: str,
    index: int,
    dry_run: bool = False,
) -> RenameResult:
    p = Path(image_path)

    try:
        if not role:
            return RenameResult(p, None, None, None, None, "skipped_no_role")
        if not barcode:
            return RenameResult(p, None, None, None, None, "skipped_no_barcode")

        ext = p.suffix.lower()
        kind_s = slugify(kind)
        new_name = f"{role}_{barcode}_{kind_s}_{index}{ext}"
        target = pick_nonconflicting_path(p.with_name(new_name))

        if not dry_run:
            p.rename(target)

        return RenameResult(p, target, None, None, role, "renamed")
    except Exception as e:
        return RenameResult(
            Path(image_path), None, None, None, None, f"error: {type(e).__name__}"
        )


def rename_image_by_lr_xmp(
    image_path: Union[str, Path],
    rating_to_role: Optional[Dict[int, str]] = None,
    prefer: str = "rating",
    label_to_role: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> RenameResult:
    rating_to_role = rating_to_role or DEFAULT_RATING_TO_ROLE
    p = Path(image_path)

    try:
        meta = read_xmp_rating_and_label(p)
        rating, label = meta.rating, meta.label

        role: Optional[str] = None
        prefer_l = (prefer or "").lower().strip()

        if prefer_l == "rating":
            role = rating_to_role.get(rating) if rating is not None else None
        elif prefer_l == "label":
            if label_to_role and label:
                role = label_to_role.get(label)
        else:
            role = rating_to_role.get(rating) if rating is not None else None
            if role is None and label_to_role and label:
                role = label_to_role.get(label)

        if role is None:
            if rating is None and (label is None or label == ""):
                return RenameResult(p, None, None, None, None, "skipped_no_xmp")
            return RenameResult(p, None, rating, label, None, "skipped_unmapped")

        new_name = f"{role}__{slugify(p.stem)}{p.suffix.lower()}"
        target = pick_nonconflicting_path(p.with_name(new_name))

        if not dry_run:
            p.rename(target)

        return RenameResult(p, target, rating, label, role, "renamed")
    except Exception:
        return RenameResult(Path(image_path), None, None, None, None, "error")
