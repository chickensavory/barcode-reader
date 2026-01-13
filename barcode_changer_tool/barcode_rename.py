from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

DEFAULT_LABEL_TO_TOKEN: Dict[str, str] = {
    "Red": "upc",
    "Yellow": "hero",
    "Green": "packaging",
    "Blue": "nutritional",
}

SOI = b"\xff\xd8"
XMP_ID = b"http://ns.adobe.com/xap/1.0/\x00"

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


def _normalize_adobe_label(label: str) -> str:
    s = (label or "").strip()
    if not s:
        return ""
    s = s.lower()
    return s[:1].upper() + s[1:]


def _localname(tag_or_attr: str) -> str:
    if not isinstance(tag_or_attr, str):
        return ""
    t = tag_or_attr
    if "}" in t:
        t = t.split("}", 1)[1]
    if ":" in t:
        t = t.split(":", 1)[1]
    return t.lower().strip()


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

        if marker in (0xD9, 0xDA):  # EOI or SOS
            break

        if i + 2 > n:
            break

        seglen = struct.unpack(">H", data[i : i + 2])[0]
        seg_end = i + seglen
        payload_start = i + 2
        yield (marker, payload_start, seg_end)
        i = seg_end


def _extract_xmp_from_jpeg_bytes(data: bytes) -> Optional[bytes]:
    if not data.startswith(SOI):
        return None

    for marker, payload_start, seg_end in _iter_jpeg_segments(data):
        if marker != 0xE1:
            continue
        payload = data[payload_start:seg_end]
        if payload.startswith(XMP_ID):
            return payload[len(XMP_ID) :]
    return None


def _extract_xmpmeta_fragment_anywhere(data: bytes) -> Optional[bytes]:
    low = data.lower()
    start = low.find(b"<x:xmpmeta")
    if start == -1:
        return None
    end = low.find(b"</x:xmpmeta>", start)
    if end == -1:
        return None
    end += len(b"</x:xmpmeta>")
    return data[start:end]


@dataclass(frozen=True)
class XmpMeta:
    rating: Optional[int]
    label: Optional[str]


def _decode_xml_bytes(xmp_xml_bytes: bytes) -> Optional[str]:
    if not xmp_xml_bytes:
        return None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return xmp_xml_bytes.decode(enc, errors="replace")
        except Exception:
            continue
    return None


def _parse_xmp_label_and_rating(xmp_xml_bytes: bytes) -> XmpMeta:
    txt = _decode_xml_bytes(xmp_xml_bytes)
    if txt is None:
        return XmpMeta(None, None)

    try:
        root = ET.fromstring(txt)
    except Exception:
        return XmpMeta(None, None)

    label: Optional[str] = None
    rating: Optional[int] = None

    for elem in root.iter():
        for k, v in elem.attrib.items():
            lk = _localname(str(k))
            if lk == "label" and not label:
                sv = str(v).strip()
                if sv:
                    label = sv
            elif lk == "rating" and rating is None:
                sv = str(v).strip()
                try:
                    rating = int(sv)
                except Exception:
                    pass

    for elem in root.iter():
        lt = _localname(elem.tag)
        if lt == "label" and not label:
            t = (elem.text or "").strip()
            if t:
                label = t
        elif lt == "rating" and rating is None:
            t = (elem.text or "").strip()
            try:
                rating = int(t)
            except Exception:
                pass

    return XmpMeta(rating=rating, label=label)


def _xmp_sidecar_path(image_path: Path) -> Path:
    return image_path.with_suffix(".xmp")


def _read_xmp_packet_bytes_from_file(image_path: Path) -> Optional[bytes]:
    sidecar = _xmp_sidecar_path(image_path)
    if sidecar.exists() and sidecar.is_file():
        try:
            return sidecar.read_bytes()
        except Exception:
            pass

    try:
        data = image_path.read_bytes()
    except Exception:
        return None

    if image_path.suffix.lower() in (".jpg", ".jpeg"):
        xmp_bytes = _extract_xmp_from_jpeg_bytes(data)
        if xmp_bytes:
            return xmp_bytes

    frag = _extract_xmpmeta_fragment_anywhere(data)
    if frag:
        return frag

    return None


def read_xmp_meta(image_path: Union[str, Path]) -> XmpMeta:
    p = Path(image_path)
    xmp_bytes = _read_xmp_packet_bytes_from_file(p)
    if not xmp_bytes:
        return XmpMeta(None, None)
    return _parse_xmp_label_and_rating(xmp_bytes)


def read_xmp_label(image_path: Union[str, Path]) -> XmpMeta:
    return read_xmp_meta(image_path)


def token_from_color_label(
    label: Optional[str],
    label_to_token: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    if not label:
        return None
    mapping = label_to_token or DEFAULT_LABEL_TO_TOKEN
    norm = _normalize_adobe_label(label)
    return mapping.get(norm)


def token_from_xmp_label(
    image_path: Union[str, Path],
    label_to_token: Optional[Dict[str, str]] = None,
) -> tuple[Optional[str], Optional[str], str]:
    meta = read_xmp_meta(image_path)
    label = meta.label

    if not label:
        return None, None, "no_xmp_label_found"

    token = token_from_color_label(label, label_to_token=label_to_token)
    if not token:
        return None, label, "unmapped_label"

    return token, label, "ok"


def role_from_xmp(
    image_path: Union[str, Path],
    rating_to_role=None,
) -> tuple[Optional[str], Optional[int], Optional[str], str]:
    meta = read_xmp_meta(image_path)

    token = token_from_color_label(meta.label)
    if token:
        return token, meta.rating, meta.label, "ok"

    if meta.label:
        return None, meta.rating, meta.label, "unmapped_label"

    if meta.rating is not None:
        return None, meta.rating, None, "rating_only"

    return None, None, None, "no_xmp_found"
