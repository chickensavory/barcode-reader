from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

DEFAULT_LABEL_TO_TOKEN: Dict[str, str] = {
    "Red": "upc",
    "Yellow": "hero",
    "Green": "packaging",
    "Blue": "nutritional",
}

SOI = b"\xff\xd8"
XMP_ID = b"http://ns.adobe.com/xap/1.0/\x00"

_slug_rx = re.compile(r"[^a-z0-9]+", re.IGNORECASE)
DEFAULT_PROCESS_TOOL = "barcode-changer"

_processed_kw_rx = re.compile(
    r"^\s*ProcessedWith\s*:\s*(?P<tool>.+?)\s*$", re.IGNORECASE
)
_processed_desc_rx = re.compile(
    r"\bProcessed\s+by\s+(?P<tool>.+?)\s+on\s+(?P<date>\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)


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

        if marker in (0xD9, 0xDA):
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
    keywords: List[str]
    description: Optional[str]


@dataclass(frozen=True)
class ProcessInfo:
    processed: bool
    tool: Optional[str]
    date: Optional[str]
    keyword_present: bool
    description_present: bool
    status: str


def _decode_xml_bytes(xmp_xml_bytes: bytes) -> Optional[str]:
    if not xmp_xml_bytes:
        return None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return xmp_xml_bytes.decode(enc, errors="replace")
        except Exception:
            continue
    return None


def _collect_keywords_and_description(
    root: ET.Element,
) -> tuple[List[str], Optional[str]]:
    keywords: List[str] = []
    description: Optional[str] = None

    li_texts: List[str] = []
    for elem in root.iter():
        if _localname(elem.tag) == "li":
            t = (elem.text or "").strip()
            if t:
                li_texts.append(t)

    for t in li_texts:
        if _processed_kw_rx.match(t):
            keywords.append(t)

    for elem in root.iter():
        if _localname(elem.tag) == "description":
            best: Optional[str] = None
            xdefault: Optional[str] = None
            for child in elem.iter():
                if _localname(child.tag) != "li":
                    continue
                txt = (child.text or "").strip()
                if not txt:
                    continue
                lang = ""
                for ak, av in child.attrib.items():
                    if _localname(str(ak)) == "lang":
                        lang = str(av).strip().lower()
                        break
                if lang == "x-default":
                    xdefault = txt
                if best is None:
                    best = txt
            description = xdefault or best
            if description:
                break

    if not description:
        for elem in root.iter():
            lt = _localname(elem.tag)
            if lt in ("caption", "headline", "instructions", "title"):
                t = (elem.text or "").strip()
                if t:
                    description = t
                    break

    return keywords, description


def _parse_xmp_all(xmp_xml_bytes: bytes) -> XmpMeta:
    txt = _decode_xml_bytes(xmp_xml_bytes)
    if txt is None:
        return XmpMeta(None, None, [], None)

    try:
        root = ET.fromstring(txt)
    except Exception:
        return XmpMeta(None, None, [], None)

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

    keywords, description = _collect_keywords_and_description(root)
    return XmpMeta(
        rating=rating, label=label, keywords=keywords, description=description
    )


def _xmp_sidecar_path(image_path: Path) -> Path:
    p1 = image_path.with_suffix(".xmp")
    if p1.exists():
        return p1

    p2 = image_path.with_name(image_path.name + ".xmp")
    return p2


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
        return XmpMeta(None, None, [], None)
    return _parse_xmp_all(xmp_bytes)


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


def process_info_from_xmp(
    image_path: Union[str, Path],
    tool: str = DEFAULT_PROCESS_TOOL,
) -> ProcessInfo:
    meta = read_xmp_meta(image_path)

    if (
        (meta.rating is None)
        and (not meta.label)
        and (not meta.keywords)
        and (not meta.description)
    ):
        return ProcessInfo(
            processed=False,
            tool=None,
            date=None,
            keyword_present=False,
            description_present=False,
            status="no_xmp_found",
        )

    tool_norm = (tool or "").strip().lower()
    found_kw_tool: Optional[str] = None

    for kw in meta.keywords or []:
        m = _processed_kw_rx.match(kw)
        if not m:
            continue
        kw_tool = (m.group("tool") or "").strip()
        if kw_tool.lower() == tool_norm:
            found_kw_tool = kw_tool
            break

    found_desc_tool: Optional[str] = None
    found_desc_date: Optional[str] = None
    if meta.description:
        md = _processed_desc_rx.search(meta.description)
        if md:
            desc_tool = (md.group("tool") or "").strip()
            desc_date = (md.group("date") or "").strip()
            if desc_tool.lower() == tool_norm:
                found_desc_tool = desc_tool
                found_desc_date = desc_date

    keyword_present = found_kw_tool is not None
    description_present = found_desc_tool is not None

    if keyword_present and description_present:
        return ProcessInfo(
            processed=True,
            tool=found_desc_tool or found_kw_tool,
            date=found_desc_date,
            keyword_present=True,
            description_present=True,
            status="processed",
        )

    if keyword_present and not description_present:
        return ProcessInfo(
            processed=True,
            tool=found_kw_tool,
            date=None,
            keyword_present=True,
            description_present=False,
            status="keyword_only",
        )

    if (not keyword_present) and description_present:
        return ProcessInfo(
            processed=True,
            tool=found_desc_tool,
            date=found_desc_date,
            keyword_present=False,
            description_present=True,
            status="description_only",
        )

    return ProcessInfo(
        processed=False,
        tool=None,
        date=None,
        keyword_present=False,
        description_present=False,
        status="not_processed",
    )


def processed_status_string(
    image_path: Union[str, Path],
    tool: str = DEFAULT_PROCESS_TOOL,
) -> str:
    info = process_info_from_xmp(image_path, tool=tool)

    if info.processed:
        if info.date:
            return f"processed by {tool} on {info.date}"
        return f"processed by {tool}"

    return f"not processed with {tool}"
