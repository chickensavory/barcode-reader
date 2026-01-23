from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union
from datetime import date as _date

DEFAULT_LABEL_TO_TOKEN: Dict[str, str] = {
    "Red": "upc",
    "Yellow": "hero",
    "Green": "packaging",
    "Blue": "nutritional",
}

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"
SOS = b"\xff\xda"

XMP_ID = b"http://ns.adobe.com/xap/1.0/\x00"

_slug_rx = re.compile(r"[^a-z0-9]+", re.IGNORECASE)
DEFAULT_PROCESS_TOOL = "barcode-changer"


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


def _ensure_ns(prefix: str, uri: str):
    try:
        ET.register_namespace(prefix, uri)
    except Exception:
        pass


_NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
}
_ensure_ns("x", _NS["x"])
_ensure_ns("rdf", _NS["rdf"])
_ensure_ns("dc", _NS["dc"])


def _minimal_xmp_packet_root() -> ET.Element:
    xmpmeta = ET.Element(f"{{{_NS['x']}}}xmpmeta")
    rdf = ET.SubElement(xmpmeta, f"{{{_NS['rdf']}}}RDF")
    ET.SubElement(rdf, f"{{{_NS['rdf']}}}Description")
    return xmpmeta


def _get_or_create_rdf_description(xmpmeta_root: ET.Element) -> ET.Element:
    rdf = None
    for el in xmpmeta_root.iter():
        if el.tag == f"{{{_NS['rdf']}}}RDF":
            rdf = el
            break
    if rdf is None:
        rdf = ET.SubElement(xmpmeta_root, f"{{{_NS['rdf']}}}RDF")

    for el in list(rdf):
        if el.tag == f"{{{_NS['rdf']}}}Description":
            return el

    return ET.SubElement(rdf, f"{{{_NS['rdf']}}}Description")


def _find_child(parent: ET.Element, ns: str, name: str) -> Optional[ET.Element]:
    tag = f"{{{ns}}}{name}"
    for ch in list(parent):
        if ch.tag == tag:
            return ch
    return None


def _ensure_dc_subject_keyword(desc: ET.Element, keyword: str) -> bool:
    changed = False

    dc_subject = _find_child(desc, _NS["dc"], "subject")
    if dc_subject is None:
        dc_subject = ET.SubElement(desc, f"{{{_NS['dc']}}}subject")
        changed = True

    bag = None
    for ch in list(dc_subject):
        if ch.tag == f"{{{_NS['rdf']}}}Bag":
            bag = ch
            break
    if bag is None:
        bag = ET.SubElement(dc_subject, f"{{{_NS['rdf']}}}Bag")
        changed = True

    for li in list(bag):
        if li.tag == f"{{{_NS['rdf']}}}li" and (li.text or "").strip() == keyword:
            return changed

    li = ET.SubElement(bag, f"{{{_NS['rdf']}}}li")
    li.text = keyword
    return True


def _ensure_dc_description_xdefault(desc: ET.Element, text: str) -> bool:
    changed = False

    dc_desc = _find_child(desc, _NS["dc"], "description")
    if dc_desc is None:
        dc_desc = ET.SubElement(desc, f"{{{_NS['dc']}}}description")
        changed = True

    alt = None
    for ch in list(dc_desc):
        if ch.tag == f"{{{_NS['rdf']}}}Alt":
            alt = ch
            break
    if alt is None:
        alt = ET.SubElement(dc_desc, f"{{{_NS['rdf']}}}Alt")
        changed = True

    xml_lang_key = "{http://www.w3.org/XML/1998/namespace}lang"
    for li in list(alt):
        if li.tag != f"{{{_NS['rdf']}}}li":
            continue
        lang = (li.attrib.get(xml_lang_key, "") or "").strip().lower()
        if lang == "x-default":
            if (li.text or "") == text:
                return changed
            li.text = text
            return True

    li = ET.SubElement(alt, f"{{{_NS['rdf']}}}li")
    li.set(xml_lang_key, "x-default")
    li.text = text
    return True


def _parse_or_create_xmpmeta_root(xmp_xml_bytes: Optional[bytes]) -> ET.Element:
    if xmp_xml_bytes:
        txt = _decode_xml_bytes(xmp_xml_bytes)
        if txt:
            try:
                root = ET.fromstring(txt)
                if _localname(root.tag) == "xmpmeta":
                    return root
                for el in root.iter():
                    if _localname(el.tag) == "xmpmeta":
                        return el
            except Exception:
                pass
    return _minimal_xmp_packet_root()


def _serialize_xmpmeta(root: ET.Element) -> bytes:
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _sidecar_path_for_image(image_path: Path) -> Path:
    raw_like = image_path.suffix.lower() in (".nef", ".arw", ".cr2", ".cr3", ".dng")
    if raw_like:
        return image_path.with_suffix(".xmp")
    return image_path.with_name(image_path.name + ".xmp")


def write_processed_xmp_sidecar(
    image_path: Union[str, Path],
    *,
    tool: str = DEFAULT_PROCESS_TOOL,
    processed_date: Optional[str] = None,
) -> bool:
    p = Path(image_path)
    processed_date = processed_date or _date.today().isoformat()
    keyword = f"ProcessedWith:{tool}"
    desc_text = f"Processed by {tool} on {processed_date}"

    sidecar = _sidecar_path_for_image(p)

    existing = None
    if sidecar.exists():
        try:
            existing = sidecar.read_bytes()
        except Exception:
            existing = None

    root = _parse_or_create_xmpmeta_root(existing)
    rdf_desc = _get_or_create_rdf_description(root)

    changed = False
    changed = _ensure_dc_subject_keyword(rdf_desc, keyword) or changed
    changed = _ensure_dc_description_xdefault(rdf_desc, desc_text) or changed

    if not changed and sidecar.exists():
        return True

    try:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_bytes(_serialize_xmpmeta(root))
        return True
    except Exception:
        return False


def _iter_jpeg_segments_with_bounds(data: bytes):
    if not data.startswith(SOI):
        return
    i = 2
    n = len(data)

    while i < n:
        if data[i] != 0xFF:
            i += 1
            continue

        seg_start = i
        j = i
        while j < n and data[j] == 0xFF:
            j += 1
        if j >= n:
            break

        marker = data[j]
        i = j + 1

        if marker == 0xD9:
            yield (marker, seg_start, seg_start + 2, seg_start + 2)
            break
        if marker == 0xDA:
            yield (marker, seg_start, seg_start + 2, seg_start + 2)
            break

        if i + 2 > n:
            break

        seglen = struct.unpack(">H", data[i : i + 2])[0]
        payload_start = i + 2
        seg_end = i + seglen
        yield (marker, seg_start, seg_end, payload_start)
        i = seg_end


def _build_app1_xmp_segment(xmp_packet: bytes) -> bytes:
    payload = XMP_ID + xmp_packet
    seg = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    return seg


def _make_updated_xmp_packet(
    existing_xmp_packet: Optional[bytes], *, tool: str, processed_date: str
) -> bytes:
    keyword = f"ProcessedWith:{tool}"
    desc_text = f"Processed by {tool} on {processed_date}"

    root = _parse_or_create_xmpmeta_root(existing_xmp_packet)
    rdf_desc = _get_or_create_rdf_description(root)

    _ensure_dc_subject_keyword(rdf_desc, keyword)
    _ensure_dc_description_xdefault(rdf_desc, desc_text)

    return _serialize_xmpmeta(root)


def write_processed_xmp_embed_jpeg(
    jpeg_path: Union[str, Path],
    *,
    tool: str = DEFAULT_PROCESS_TOOL,
    processed_date: Optional[str] = None,
) -> bool:
    p = Path(jpeg_path)
    if p.suffix.lower() not in (".jpg", ".jpeg"):
        return False

    processed_date = processed_date or _date.today().isoformat()

    try:
        data = p.read_bytes()
    except Exception:
        return False

    if not data.startswith(SOI):
        return False

    existing_xmp = _extract_xmp_from_jpeg_bytes(data)
    new_xmp_packet = _make_updated_xmp_packet(
        existing_xmp, tool=tool, processed_date=processed_date
    )
    new_seg = _build_app1_xmp_segment(new_xmp_packet)

    replace_start = None
    replace_end = None

    insert_at = 2

    for marker, seg_start, seg_end, payload_start in _iter_jpeg_segments_with_bounds(
        data
    ):
        if marker == 0xDA:
            break

        if marker == 0xE1:
            payload = data[payload_start:seg_end]
            if payload.startswith(XMP_ID):
                replace_start = seg_start
                replace_end = seg_end
                break

        if seg_start == insert_at and marker == 0xE0:
            insert_at = seg_end

    if replace_start is not None and replace_end is not None:
        new_data = data[:replace_start] + new_seg + data[replace_end:]
    else:
        new_data = data[:insert_at] + new_seg + data[insert_at:]

    try:
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(new_data)
        tmp.replace(p)
        return True
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


def write_processed_tags(
    image_path: Union[str, Path],
    *,
    tool: str = DEFAULT_PROCESS_TOOL,
    processed_date: Optional[str] = None,
    write_sidecar: bool = True,
    embed_jpeg: bool = True,
) -> bool:
    p = Path(image_path)
    processed_date = processed_date or _date.today().isoformat()

    ok_any = False

    if write_sidecar:
        ok_any = (
            write_processed_xmp_sidecar(p, tool=tool, processed_date=processed_date)
            or ok_any
        )

    if embed_jpeg and p.suffix.lower() in (".jpg", ".jpeg"):
        ok_any = (
            write_processed_xmp_embed_jpeg(p, tool=tool, processed_date=processed_date)
            or ok_any
        )

    return ok_any
