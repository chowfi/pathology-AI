"""Read Aperio .svs structure over HTTP Range and copy one pyramid level into a single-level tiled TIFF.

Why: the full .svs files total about 650 GB. We need only the 4x level (0.41533 um/px, rescaled later to 0.5).
In these files the tiles of one level are stored back to back, so one contiguous Range read gets the whole level.

The JPEG tiles are copied byte for byte (no decode, no re-encode, no quality loss). The output is a standard
BigTIFF with one tiled image, readable by tifffile / pyvips / libtiff.

Privacy: we read only the TIFF directory tags of every image in the file. We never read the pixel data of the
label or macro images (they can show the paper label of the glass slide).
"""
from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import requests

ReadFn = Callable[[int, int], bytes]          # read(offset, n_bytes) -> bytes

TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8, 17: 8}
TYPE_FMT = {1: "B", 3: "H", 4: "I", 16: "Q", 8: "h", 9: "i", 17: "q"}
T_WIDTH, T_LENGTH, T_DESC, T_TILE_W, T_TILE_H, T_TILE_OFFSETS, T_TILE_COUNTS = 256, 257, 270, 322, 323, 324, 325
UA = {"User-Agent": "Mozilla/5.0"}


# --------------------------------------------------------------------------------------------- HTTP
class RangeReader:
    """Callable read(offset, n) over HTTP Range with retries and a keep-alive session."""

    def __init__(self, url: str, tries: int = 8, timeout: float = 120):
        self.url, self.tries, self.timeout = url, tries, timeout
        self.session = requests.Session()
        self.session.headers.update(UA)

    def __call__(self, offset: int, n: int) -> bytes:
        last: Exception | None = None
        for k in range(self.tries):
            try:
                r = self.session.get(self.url, headers={"Range": f"bytes={offset}-{offset + n - 1}"},
                                     timeout=self.timeout)
                if r.status_code == 404:
                    raise FileNotFoundError(self.url)
                if r.status_code != 206:
                    raise ConnectionError(f"HTTP {r.status_code} (expected 206 Partial Content)")
                if len(r.content) != n:
                    raise ConnectionError(f"short read {len(r.content)} of {n} bytes")
                return r.content
            except FileNotFoundError:
                raise
            except Exception as e:  # network errors, 5xx, short reads -> back off and retry
                last = e
                time.sleep(min(60, 2 * 2 ** k))
        raise ConnectionError(f"{self.url} [{offset}+{n}] failed after {self.tries} tries: {last}")


def file_reader(path: str | Path) -> ReadFn:
    """read(offset, n) for a local file (used in tests and for verifying written files)."""
    def read(offset: int, n: int) -> bytes:
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read(n)
    return read


# --------------------------------------------------------------------------------------------- parse
@dataclass
class Tag:
    code: int
    type: int
    count: int
    raw: bytes                                   # the full value bytes (fetched if stored out of line)

    def values(self, endian: str = "<") -> tuple:
        if self.type == 2:
            return (self.raw.rstrip(b"\x00").decode("latin1"),)
        if self.type in (5, 10):
            f = "I" if self.type == 5 else "i"
            v = struct.unpack(f"{endian}{2 * self.count}{f}", self.raw)
            return tuple(zip(v[::2], v[1::2]))
        if self.type in TYPE_FMT:
            return struct.unpack(f"{endian}{self.count}{TYPE_FMT[self.type]}", self.raw)
        return (self.raw,)


@dataclass
class Ifd:
    index: int
    tags: dict[int, Tag] = field(default_factory=dict)
    endian: str = "<"

    def get(self, code: int) -> tuple:
        return self.tags[code].values(self.endian)

    @property
    def tiled(self) -> bool:
        return T_TILE_W in self.tags

    @property
    def width(self) -> int:
        return self.get(T_WIDTH)[0]

    @property
    def height(self) -> int:
        return self.get(T_LENGTH)[0]


def parse_ifds(read: ReadFn, max_inline_fetch: int = 64 << 20) -> list[Ifd]:
    """Parse every image directory of a TIFF/BigTIFF. Reads tag data only, never pixel data."""
    head = read(0, 16)
    e = "<" if head[:2] == b"II" else ">"
    big = struct.unpack(e + "H", head[2:4])[0] == 43
    off = struct.unpack(e + "Q", head[8:16])[0] if big else struct.unpack(e + "I", head[4:8])[0]
    hdr, ent, cnt_fmt, ptr_fmt, inline = (8, 20, "Q", "Q", 8) if big else (2, 12, "H", "I", 4)
    out: list[Ifd] = []
    while off:
        n = struct.unpack(e + cnt_fmt, read(off, hdr))[0]
        block = read(off, hdr + ent * n + inline)
        ifd = Ifd(index=len(out), endian=e)
        for i in range(n):
            x = block[hdr + i * ent: hdr + (i + 1) * ent]
            code, typ = struct.unpack(e + "HH", x[:4])
            count = struct.unpack(e + ("Q" if big else "I"), x[4:12] if big else x[4:8])[0]
            val = x[12:20] if big else x[8:12]
            nbytes = count * TYPE_SIZE.get(typ, 1)
            if nbytes <= inline:
                raw = val[:nbytes]
            elif nbytes <= max_inline_fetch:
                raw = read(struct.unpack(e + ptr_fmt, val)[0], nbytes)
            else:
                raise ValueError(f"tag {code} too large ({nbytes} bytes)")
            ifd.tags[code] = Tag(code, typ, count, raw)
        out.append(ifd)
        off = struct.unpack(e + ptr_fmt, block[hdr + ent * n: hdr + ent * n + inline])[0]
    return out


def find_level(ifds: list[Ifd], downsample: float, tol: float = 0.05) -> Ifd:
    """The tiled pyramid level whose width is base_width / downsample (base = first tiled image)."""
    tiled = [d for d in ifds if d.tiled]
    base = tiled[0].width
    for d in tiled:
        if abs(base / d.width - downsample) < tol * downsample:
            return d
    raise LookupError(f"no level with downsample {downsample}; have {[round(base / d.width, 2) for d in tiled]}")


def tile_span(ifd: Ifd) -> tuple[int, int]:
    """(first byte, end byte) of the level's tile data. Raises if the tiles are not stored back to back."""
    offs, cnts = ifd.get(T_TILE_OFFSETS), ifd.get(T_TILE_COUNTS)
    pairs = sorted(zip(offs, cnts))
    for (o1, c1), (o2, _) in zip(pairs, pairs[1:]):
        if o1 + c1 != o2:
            raise ValueError(f"tiles not contiguous at offset {o1 + c1} (next tile at {o2})")
    return pairs[0][0], pairs[-1][0] + pairs[-1][1]


# --------------------------------------------------------------------------------------------- write
# Tags copied from the source level: everything that describes how to decode the tiles.
COPY_TAGS = (258, 259, 262, 277, 284, 317, 338, 339, 347, 529, 530, 531, 532)
DATA_START = 16                                   # BigTIFF header size; tile bytes start right after it


def _entry(code: int, typ: int, count: int, raw: bytes) -> tuple[int, int, int, bytes]:
    return code, typ, count, raw


def build_ifd(src: Ifd, delta: int, mpp: float, description: str, ifd_offset: int) -> bytes:
    """BigTIFF IFD bytes (plus out-of-line values) for one image whose tiles were shifted by `delta` bytes."""
    e = src.endian
    if e != "<":
        raise NotImplementedError("big-endian source")
    offs = [o + delta for o in src.get(T_TILE_OFFSETS)]
    cnts = list(src.get(T_TILE_COUNTS))
    px_per_cm = round(1e4 / mpp * 1000)           # RATIONAL with denominator 1000
    desc = description.encode("ascii") + b"\x00"
    entries = [
        _entry(256, 16, 1, struct.pack("<Q", src.width)),
        _entry(257, 16, 1, struct.pack("<Q", src.height)),
        _entry(270, 2, len(desc), desc),
        _entry(282, 5, 1, struct.pack("<II", px_per_cm, 1000)),
        _entry(283, 5, 1, struct.pack("<II", px_per_cm, 1000)),
        _entry(296, 3, 1, struct.pack("<H", 3)),          # resolution unit: centimetre
        _entry(322, 16, 1, struct.pack("<Q", src.get(T_TILE_W)[0])),
        _entry(323, 16, 1, struct.pack("<Q", src.get(T_TILE_H)[0])),
        _entry(324, 16, len(offs), struct.pack(f"<{len(offs)}Q", *offs)),
        _entry(325, 16, len(cnts), struct.pack(f"<{len(cnts)}Q", *cnts)),
    ]
    entries += [_entry(c, src.tags[c].type, src.tags[c].count, src.tags[c].raw) for c in COPY_TAGS if c in src.tags]
    entries.sort(key=lambda t: t[0])

    n = len(entries)
    table_size = 8 + 20 * n + 8
    extra = bytearray()
    table = bytearray(struct.pack("<Q", n))
    for code, typ, count, raw in entries:
        if len(raw) <= 8:
            val = raw.ljust(8, b"\x00")
        else:
            pos = ifd_offset + table_size + len(extra)
            val = struct.pack("<Q", pos)
            extra += raw + (b"\x00" if len(raw) % 2 else b"")   # keep word alignment
        table += struct.pack("<HHQ", code, typ, count) + val
    table += struct.pack("<Q", 0)                                # no next IFD
    return bytes(table) + bytes(extra)


def bigtiff_header(first_ifd: int) -> bytes:
    return b"II" + struct.pack("<HHHQ", 43, 8, 0, first_ifd)


def check_jpeg_tiles(read: ReadFn, offsets: tuple, counts: tuple, delta: int = 0) -> int:
    """Every tile must start with a JPEG SOI marker and contain an EOI marker near its end. Returns bad count."""
    bad = 0
    for o, c in zip(offsets, counts):
        if c == 0:
            bad += 1
            continue
        b = read(o + delta, 2)
        tail = read(o + delta + max(0, c - 16), min(16, c))
        if b != b"\xff\xd8" or b"\xff\xd9" not in tail:
            bad += 1
    return bad
