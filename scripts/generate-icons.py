#!/usr/bin/env python3
"""Build the application icon set from the design source sheet.

    python3 scripts/generate-icons.py

Reads `design/app-icon-source.png` — a contact sheet holding the artwork at
several sizes plus dimension labels — lifts the one large icon out of it,
and writes every asset Tauri bundles into `src-tauri/icons/`.

Only the standard library is used. Resizing, PNG encoding and the ICO
container are implemented here; the macOS `.icns` is produced by
`iconutil`, which ships with Xcode's command line tools.

The design sheet is never modified.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_SHEET = REPO_ROOT / "design" / "app-icon-source.png"
ICON_DIR = REPO_ROOT / "src-tauri" / "icons"

MASTER_SIZE = 1024

#: Corner rounding of the finished icon, as a fraction of its side.
#:
#: The large tile on the sheet is drawn almost square-cornered (under 3%),
#: while every smaller rendition on the same sheet sits between 9% and 12%.
#: 12% follows the design's own language and reads as a proper app icon
#: rather than a plain block in the Dock or on the taskbar.
CORNER_RADIUS_FRACTION = 0.12

#: Source pixels ignored around the tile's border when compositing.
EDGE_TRIM = 2

#: Square PNGs written to `src-tauri/icons/<n>x<n>.png`.
PNG_SIZES = (32, 64, 128, 256, 512, 1024)

#: `icon.png` is Tauri's conventional window/Linux icon; 512 is the size
#: its templates ship.
DEFAULT_ICON_SIZE = 512

#: Sizes embedded in the Windows `.ico`.
ICO_SIZES = (16, 32, 48, 64, 128, 256)

#: `.iconset` members `iconutil` expects, as (filename, pixel size). The
#: @2x entries are the Retina renditions, so `icon_512x512@2x` is 1024.
ICONSET_MEMBERS = (
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
)


@dataclass
class Image:
    """8-bit RGBA pixels, row-major, 4 bytes per pixel."""

    width: int
    height: int
    pixels: bytearray

    def pixel(self, x: int, y: int) -> tuple[int, int, int, int]:
        i = (y * self.width + x) * 4
        return (
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        )


# --------------------------------------------------------------------------
# PNG
# --------------------------------------------------------------------------


def read_png(path: Path) -> Image:
    """Decode an 8-bit RGB/RGBA non-interlaced PNG."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\x0a":
        raise ValueError(f"{path} is not a PNG")

    width = height = channels = 0
    idat = bytearray()
    offset = 8
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        kind = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + length]
        if kind == b"IHDR":
            width, height, depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", body
            )
            if depth != 8 or interlace != 0 or color_type not in (2, 6):
                raise ValueError(
                    f"{path}: need an 8-bit non-interlaced RGB/RGBA PNG "
                    f"(got depth={depth} colorType={color_type} "
                    f"interlace={interlace})"
                )
            channels = 3 if color_type == 2 else 4
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
        offset += 12 + length

    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    rows = bytearray(height * stride)
    previous = bytearray(stride)
    pos = 0
    for y in range(height):
        filter_type = raw[pos]
        pos += 1
        line = bytearray(raw[pos : pos + stride])
        pos += stride
        _unfilter(line, previous, filter_type, channels)
        rows[y * stride : (y + 1) * stride] = line
        previous = line

    if channels == 4:
        return Image(width, height, rows)

    # Widen RGB to RGBA so everything downstream sees one layout.
    rgba = bytearray(width * height * 4)
    for i in range(width * height):
        rgba[i * 4 : i * 4 + 3] = rows[i * 3 : i * 3 + 3]
        rgba[i * 4 + 3] = 255
    return Image(width, height, rgba)


def _unfilter(line: bytearray, previous: bytearray, filter_type: int, bpp: int) -> None:
    """Reverse one PNG scanline filter in place (RFC 2083 section 6)."""
    if filter_type == 0:
        return
    if filter_type == 1:
        for i in range(bpp, len(line)):
            line[i] = (line[i] + line[i - bpp]) & 0xFF
    elif filter_type == 2:
        for i in range(len(line)):
            line[i] = (line[i] + previous[i]) & 0xFF
    elif filter_type == 3:
        for i in range(len(line)):
            left = line[i - bpp] if i >= bpp else 0
            line[i] = (line[i] + ((left + previous[i]) >> 1)) & 0xFF
    elif filter_type == 4:
        for i in range(len(line)):
            left = line[i - bpp] if i >= bpp else 0
            up = previous[i]
            up_left = previous[i - bpp] if i >= bpp else 0
            pa = abs(up - up_left)
            pb = abs(left - up_left)
            pc = abs(left + up - 2 * up_left)
            if pa <= pb and pa <= pc:
                predictor = left
            elif pb <= pc:
                predictor = up
            else:
                predictor = up_left
            line[i] = (line[i] + predictor) & 0xFF
    else:
        raise ValueError(f"unknown PNG filter type {filter_type}")


def encode_png(image: Image) -> bytes:
    """Encode RGBA as a PNG.

    One scanline filter is chosen for the whole image by scoring the
    candidates on a handful of sample rows. Icon artwork is uniform enough
    that the winner holds throughout, and picking well matters: on a
    gradient this large the wrong filter costs a few hundred kilobytes.
    """
    stride = image.width * 4
    filter_type = _choose_filter(image, stride)
    raw = bytearray()
    previous = bytearray(stride)
    for y in range(image.height):
        line = image.pixels[y * stride : (y + 1) * stride]
        raw.append(filter_type)
        raw.extend(_filter_scanline(line, previous, filter_type))
        previous = line

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", image.width, image.height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\x0a"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def _filter_scanline(
    line: bytes | bytearray, previous: bytes | bytearray, filter_type: int
) -> bytearray:
    """Apply one PNG scanline filter (the inverse of `_unfilter`)."""
    out = bytearray(len(line))
    bpp = 4
    for i in range(len(line)):
        left = line[i - bpp] if i >= bpp else 0
        up = previous[i]
        if filter_type == 0:
            predictor = 0
        elif filter_type == 1:
            predictor = left
        elif filter_type == 2:
            predictor = up
        elif filter_type == 3:
            predictor = (left + up) >> 1
        else:
            up_left = previous[i - bpp] if i >= bpp else 0
            pa = abs(up - up_left)
            pb = abs(left - up_left)
            pc = abs(left + up - 2 * up_left)
            if pa <= pb and pa <= pc:
                predictor = left
            elif pb <= pc:
                predictor = up
            else:
                predictor = up_left
        out[i] = (line[i] - predictor) & 0xFF
    return out


def _choose_filter(image: Image, stride: int) -> int:
    """Score every filter on sample rows; return the cheapest to compress.

    The score sums each byte's distance from zero, wrapping at 128, which
    is the usual stand-in for how well a scanline will deflate.
    """
    rows = range(1, image.height, max(1, image.height // 8))
    best_type, best_score = 2, None
    for filter_type in range(5):
        score = 0
        for y in rows:
            line = image.pixels[y * stride : (y + 1) * stride]
            previous = image.pixels[(y - 1) * stride : y * stride]
            score += sum(
                byte if byte < 128 else 256 - byte
                for byte in _filter_scanline(line, previous, filter_type)
            )
        if best_score is None or score < best_score:
            best_type, best_score = filter_type, score
    return best_type


def write_png(image: Image, path: Path) -> None:
    path.write_bytes(encode_png(image))


# --------------------------------------------------------------------------
# Finding the icon on the sheet
# --------------------------------------------------------------------------

#: The sheet is transparent between the tiles, so alpha alone separates
#: artwork from background. This tolerance is only a fallback for a sheet
#: flattened onto white: how far from white a pixel must be to count as
#: artwork.
BACKGROUND_TOLERANCE = 60


def locate_primary_icon(sheet: Image) -> tuple[int, int, int, int]:
    """Return `(left, top, width, height)` of the largest icon on the sheet.

    The sheet is empty between the tiles, so a flood fill inward from the
    border marks the background and leaves each tile, preview and text
    label as a separate island. The biggest island is the icon we want;
    no label comes close in area. Filling from the border rather than
    thresholding also keeps the icon's own bright details — the film
    frame, the play symbol, the subtitle bar — because they are walled in
    by the tile's opaque body, which the fill cannot cross.
    """
    width, height, pixels = sheet.width, sheet.height, sheet.pixels

    is_background = bytearray(width * height)
    queue: deque[tuple[int, int]] = deque()

    def background_like(x: int, y: int) -> bool:
        i = (y * width + x) * 4
        if pixels[i + 3] < 128:  # transparent sheet margin
            return True
        return all(255 - pixels[i + c] <= BACKGROUND_TOLERANCE for c in range(3))

    for x in range(width):
        for y in (0, height - 1):
            if not is_background[y * width + x] and background_like(x, y):
                is_background[y * width + x] = 1
                queue.append((x, y))
    for y in range(height):
        for x in (0, width - 1):
            if not is_background[y * width + x] and background_like(x, y):
                is_background[y * width + x] = 1
                queue.append((x, y))

    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height:
                index = ny * width + nx
                if not is_background[index] and background_like(nx, ny):
                    is_background[index] = 1
                    queue.append((nx, ny))

    best_area = 0
    best_box = None
    visited = bytearray(width * height)
    for start in range(width * height):
        if is_background[start] or visited[start]:
            continue
        visited[start] = 1
        island = deque([start])
        area = 0
        x0 = x1 = start % width
        y0 = y1 = start // width
        while island:
            index = island.popleft()
            area += 1
            x, y = index % width, index // width
            x0, x1 = min(x0, x), max(x1, x)
            y0, y1 = min(y0, y), max(y1, y)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < width and 0 <= ny < height:
                    neighbour = ny * width + nx
                    if not is_background[neighbour] and not visited[neighbour]:
                        visited[neighbour] = 1
                        island.append(neighbour)
        if area > best_area:
            best_area = area
            best_box = (x0, y0, x1, y1)

    if best_box is None:
        raise RuntimeError(f"found no artwork on {SOURCE_SHEET}")

    x0, y0, x1, y1 = best_box
    box_width, box_height = x1 - x0 + 1, y1 - y0 + 1
    print(f"  primary icon at ({x0},{y0})–({x1},{y1}), {box_width}×{box_height}px")
    return x0, y0, box_width, box_height


def measure_corner_radius(sheet: Image, left: int, top: int, height: int) -> float:
    """Estimate the tile's own corner radius, in source pixels.

    Walk down its left edge: on a rounded rectangle the first opaque pixel
    of each row moves inward near the corner and settles once the straight
    edge begins, so the row where it settles is the radius.
    """
    for dy in range(height // 2):
        y = top + dy
        for dx in range(height // 2):
            if sheet.pixel(left + dx, y)[3] >= 128:
                if dx <= 1:
                    return float(dy)
                break
    return 0.0


# --------------------------------------------------------------------------
# Master icon
# --------------------------------------------------------------------------


def build_master(sheet: Image) -> Image:
    """Lift the icon off the sheet, square it up, scale to the master size."""
    left, top, width, height = locate_primary_icon(sheet)
    square = _square_up(sheet, left, top, width, height)
    if square.width == MASTER_SIZE:
        return square
    print(f"  scaling {square.width} → {MASTER_SIZE}")
    return resize(square, MASTER_SIZE)


def _square_up(sheet: Image, left: int, top: int, width: int, height: int) -> Image:
    """Place the tile on a square canvas and cut its rounded corners.

    Every tile on the sheet was drawn a little wider than it is tall, and
    an app icon has to be square. Cropping to the short side would shave
    the side margins to a third of the top and bottom ones, so the canvas
    grows instead: the tile's own background continues past its short
    edges, and the artwork keeps the proportions it was drawn with.

    The sheet is transparent behind each tile, so compositing over that
    extended background with the tile's own alpha reproduces its
    antialiased edge exactly, with nothing pale left to fringe it.
    """
    side = max(width, height)
    offset_x = (side - width) // 2
    offset_y = (side - height) // 2
    canvas = bytearray(_background_fill(sheet, left, top, width, height, side))

    # Composite through the tile's own outline, pulled in by a couple of
    # pixels. Its outermost ring is far darker than its body — an edge that
    # made sense against the sheet, but which would read as a stray line
    # and as flecks in the corners once the canvas extends past it. Below
    # that ring the background fill continues the gradient instead.
    tile_radius = measure_corner_radius(sheet, left, top, height)
    outline_radius = max(0.0, tile_radius - EDGE_TRIM)
    right = width - EDGE_TRIM
    bottom = height - EDGE_TRIM
    pixels = sheet.pixels
    for y in range(EDGE_TRIM, height - EDGE_TRIM):
        row = (top + y) * sheet.width + left
        target_row = (y + offset_y) * side + offset_x
        for x in range(EDGE_TRIM, width - EDGE_TRIM):
            if not _inside_rounded_rect(
                x + 0.5, y + 0.5, EDGE_TRIM, right, bottom, outline_radius
            ):
                continue
            source = (row + x) * 4
            alpha = pixels[source + 3]
            if alpha == 0:
                continue
            target = (target_row + x) * 4
            if alpha == 255:
                canvas[target : target + 3] = pixels[source : source + 3]
                continue
            weight = alpha / 255.0
            for channel in range(3):
                under = canvas[target + channel]
                canvas[target + channel] = int(
                    under + (pixels[source + channel] - under) * weight + 0.5
                )

    square = Image(side, side, canvas)
    radius = side * CORNER_RADIUS_FRACTION
    print(f"  rounding corners at {radius:.0f}px ({CORNER_RADIUS_FRACTION:.0%})")
    _apply_rounded_mask(square, inset=0.0, radius=radius)
    return square


def _inside_rounded_rect(
    px: float, py: float, low: float, right: float, bottom: float, radius: float
) -> bool:
    """Whether the point sits inside a rounded rectangle.

    A plain inside/outside test is enough here: it decides between the tile
    and a background fill sampled from the tile itself, so there is nothing
    for antialiasing to smooth between.
    """
    if not (low <= px <= right and low <= py <= bottom):
        return False
    near_left = px < low + radius
    near_top = py < low + radius
    if (near_left or px > right - radius) and (near_top or py > bottom - radius):
        cx = low + radius if near_left else right - radius
        cy = low + radius if near_top else bottom - radius
        return (px - cx) ** 2 + (py - cy) ** 2 <= radius * radius
    return True


def _background_fill(
    sheet: Image, left: int, top: int, width: int, height: int, side: int
) -> bytearray:
    """An opaque square of the tile's background, extended to `side`.

    The background is a soft diagonal gradient, so each row is a ramp
    between the colours sampled just inside the tile's left and right
    margins. Rows beyond the tile repeat its nearest *interior* row —
    sampling the very edge would pick up the tile's dark outline and paint
    the extension almost black.
    """
    probe_inset = max(2, int(width * 0.04))
    canvas = bytearray(side * side * 4)
    offset_y = (side - height) // 2
    lowest = probe_inset
    highest = height - 1 - probe_inset
    for y in range(side):
        source_y = top + min(max(y - offset_y, lowest), highest)
        left_rgb = _opaque_sample(sheet, left + probe_inset, source_y, +1)
        right_rgb = _opaque_sample(sheet, left + width - 1 - probe_inset, source_y, -1)
        base = y * side * 4
        for x in range(side):
            t = x / (side - 1)
            index = base + x * 4
            for channel in range(3):
                canvas[index + channel] = int(
                    left_rgb[channel]
                    + (right_rgb[channel] - left_rgb[channel]) * t
                    + 0.5
                )
            canvas[index + 3] = 255
    return canvas


def _opaque_sample(sheet: Image, x: int, y: int, step: int) -> tuple[int, int, int]:
    """Colour at (x, y), walking `step` pixels inward until opaque.

    Guards the corner rows, where the probe column can fall outside the
    tile's rounded outline and pick up transparent pixels.
    """
    for _ in range(sheet.width):
        r, g, b, a = sheet.pixel(x, y)
        if a >= 200:
            return r, g, b
        x += step
        if not 0 <= x < sheet.width:
            break
    return 0, 0, 0


def _apply_rounded_mask(image: Image, inset: float, radius: float) -> None:
    """Multiply alpha by an antialiased rounded-rectangle mask, in place."""
    size = image.width
    low = inset
    high = size - inset
    radius = max(0.0, radius)
    samples = 4  # 4×4 per pixel: plenty for a smooth edge at this scale
    offsets = [(i + 0.5) / samples for i in range(samples)]

    for y in range(size):
        for x in range(size):
            coverage = _corner_coverage(x, y, low, high, radius, offsets, samples)
            if coverage >= 1.0:
                continue
            index = (y * size + x) * 4
            if coverage <= 0.0:
                image.pixels[index + 3] = 0
            else:
                image.pixels[index + 3] = int(
                    image.pixels[index + 3] * coverage + 0.5
                )


def _corner_coverage(
    x: int,
    y: int,
    low: float,
    high: float,
    radius: float,
    offsets: list[float],
    samples: int,
) -> float:
    """Fraction of pixel (x, y) covered by the rounded square spanning
    `low`…`high` on both axes."""
    # Whole pixels clear of all four arcs are the common case: the
    # cross-shaped region through the middle.
    spans_x = low + radius <= x and x + 1 <= high - radius
    spans_y = low + radius <= y and y + 1 <= high - radius
    within_x = low <= x and x + 1 <= high
    within_y = low <= y and y + 1 <= high
    if (spans_x and within_y) or (within_x and spans_y):
        return 1.0

    inside = 0
    for oy in offsets:
        py = y + oy
        for ox in offsets:
            px = x + ox
            if not (low <= px <= high and low <= py <= high):
                continue
            near_left = px < low + radius
            near_top = py < low + radius
            if (near_left or px > high - radius) and (
                near_top or py > high - radius
            ):
                cx = low + radius if near_left else high - radius
                cy = low + radius if near_top else high - radius
                if (px - cx) ** 2 + (py - cy) ** 2 <= radius * radius:
                    inside += 1
            else:
                inside += 1
    return inside / (samples * samples)


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------


def _mitchell(x: float) -> float:
    """Mitchell–Netravali kernel (B = C = 1/3).

    Sharper than a triangle filter without the ringing a Catmull-Rom would
    put around this artwork's hard white-on-navy edges.
    """
    x = abs(x)
    if x < 1.0:
        return (7.0 * x * x * x - 12.0 * x * x + 16.0 / 3.0) / 6.0
    if x < 2.0:
        return (
            -7.0 / 3.0 * x * x * x + 12.0 * x * x - 20.0 * x + 32.0 / 3.0
        ) / 6.0
    return 0.0


def resize(image: Image, target: int) -> Image:
    """Resample to `target`×`target`, separably, in premultiplied alpha.

    Premultiplying matters at the transparent corners: interpolating raw
    colour against fully transparent pixels would drag their meaningless
    RGB into the edge and leave a visible halo.
    """
    premultiplied = _premultiply(image)
    horizontal = _resample_axis(premultiplied, image.width, image.height, target)
    vertical = _resample_axis_vertical(horizontal, target, image.height, target)
    return _unpremultiply(vertical, target, target)


def _premultiply(image: Image) -> list[float]:
    out = [0.0] * (image.width * image.height * 4)
    pixels = image.pixels
    for i in range(image.width * image.height):
        alpha = pixels[i * 4 + 3] / 255.0
        out[i * 4] = pixels[i * 4] * alpha
        out[i * 4 + 1] = pixels[i * 4 + 1] * alpha
        out[i * 4 + 2] = pixels[i * 4 + 2] * alpha
        out[i * 4 + 3] = pixels[i * 4 + 3]
    return out


def _unpremultiply(data: list[float], width: int, height: int) -> Image:
    out = bytearray(width * height * 4)
    for i in range(width * height):
        alpha = data[i * 4 + 3]
        alpha = 0.0 if alpha < 0.0 else (255.0 if alpha > 255.0 else alpha)
        out[i * 4 + 3] = int(alpha + 0.5)
        if alpha <= 0.0:
            continue
        scale = 255.0 / alpha
        for c in range(3):
            value = data[i * 4 + c] * scale
            out[i * 4 + c] = 0 if value < 0 else (255 if value > 255 else int(value + 0.5))
    return Image(width, height, out)


def _weights(source: int, target: int) -> list[tuple[int, list[float]]]:
    """Per-output-pixel `(first_input_index, weights)`.

    When shrinking, the kernel widens to average whole neighbourhoods,
    which is what keeps the 32px rendition free of aliasing.
    """
    scale = source / target
    filter_scale = max(1.0, scale)
    support = 2.0 * filter_scale
    plan = []
    for i in range(target):
        center = (i + 0.5) * scale
        first = max(0, int(center - support + 0.5))
        last = min(source, int(center + support + 0.5))
        raw = [_mitchell((x + 0.5 - center) / filter_scale) for x in range(first, last)]
        total = sum(raw)
        if total == 0:
            raw, total = [1.0], 1.0
            first, last = min(int(center), source - 1), min(int(center), source - 1) + 1
        plan.append((first, [w / total for w in raw]))
    return plan


def _resample_axis(
    data: list[float], width: int, height: int, target: int
) -> list[float]:
    """Horizontal pass: width → target, height unchanged."""
    plan = _weights(width, target)
    out = [0.0] * (target * height * 4)
    for y in range(height):
        row = y * width * 4
        out_row = y * target * 4
        for i, (first, weights) in enumerate(plan):
            base = row + first * 4
            r = g = b = a = 0.0
            for k, weight in enumerate(weights):
                j = base + k * 4
                r += data[j] * weight
                g += data[j + 1] * weight
                b += data[j + 2] * weight
                a += data[j + 3] * weight
            o = out_row + i * 4
            out[o] = r
            out[o + 1] = g
            out[o + 2] = b
            out[o + 3] = a
    return out


def _resample_axis_vertical(
    data: list[float], width: int, height: int, target: int
) -> list[float]:
    """Vertical pass: height → target, width unchanged."""
    plan = _weights(height, target)
    out = [0.0] * (width * target * 4)
    for i, (first, weights) in enumerate(plan):
        out_row = i * width * 4
        for x in range(width):
            r = g = b = a = 0.0
            for k, weight in enumerate(weights):
                j = ((first + k) * width + x) * 4
                r += data[j] * weight
                g += data[j + 1] * weight
                b += data[j + 2] * weight
                a += data[j + 3] * weight
            o = out_row + x * 4
            out[o] = r
            out[o + 1] = g
            out[o + 2] = b
            out[o + 3] = a
    return out


# --------------------------------------------------------------------------
# Platform containers
# --------------------------------------------------------------------------


def write_ico(renditions: dict[int, Image], path: Path) -> None:
    """Write a real ICO holding PNG-compressed images.

    PNG entries are the modern form of the format (Vista onward) and keep
    the file small; the alternative is an uncompressed DIB per size.
    """
    entries = [(size, encode_png(renditions[size])) for size in sorted(ICO_SIZES)]
    header = struct.pack("<HHH", 0, 1, len(entries))
    offset = len(header) + 16 * len(entries)
    directory = bytearray()
    for size, blob in entries:
        directory += struct.pack(
            "<BBBBHHII",
            0 if size >= 256 else size,  # 0 means 256 in this field
            0 if size >= 256 else size,
            0,  # palette size: none, it's a true-colour image
            0,
            1,  # colour planes
            32,  # bits per pixel
            len(blob),
            offset,
        )
        offset += len(blob)
    path.write_bytes(header + bytes(directory) + b"".join(blob for _, blob in entries))


def write_icns(renditions: dict[int, Image], path: Path) -> None:
    """Build a real ICNS via `iconutil`, from a temporary `.iconset`."""
    iconutil = shutil.which("iconutil")
    if not iconutil:
        raise RuntimeError(
            "iconutil not found — it comes with the Xcode command line "
            "tools (xcode-select --install). Skipping would leave a stale "
            "or missing icon.icns, so refusing to continue."
        )
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "AppIcon.iconset"
        iconset.mkdir()
        for name, size in ICONSET_MEMBERS:
            write_png(renditions[size], iconset / name)
        subprocess.run(
            [iconutil, "-c", "icns", str(iconset), "-o", str(path)],
            check=True,
            capture_output=True,
        )


# --------------------------------------------------------------------------


def main() -> int:
    if not SOURCE_SHEET.exists():
        print(f"missing design source: {SOURCE_SHEET}", file=sys.stderr)
        return 1

    print(f"reading {SOURCE_SHEET.relative_to(REPO_ROOT)}")
    sheet = read_png(SOURCE_SHEET)
    print(f"  sheet is {sheet.width}×{sheet.height}")

    master = build_master(sheet)
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    write_png(master, ICON_DIR / "icon-master.png")
    print(f"  wrote icon-master.png ({master.width}×{master.height})")

    needed = sorted({*PNG_SIZES, *ICO_SIZES, DEFAULT_ICON_SIZE, MASTER_SIZE})
    renditions: dict[int, Image] = {}
    for size in needed:
        renditions[size] = master if size == MASTER_SIZE else resize(master, size)
        print(f"  rendered {size}×{size}")

    for size in PNG_SIZES:
        write_png(renditions[size], ICON_DIR / f"{size}x{size}.png")
    write_png(renditions[DEFAULT_ICON_SIZE], ICON_DIR / "icon.png")
    print(f"  wrote {len(PNG_SIZES)} sized PNGs and icon.png")

    write_ico(renditions, ICON_DIR / "icon.ico")
    print(f"  wrote icon.ico ({', '.join(str(s) for s in sorted(ICO_SIZES))})")

    write_icns(renditions, ICON_DIR / "icon.icns")
    print("  wrote icon.icns")

    print(f"\nicons are in {ICON_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
