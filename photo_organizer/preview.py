"""Load an analysis preview for a capture: never the full raw unless there's no other way.

Order: the capture's JPG -> HEIC -> the NEF's embedded JPEG (rawpy.extract_thumb)
-> rawpy demosaic (half size) as the last resort. Orientation is applied, and the
result is an RGB uint8 array with its long edge at most `long_edge` pixels.
Read-only on photo files.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HAVE_HEIF = True
except ImportError:          # pragma: no cover
    HAVE_HEIF = False

DEFAULT_LONG_EDGE = 2048
# EXIF orientation -> PIL transpose op (same table ImageOps.exif_transpose uses).
_TRANSPOSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


class PreviewError(RuntimeError):
    pass


@dataclass
class Preview:
    rgb: np.ndarray           # H x W x 3, uint8, orientation applied
    source: str               # jpg | heic | nef_preview | nef_raw
    source_path: str          # relative path of the file decoded


def _shrink(im: Image.Image, long_edge: int) -> Image.Image:
    w, h = im.size
    scale = long_edge / max(w, h)
    if scale < 1:
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                       Image.Resampling.LANCZOS)
    return im


def _draft(im: Image.Image, long_edge: int) -> None:
    """Let the JPEG decoder skip detail we would throw away (DCT scaling)."""
    if im.format == "JPEG":
        w, h = im.size
        scale = long_edge / max(w, h)
        if scale < 1:
            im.draft("RGB", (max(1, int(w * scale)), max(1, int(h * scale))))


def _open_pil(data_or_path, long_edge: int, orientation: int | None = None) -> np.ndarray:
    # Close the file promptly: on Windows an open handle blocks exiftool from
    # replacing the file when labels are written later in the same run.
    with Image.open(data_or_path) as src:
        _draft(src, long_edge)
        im = src
        if orientation is None:
            im = ImageOps.exif_transpose(im)
        elif orientation in _TRANSPOSE:
            im = im.transpose(_TRANSPOSE[orientation])
        im = _shrink(im.convert("RGB"), long_edge)
        return np.asarray(im)


def _nef(path: Path, long_edge: int, orientation: int | None) -> tuple[np.ndarray, str]:
    import rawpy
    with rawpy.imread(str(path)) as raw:
        try:
            thumb = raw.extract_thumb()
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            thumb = None
        if thumb is not None and thumb.format == rawpy.ThumbFormat.JPEG:
            im = Image.open(io.BytesIO(thumb.data))
            if max(im.size) >= min(long_edge, 1024):     # big enough to judge sharpness
                # The embedded JPEG usually carries no orientation; use the NEF's.
                return _open_pil(io.BytesIO(thumb.data), long_edge, orientation or 1), "nef_preview"
        rgb = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
    # rawpy's postprocess already applies the raw's own flip.
    return np.asarray(_shrink(Image.fromarray(rgb), long_edge)), "nef_raw"


def load_preview(root: Path, jpg: str | None, nef: str | None, heic: str | None,
                 nef_orientation: int | None = None,
                 long_edge: int = DEFAULT_LONG_EDGE) -> Preview:
    """Decode the best available source for one capture (paths relative to root)."""
    root = Path(root)
    errors = []
    if jpg:
        try:
            return Preview(_open_pil(root / jpg, long_edge), "jpg", jpg)
        except (OSError, ValueError) as err:
            errors.append(f"{jpg}: {err}")
    if heic and HAVE_HEIF:
        try:
            return Preview(_open_pil(root / heic, long_edge), "heic", heic)
        except (OSError, ValueError) as err:
            errors.append(f"{heic}: {err}")
    if nef:
        try:
            rgb, source = _nef(root / nef, long_edge, nef_orientation)
            return Preview(rgb, source, nef)
        except Exception as err:          # rawpy raises its own LibRaw* error types
            errors.append(f"{nef}: {err}")
    raise PreviewError("; ".join(errors) or "capture has no decodable file")
