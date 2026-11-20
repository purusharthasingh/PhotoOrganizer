"""exiftool access through pyexiftool, kept running in -stay_open mode.

Usage:
    with Exif() as exif:
        hashes = exif.image_data_hashes([path1, path2])
"""

from __future__ import annotations

import shutil
from pathlib import Path

import exiftool
from exiftool.exceptions import ExifToolException

# -charset filename=utf8: non-ASCII paths on Windows. Numeric (-n) output is chosen per call.
COMMON_ARGS = ["-charset", "filename=utf8"]
HASH_ARGS = ["-api", "ImageHashType=SHA256"]
RAW_SUFFIXES = {".nef", ".nrw", ".cr2", ".cr3", ".arw", ".raf", ".orf", ".rw2", ".dng"}


class ExifError(RuntimeError):
    pass


def _key(path: Path | str) -> str:
    """How we name a file to exiftool, and how it names it back in SourceFile."""
    return Path(path).resolve().as_posix()


class Exif:
    def __init__(self, executable: str | None = None):
        exe = executable or shutil.which("exiftool")
        if exe is None:
            raise ExifError("exiftool not found on PATH (see README -> Setup -> exiftool)")
        self._et = exiftool.ExifToolHelper(
            executable=exe, common_args=COMMON_ARGS, encoding="utf-8", check_execute=False)

    def __enter__(self) -> "Exif":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._et.running:
            self._et.terminate()

    def _rows(self, keys: list[str], tags: list[str], params: list[str]) -> dict[str, dict]:
        """{resolved posix path: exiftool row}. Unreadable files are left out."""
        if not keys:
            return {}
        try:
            rows = self._et.get_tags(keys, tags, params=params)
        except ExifToolException:
            # One bad file can fail a whole batch; fall back to one file at a time.
            if len(keys) == 1:
                return {}
            out: dict[str, dict] = {}
            for k in keys:
                out.update(self._rows([k], tags, params))
            return out
        return {_key(r["SourceFile"]): r for r in rows if "SourceFile" in r}

    def image_data_hashes(self, paths: list[Path | str]) -> dict[str, str]:
        """SHA-256 of each file's image data, metadata excluded (exiftool ImageDataHash).

        Returns {resolved posix path: hex hash or ''}. '' means exiftool read the
        file but could not hash its image data. Unreadable files are left out.
        """
        rows = self._rows([_key(p) for p in paths], ["ImageDataHash"], HASH_ARGS)
        return {k: str(r.get("ImageDataHash") or "") for k, r in rows.items()}

    def write(self, path: Path | str, args: list[str]) -> None:
        """Write tags to one file in place (exiftool -overwrite_original).

        Writing to a .xmp path that doesn't exist creates that sidecar. Refuses raw
        files: raws are read-only in this project.
        """
        p = Path(path)
        if p.suffix.lower() in RAW_SUFFIXES:
            raise ExifError(f"refusing to write to a raw file: {p}")
        self._et.execute("-overwrite_original", *args, _key(p))
        status = self._et.last_status
        stderr = (self._et.last_stderr or "").strip()
        if status != 0 or "Error" in stderr:
            raise ExifError(f"exiftool failed on {p}: {stderr or f'status {status}'}")

    def read_tags(self, paths: list[Path | str], tags: list[str],
                  numeric: bool = True) -> dict[str, dict]:
        """Read the given tags. Returns {resolved posix path: {tag: value}}, keys
        without group prefixes (so request XMP tags by group to disambiguate).

        numeric=True gives raw values (-n); False gives exiftool's readable form,
        e.g. PrimaryAFPoint 'C6 (Center)' instead of 1."""
        return self._rows([_key(p) for p in paths], tags, ["-n"] if numeric else [])
