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

# -n: numeric values. -charset filename=utf8: non-ASCII paths on Windows.
COMMON_ARGS = ["-n", "-charset", "filename=utf8"]
HASH_ARGS = ["-api", "ImageHashType=SHA256"]


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

    def image_data_hashes(self, paths: list[Path | str]) -> dict[str, str]:
        """SHA-256 of each file's image data, metadata excluded (exiftool ImageDataHash).

        Returns {resolved posix path: hex hash or ''}. '' means exiftool read the
        file but could not hash its image data. Unreadable files are left out.
        """
        if not paths:
            return {}
        keys = [_key(p) for p in paths]
        try:
            rows = self._et.get_tags(keys, ["ImageDataHash"], params=HASH_ARGS)
        except ExifToolException:
            # One bad file can fail a whole batch; fall back to one file at a time.
            if len(keys) == 1:
                return {}
            out: dict[str, str] = {}
            for k in keys:
                out.update(self.image_data_hashes([k]))
            return out
        out = {}
        for row in rows:
            src = row.get("SourceFile")
            if src is None:
                continue
            out[_key(src)] = str(row.get("ImageDataHash") or row.get("File:ImageDataHash") or "")
        return out
