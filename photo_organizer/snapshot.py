"""Stage 0: snapshot the library's fingerprints, and verify a library or backup against one.

A snapshot is a TSV in <root>/.photoorganizer/snapshots/ listing every scanned
file with two fingerprints:
  sha256      whole-file hash. Any change to the file changes it.
  image_hash  exiftool ImageDataHash: SHA-256 of the picture data only. It stays
              the same when metadata such as pipeline labels is written.

`verify` re-reads every file and sorts it into ok / metadata_only / changed /
missing / unreadable, plus files present but not in the snapshot (extra).
"""

from __future__ import annotations

import csv
import hashlib
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import manifest
from .config import ScanConfig
from .exif_io import Exif
from .scan import ScanResult, scan, walk

SNAPSHOT_DIRNAME = "snapshots"
COLUMNS = ["path", "kind", "size", "mtime_ns", "sha256", "image_hash"]
PICTURE_KINDS = ("nef", "jpg", "heic")      # kinds that have image data
METADATA_WRITABLE = ("jpg", "heic")         # kinds the pipeline may write XMP into
EXIF_BATCH = 32
CHUNK = 1 << 20

Progress = Callable[[str], None]


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


class _Meter:
    """Prints progress at most every `every` seconds."""

    def __init__(self, label: str, total_files: int, total_bytes: int, out: Progress, every=2.0):
        self.label, self.total_files, self.total_bytes = label, total_files, total_bytes
        self.out, self.every = out, every
        self.files = self.bytes = 0
        self.start = self.last = time.monotonic()

    def add(self, nbytes: int) -> None:
        self.files += 1
        self.bytes += nbytes
        now = time.monotonic()
        if now - self.last >= self.every:
            self.last = now
            self.out(self._line(now))

    def done(self) -> None:
        if self.files:
            self.out(self._line(time.monotonic()))

    def _line(self, now: float) -> str:
        gb = 1024 ** 3
        rate = self.bytes / max(now - self.start, 1e-6) / 1024 ** 2
        return (f"  {self.label}: {self.files}/{self.total_files} files, "
                f"{self.bytes / gb:.2f}/{self.total_bytes / gb:.2f} GB, {rate:.0f} MB/s")


# --------------------------------------------------------------------------- snapshot

@dataclass
class HashStats:
    hashed_sha: int = 0
    hashed_image: int = 0
    no_image_hash: list[str] = field(default_factory=list)   # picture files exiftool couldn't hash
    unstable: list[str] = field(default_factory=list)         # changed while being hashed
    errors: list[str] = field(default_factory=list)


@dataclass
class SnapshotResult(HashStats):
    scan: ScanResult | None = None
    path: Path | None = None
    files: int = 0


def _stat_matches(full: Path, size: int, mtime_ns: int) -> bool:
    try:
        st = os.stat(full)
    except OSError:
        return False
    return st.st_size == size and st.st_mtime_ns == mtime_ns


def fill_hashes(root: Path, conn: sqlite3.Connection, exif: Exif, out: Progress,
                result: HashStats | None = None, *, sha: bool = True,
                image: bool = True) -> HashStats:
    """Compute missing sha256 / image_hash values in `files`. Cached ones are reused:
    Stage 1 clears them whenever a file's size or mtime changes."""
    result = result if result is not None else HashStats()
    rows = conn.execute(
        "SELECT path, kind, size, mtime_ns, sha256, image_hash FROM files WHERE present=1"
        " ORDER BY path").fetchall()

    need_sha = [r for r in rows if r["sha256"] is None] if sha else []
    if need_sha:
        meter = _Meter("sha256", len(need_sha), sum(r["size"] for r in need_sha), out)
        for r in need_sha:
            full = root / r["path"]
            try:
                digest = sha256_file(full)
            except OSError as err:
                result.errors.append(f"{r['path']}: {err.strerror}")
                continue
            meter.add(r["size"])
            if not _stat_matches(full, r["size"], r["mtime_ns"]):
                result.unstable.append(r["path"])
                continue
            with conn:
                conn.execute("UPDATE files SET sha256=? WHERE path=?", (digest, r["path"]))
            result.hashed_sha += 1
        meter.done()

    need_img = [r for r in rows
                if r["kind"] in PICTURE_KINDS and r["image_hash"] is None] if image else []
    if need_img:
        meter = _Meter("image hash", len(need_img), sum(r["size"] for r in need_img), out)
        for i in range(0, len(need_img), EXIF_BATCH):
            batch = need_img[i:i + EXIF_BATCH]
            hashes = exif.image_data_hashes([root / r["path"] for r in batch])
            with conn:
                for r in batch:
                    full = root / r["path"]
                    meter.add(r["size"])
                    value = hashes.get(full.resolve().as_posix())
                    if value is None:
                        result.errors.append(f"{r['path']}: exiftool could not read the file")
                        continue
                    if not _stat_matches(full, r["size"], r["mtime_ns"]):
                        result.unstable.append(r["path"])
                        continue
                    # '' = readable but no hashable image data; cached so we don't retry.
                    conn.execute("UPDATE files SET image_hash=? WHERE path=?", (value, r["path"]))
                    result.hashed_image += 1
        meter.done()

    result.no_image_hash = [
        r["path"] for r in conn.execute(
            "SELECT path FROM files WHERE present=1 AND image_hash='' ORDER BY path")]
    return result


def snapshot_dir(root: Path) -> Path:
    return Path(root) / manifest.STATE_DIRNAME / SNAPSHOT_DIRNAME


def take_snapshot(root: Path, cfg: ScanConfig, conn: sqlite3.Connection, exif: Exif,
                  out: Progress = print) -> SnapshotResult:
    """Scan, fill in fingerprints, and write <root>/.photoorganizer/snapshots/<UTC time>.tsv."""
    root = Path(root).resolve()
    result = SnapshotResult(scan=scan(root, cfg, conn))
    fill_hashes(root, conn, exif, out, result)

    rows = conn.execute(
        "SELECT path, kind, size, mtime_ns, sha256, image_hash FROM files"
        " WHERE present=1 AND sha256 IS NOT NULL ORDER BY path").fetchall()
    folder = snapshot_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    path = folder / f"{stamp}.tsv"
    n = 1
    while path.exists():
        n += 1
        # '_' sorts after '.', so <stamp>_2.tsv comes after <stamp>.tsv by name.
        path = folder / f"{stamp}_{n}.tsv"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([r["path"], r["kind"], r["size"], r["mtime_ns"], r["sha256"],
                        r["image_hash"] or ""])
    os.replace(tmp, path)       # never leave a half-written snapshot behind
    result.path, result.files = path, len(rows)
    return result


# --------------------------------------------------------------------------- verify

@dataclass(frozen=True)
class SnapshotEntry:
    path: str
    kind: str
    size: int
    sha256: str
    image_hash: str


def read_snapshot(path: Path) -> list[SnapshotEntry]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"{path} is not a PhotoOrganizer snapshot (columns {reader.fieldnames})")
        return [SnapshotEntry(r["path"], r["kind"], int(r["size"]), r["sha256"], r["image_hash"])
                for r in reader]


def latest_snapshot(root: Path) -> Path | None:
    snaps = sorted(snapshot_dir(root).glob("*.tsv"))
    return snaps[-1] if snaps else None


@dataclass
class VerifyResult:
    snapshot: Path
    against: Path
    ok: list[str] = field(default_factory=list)
    metadata_only: list[str] = field(default_factory=list)  # jpg/heic: labels changed, picture identical
    sidecar_changed: list[str] = field(default_factory=list)  # .xmp edited (expected after review)
    changed: list[str] = field(default_factory=list)        # picture data or raw differs: a problem
    missing: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)           # on disk, not in the snapshot

    @property
    def problems(self) -> int:
        return len(self.changed) + len(self.missing) + len(self.unreadable)


def verify(snapshot_path: Path, against: Path, cfg: ScanConfig, exif: Exif,
           out: Progress = print) -> VerifyResult:
    """Re-hash every file in the snapshot under `against` (the library or a backup copy)."""
    against = Path(against).resolve()
    entries = read_snapshot(snapshot_path)
    result = VerifyResult(snapshot=Path(snapshot_path), against=against)

    present: list[SnapshotEntry] = []
    for e in entries:
        if (against / e.path).is_file():
            present.append(e)
        else:
            result.missing.append(e.path)

    meter = _Meter("verify", len(present), sum(e.size for e in present), out)
    recheck: list[SnapshotEntry] = []
    for e in present:
        full = against / e.path
        try:
            digest = sha256_file(full)
        except OSError:
            result.unreadable.append(e.path)
            continue
        meter.add(e.size)
        if digest == e.sha256:
            result.ok.append(e.path)
        elif e.kind == "xmp":
            result.sidecar_changed.append(e.path)
        elif e.kind in METADATA_WRITABLE and e.image_hash:
            recheck.append(e)
        else:
            result.changed.append(e.path)
    meter.done()

    for i in range(0, len(recheck), EXIF_BATCH):
        batch = recheck[i:i + EXIF_BATCH]
        hashes = exif.image_data_hashes([against / e.path for e in batch])
        for e in batch:
            value = hashes.get((against / e.path).resolve().as_posix())
            if value is None:
                result.unreadable.append(e.path)
            elif value == e.image_hash:
                result.metadata_only.append(e.path)
            else:
                result.changed.append(e.path)

    known = {e.path.lower() for e in entries}
    result.extra = [f.path for f in walk(against, cfg).files if f.path.lower() not in known]
    return result
