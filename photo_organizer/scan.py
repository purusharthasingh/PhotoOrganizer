"""Stage 1: scan any folder layout and inventory photo files into the manifest.

The scan never touches photo files (its only write is the manifest under
<root>/.photoorganizer/). It records each NEF/JPG/HEIC/XMP file's
size and mtime in the `files` table and reports what is new, changed, missing
or has a long path since the previous scan. Hashes and metadata are Stage 2's job.
"""

from __future__ import annotations

import os
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .config import ScanConfig


@dataclass(frozen=True)
class FileEntry:
    path: str          # relative to the root, '/'-separated
    kind: str
    size: int
    mtime_ns: int


@dataclass
class ScanResult:
    root: Path
    files: list[FileEntry] = field(default_factory=list)
    new: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    reappeared: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    long_paths: list[str] = field(default_factory=list)
    ignored: Counter = field(default_factory=Counter)   # extension -> count
    errors: list[str] = field(default_factory=list)

    def kind_counts(self) -> Counter:
        return Counter(f.kind for f in self.files)


def folder_hint(rel_path: str, format_subfolders: frozenset[str]) -> str:
    """Folder of one file, with trailing format subfolders (Raw/, JPG/...) dropped.

    'Puerto Rico 26/Raw/DSC_9474.NEF' -> 'Puerto Rico 26'. Files at the root give ''.
    """
    parts = list(PurePosixPath(rel_path).parent.parts)
    while parts and parts[-1].lower() in format_subfolders:
        parts.pop()
    return "/".join(parts)


def shared_folder(rel_paths: list[str], format_subfolders: frozenset[str]) -> str:
    """A capture's folder: the common parent of its files' folder hints.

    Sibling format folders ('Shoot/RAW/x.NEF', 'Shoot/JPG/x.JPG') both give 'Shoot'.
    """
    if not rel_paths:
        raise ValueError("shared_folder needs at least one path")
    split = [folder_hint(p, format_subfolders).split("/") for p in rel_paths]
    common: list[str] = []
    for level in zip(*split):
        if any(part.lower() != level[0].lower() for part in level):
            break
        common.append(level[0])
    return "/".join(p for p in common if p)


def walk(root: Path, cfg: ScanConfig) -> ScanResult:
    """List photo files under root. Read-only; touches only the file system."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"not a folder: {root}")
    result = ScanResult(root=root)

    def on_error(err: OSError) -> None:
        result.errors.append(f"{err.filename}: {err.strerror}")

    for dirpath, dirnames, filenames in os.walk(root, onerror=on_error):
        # Prune in place so skipped folders are never entered.
        dirnames[:] = sorted(d for d in dirnames if d.lower() not in cfg.skip_dirs)
        for name in sorted(filenames):
            ext = os.path.splitext(name)[1].lower()
            kind = cfg.extensions.get(ext)
            full = os.path.join(dirpath, name)
            if kind is None:
                result.ignored[ext or "(none)"] += 1
                continue
            try:
                st = os.stat(full)
            except OSError as err:
                result.errors.append(f"{full}: {err.strerror}")
                continue
            rel = Path(full).relative_to(root).as_posix()
            result.files.append(FileEntry(rel, kind, st.st_size, st.st_mtime_ns))
            if len(full) > cfg.path_warn_length:
                result.long_paths.append(full)
    return result


def scan(root: Path, cfg: ScanConfig, conn: sqlite3.Connection) -> ScanResult:
    """Walk root and reconcile the result with the manifest's `files` table."""
    result = walk(root, cfg)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    previous = {
        row["path"]: row
        for row in conn.execute("SELECT path, kind, size, mtime_ns, present FROM files")
    }
    seen: set[str] = set()

    with conn:
        for f in result.files:
            seen.add(f.path)
            old = previous.get(f.path)
            if old is None:
                result.new.append(f.path)
                conn.execute(
                    "INSERT INTO files (path, kind, size, mtime_ns, present, first_seen, last_seen)"
                    " VALUES (?, ?, ?, ?, 1, ?, ?)",
                    (f.path, f.kind, f.size, f.mtime_ns, now, now))
                continue
            if not old["present"]:
                result.reappeared.append(f.path)
            if (old["size"], old["mtime_ns"], old["kind"]) != (f.size, f.mtime_ns, f.kind):
                result.changed.append(f.path)
                # Content may differ: drop cached hashes so Stage 2 recomputes them.
                conn.execute(
                    "UPDATE files SET kind=?, size=?, mtime_ns=?, sha256=NULL, image_hash=NULL,"
                    " present=1, last_seen=? WHERE path=?",
                    (f.kind, f.size, f.mtime_ns, now, f.path))
            else:
                if old["present"]:
                    result.unchanged.append(f.path)
                conn.execute("UPDATE files SET present=1, last_seen=? WHERE path=?", (now, f.path))

        for path, row in previous.items():
            if path not in seen and row["present"]:
                result.missing.append(path)
                conn.execute("UPDATE files SET present=0 WHERE path=?", (path,))

    return result
