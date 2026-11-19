"""Stage 2 input: read capture metadata and XMP decisions with exiftool, cached in `file_meta`.

Nothing here writes to photo files; only the manifest is updated.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .exif_io import Exif
from .snapshot import Progress, _Meter

EXIF_BATCH = 64
# Bump when the tag lists change, so cached rows are re-read.
META_VERSION = 3
VERSION_KEY = "_meta_version"

CAPTURE_TAGS = [
    "DateTimeOriginal", "SubSecTimeOriginal", "OffsetTimeOriginal",
    "Make", "Model", "SerialNumber", "ShutterCount",
    "AFAreaXPosition", "AFAreaYPosition", "AFAreaWidth", "AFAreaHeight",
    "AFImageWidth", "AFImageHeight", "AFCoordinatesAvailable", "FocusResult",
    "FocusDistance",
    "Orientation", "ImageWidth", "ImageHeight", "GPSLatitude", "GPSLongitude",
]
# Viewfinder (phase-detect) AF records named grid points instead of coordinates.
# Read in exiftool's readable form ('C6 (Center)', 'B5,C5,C6'), not as raw codes.
AF_NAME_TAGS = [
    "AFDetectionMethod", "AFAreaMode", "FocusPointSchema", "PrimaryAFPoint",
    "AFPointsUsed", "AFPointsInFocus",
]
# Requested with their XMP group so EXIF:Rating and friends don't shadow them.
DECISION_TAGS = [
    "XMP-xmp:Rating", "XMP-digiKam:PickLabel", "XMP-xmp:Label",
    "XMP-lr:HierarchicalSubject", "XMP-dc:Subject", "XMP-digiKam:TagsList",
]
# Keyword namespaces the pipeline owns (README: Group IDs, rule 3).
OWNED_NAMESPACES = ("cull", "burst", "scene", "dup", "subject", "dupcopy")
# PickLabel the pipeline writes for each cull state.
MACHINE_PICK = {"pending": 2, "reject": 1}


def _as_list(value) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _owned(keyword: str, sep: str) -> bool:
    return keyword.split(sep, 1)[0].lower() in OWNED_NAMESPACES


@dataclass(frozen=True)
class Decision:
    """What a person set in the XMP, with the pipeline's own labels taken out."""
    pick: int | None = None
    rating: int | None = None
    label: str | None = None
    keywords: frozenset[str] = frozenset()

    def __bool__(self) -> bool:
        return any(v is not None for v in (self.pick, self.rating, self.label)) or bool(self.keywords)

    def describe(self) -> str:
        parts = []
        if self.pick is not None:
            parts.append(f"pick={self.pick}")
        if self.rating is not None:
            parts.append(f"rating={self.rating}")
        if self.label:
            parts.append(f"label={self.label}")
        if self.keywords:
            parts.append("keywords=" + ",".join(sorted(self.keywords)))
        return " ".join(parts) or "none"


def human_decision(rows: list[dict]) -> Decision:
    """Combine the XMP of a file and its sidecar into a human Decision."""
    hier: list[str] = []
    tags: list[str] = []
    leaves: list[str] = []
    pick = rating = None
    label = None
    for r in rows:
        hier += _as_list(r.get("HierarchicalSubject"))
        tags += _as_list(r.get("TagsList"))
        leaves += _as_list(r.get("Subject"))
        pick = _int(r.get("PickLabel")) if r.get("PickLabel") is not None else pick
        rating = _int(r.get("Rating")) if r.get("Rating") is not None else rating
        label = str(r["Label"]) if r.get("Label") else label

    owned = [k for k in hier if _owned(k, "|")]
    cull_states = {k.split("|")[1].lower() for k in owned
                   if k.lower().startswith("cull|") and k.count("|") >= 1}
    owned_leaves = {k.rsplit("|", 1)[-1].lower() for k in owned}

    # Labels that match what the pipeline itself wrote are machine decisions.
    if pick is not None and any(MACHINE_PICK.get(s) == pick for s in cull_states):
        pick = None
    if pick == 0:
        pick = None
    if rating is not None and (rating == 0 or (rating == -1 and "reject" in cull_states)):
        rating = None
    if label and label.lower() == "yellow" and "pending" in cull_states:
        label = None

    keywords = {k for k in hier if not _owned(k, "|")}
    keywords |= {t.replace("/", "|") for t in tags if not _owned(t, "/")}
    keywords |= {s for s in leaves
                 if s.lower() not in owned_leaves
                 and not any(k.rsplit("|", 1)[-1] == s for k in keywords)}
    return Decision(pick, rating, label, frozenset(keywords))


@dataclass(frozen=True)
class FileMeta:
    path: str
    raw: dict

    def get(self, key):
        return self.raw.get(key)

    @property
    def dto(self) -> str | None:
        v = self.raw.get("DateTimeOriginal")
        return str(v) if v and not str(v).startswith("0000") else None

    @property
    def subsec(self) -> str | None:
        """Hundredths of a second as two digits. exiftool keeps leading zeros as text."""
        v = self.raw.get("SubSecTimeOriginal")
        if v is None or v == "":
            return None
        return str(v).strip().ljust(2, "0")[:2]

    @property
    def offset(self) -> str | None:
        v = self.raw.get("OffsetTimeOriginal")
        return str(v) if v else None

    @property
    def make(self) -> str | None:
        v = self.raw.get("Make")
        return str(v).strip() if v else None

    @property
    def model(self) -> str | None:
        v = self.raw.get("Model")
        return str(v).strip() if v else None

    @property
    def serial(self) -> str | None:
        v = self.raw.get("SerialNumber")
        return str(v).strip() if v not in (None, "") else None

    @property
    def shutter_count(self) -> int | None:
        return _int(self.raw.get("ShutterCount"))

    @property
    def has_gps(self) -> bool:
        return self.raw.get("GPSLatitude") is not None

    def capture_key(self) -> tuple:
        """Fields that must agree for two files to be the same capture."""
        return (self.dto, self.subsec, self.shutter_count, self.serial)


def keys_agree(a: tuple, b: tuple) -> bool:
    """README pairing rule 2: every field must match; a field missing from BOTH is ignored."""
    return all(x == y for x, y in zip(a, b) if not (x is None and y is None))


def load_metadata(root: Path, conn: sqlite3.Connection, exif: Exif,
                  out: Progress = print) -> tuple[dict[str, FileMeta], list[str]]:
    """Return {path: FileMeta} for every present file, reading only files whose cache is stale.

    Returns (metadata, unreadable paths).
    """
    root = Path(root).resolve()
    rows = conn.execute(
        "SELECT f.path, f.kind, f.size, f.mtime_ns, m.size AS msize, m.mtime_ns AS mmtime, m.data"
        " FROM files f LEFT JOIN file_meta m ON m.path = f.path"
        " WHERE f.present = 1 ORDER BY f.path").fetchall()

    meta: dict[str, FileMeta] = {}
    stale = []
    for r in rows:
        data = json.loads(r["data"]) if r["data"] is not None else None
        if data is not None and data.get(VERSION_KEY) == META_VERSION                 and (r["msize"], r["mmtime"]) == (r["size"], r["mtime_ns"]):
            meta[r["path"]] = FileMeta(r["path"], data)
        else:
            stale.append(r)

    unreadable: list[str] = []
    if stale:
        meter = _Meter("metadata", len(stale), sum(r["size"] for r in stale), out)
        tags = CAPTURE_TAGS + DECISION_TAGS
        for i in range(0, len(stale), EXIF_BATCH):
            batch = stale[i:i + EXIF_BATCH]
            batch_paths = [root / r["path"] for r in batch]
            got = exif.read_tags(batch_paths, tags)
            names = exif.read_tags(batch_paths, AF_NAME_TAGS, numeric=False)
            with conn:
                for r in batch:
                    meter.add(r["size"])
                    data = got.get((root / r["path"]).resolve().as_posix())
                    if data is None:
                        unreadable.append(r["path"])
                        continue
                    data = {k: v for k, v in data.items() if k != "SourceFile"}
                    af = names.get((root / r["path"]).resolve().as_posix(), {})
                    data.update({k: v for k, v in af.items() if k != "SourceFile"})
                    data[VERSION_KEY] = META_VERSION
                    conn.execute(
                        "INSERT OR REPLACE INTO file_meta (path, size, mtime_ns, data)"
                        " VALUES (?, ?, ?, ?)",
                        (r["path"], r["size"], r["mtime_ns"], json.dumps(data)))
                    meta[r["path"]] = FileMeta(r["path"], data)
        meter.done()
    with conn:   # drop cache rows for files no longer present
        conn.execute("DELETE FROM file_meta WHERE path NOT IN (SELECT path FROM files WHERE present=1)")
    return meta, unreadable
