"""SQLite manifest: one database per scanned root, at <root>/.photoorganizer/manifest.sqlite.

Paths are stored relative to the root, with forward slashes, so a library that
moves to another drive keeps its manifest.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

STATE_DIRNAME = ".photoorganizer"
DB_FILENAME = "manifest.sqlite"
SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
  capture_id  TEXT PRIMARY KEY,
  nef_path    TEXT,                 -- keeper copies only
  jpg_path    TEXT,
  heic_path   TEXT,
  dto         TEXT,                 -- DTO + SubSec, clock-offset corrected
  model       TEXT,
  serial      TEXT,
  af_x REAL, af_y REAL, af_w REAL, af_h REAL,
  folder      TEXT,                 -- shared parent, format subfolders ignored
  status      TEXT,                 -- paired | nef_only | jpg_only | single
  quarantined INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS files (
  path         TEXT PRIMARY KEY,    -- relative to the root, '/'-separated
  capture_id   TEXT,
  kind         TEXT NOT NULL,       -- nef | jpg | heic | xmp
  size         INTEGER NOT NULL,
  mtime_ns     INTEGER NOT NULL,
  sha256       TEXT,                -- full-file hash; cleared when size/mtime change
  image_hash   TEXT,                -- ImageDataHash; cleared when size/mtime change
  role         TEXT,                -- keeper | duplicate_copy | variant
  duplicate_of TEXT,
  present      INTEGER NOT NULL DEFAULT 1,   -- 0 = seen before, missing on the last scan
  first_seen   TEXT NOT NULL,
  last_seen    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS files_capture ON files(capture_id);

CREATE TABLE IF NOT EXISTS scores (
  capture_id    TEXT,
  metric        TEXT,
  value         REAL,
  model_version TEXT,
  PRIMARY KEY (capture_id, metric, model_version)
);

CREATE TABLE IF NOT EXISTS groups (
  capture_id TEXT PRIMARY KEY,
  burst_id   TEXT,
  scene_id   TEXT,
  dup_id     TEXT,
  subject_id TEXT
);

-- Cached exiftool metadata per file (v2). Valid while size/mtime_ns match `files`.
CREATE TABLE IF NOT EXISTS file_meta (
  path     TEXT PRIMARY KEY,
  size     INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL,
  data     TEXT NOT NULL              -- exiftool JSON row
);

CREATE TABLE IF NOT EXISTS decisions (
  capture_id TEXT,
  source     TEXT,                  -- machine | human
  decision   TEXT,                  -- reject | pending | pick
  reason     TEXT,
  ts         TEXT,
  PRIMARY KEY (capture_id, source)
);
"""


class ManifestError(RuntimeError):
    pass


def db_path(root: Path) -> Path:
    return Path(root) / STATE_DIRNAME / DB_FILENAME


def connect(root: Path) -> sqlite3.Connection:
    """Open (creating if needed) the manifest for a scanned root."""
    path = db_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        conn.close()
        raise ManifestError(
            f"{path} has schema version {version}, newer than this code ({SCHEMA_VERSION})")
    if version < SCHEMA_VERSION:
        with conn:
            conn.executescript(SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return conn
