"""Stage 2b: pair NEF+JPG keepers into captures, assign capture_ids, and write the manifest.

`run_pair` is the whole of Stage 2: scan -> image hashes -> metadata -> duplicate
copies -> pairing. It only READS photo files; all results go to the manifest.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .config import Config
from .dedupe import PICTURE_KINDS, DedupeResult, PicFile, find_duplicates, folder_report
from .exif_io import Exif
from .metadata import Decision, FileMeta, human_decision, keys_agree, load_metadata
from .scan import ScanResult, folder_hint, scan, shared_folder
from .snapshot import HashStats, Progress, fill_hashes

ID_SUFFIXES = "bcdefghijklmnopqrstuvwxyz"


@dataclass
class Capture:
    capture_id: str
    status: str                        # paired | nef_only | jpg_only | single
    nef: PicFile | None = None
    jpg: PicFile | None = None
    heic: PicFile | None = None
    fallback_id: bool = False

    @property
    def files(self) -> list[PicFile]:
        return [f for f in (self.nef, self.jpg, self.heic) if f]

    @property
    def lead(self) -> PicFile:
        return self.nef or self.jpg or self.heic


@dataclass
class PairResult:
    scan: ScanResult | None = None
    hashes: HashStats | None = None
    captures: list[Capture] = field(default_factory=list)
    dedupe: DedupeResult = field(default_factory=DedupeResult)
    folders: list = field(default_factory=list)            # dedupe.folder_report output
    variants: dict[str, str] = field(default_factory=dict)  # path -> capture_id
    collisions: list[list[str]] = field(default_factory=list)
    unconfirmed: list[list[str]] = field(default_factory=list)  # same stem, no capture data
    fallback_ids: list[str] = field(default_factory=list)
    suffixed_ids: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- helpers

def is_camera_jpg(meta: FileMeta | None) -> bool:
    """README: a JPG with a ShutterCount or a Nikon Make is a camera JPG; others are `single`."""
    if meta is None:
        return False
    return meta.shutter_count is not None or (meta.make or "").upper().startswith("NIKON")


def time_stamp(meta: FileMeta | None) -> str | None:
    """'2020:12:05 20:41:45' + SubSec '96' -> '20201205-204145.96'."""
    if meta is None or not meta.dto:
        return None
    digits = "".join(ch for ch in meta.dto if ch.isdigit())
    if len(digits) < 14:
        return None
    stamp = f"{digits[:8]}-{digits[8:14]}"
    return f"{stamp}.{meta.subsec}" if meta.subsec else stamp


def iso_time(meta: FileMeta | None) -> str | None:
    if meta is None or not meta.dto:
        return None
    d = meta.dto
    if len(d) < 19:
        return None
    iso = f"{d[0:4]}-{d[5:7]}-{d[8:10]}T{d[11:19]}"
    if meta.subsec:
        iso += f".{meta.subsec}"
    return iso + (meta.offset or "")


def model_slug(meta: FileMeta | None) -> str:
    model = (meta.model if meta else None) or "unknown"
    return "".join(model.split())


def make_capture_id(lead: PicFile) -> tuple[str, bool]:
    """(capture_id, used_fallback)."""
    m = lead.meta
    stamp = time_stamp(m)
    if m and m.serial and m.shutter_count is not None and stamp and m.subsec:
        return f"{m.serial}_{m.shutter_count}_{stamp}", False
    digest = lead.image_hash or lead.sha256 or hashlib.sha256(lead.path.encode()).hexdigest()
    return f"{model_slug(m)}_{stamp or 'nodate'}_{digest[:8]}", True


def _distance(a: PicFile, b: PicFile) -> int:
    """0 same folder, 1 sibling/sub/format-subfolder, 2 anywhere else."""
    if a.directory.lower() == b.directory.lower():
        return 0
    pa, pb = PurePosixPath(a.directory.lower()), PurePosixPath(b.directory.lower())
    if a.folder.lower() == b.folder.lower() or pa.parent == pb.parent or pa in pb.parents \
            or pb in pa.parents:
        return 1
    return 2


def _confirms(n: PicFile, j: PicFile) -> bool:
    kn, kj = n.capture_key(), j.capture_key()
    return kn[0] is not None and kj[0] is not None and keys_agree(kn, kj)


# --------------------------------------------------------------------------- pairing

def pair_keepers(keepers: list[PicFile]) -> tuple[list[Capture], PairResult]:
    res = PairResult()
    nefs: dict[str, list[PicFile]] = defaultdict(list)
    jpgs: dict[str, list[PicFile]] = defaultdict(list)
    singles: list[PicFile] = []
    for f in keepers:
        if f.kind == "nef":
            nefs[f.stem].append(f)
        elif f.kind == "jpg" and is_camera_jpg(f.meta):
            jpgs[f.stem].append(f)
        else:
            singles.append(f)

    pairs: list[tuple[PicFile | None, PicFile | None]] = []
    variant_of: dict[str, PicFile] = {}       # variant path -> file it would have paired with
    variants: list[PicFile] = []
    for stem in sorted(set(nefs) | set(jpgs)):
        ns, js = nefs.get(stem, []), jpgs.get(stem, [])
        edges = sorted(((_distance(n, j), n.path, j.path, n, j)
                        for n in ns for j in js if _confirms(n, j)), key=lambda e: e[:3])
        used: set[str] = set()
        for _d, _np, _jp, n, j in edges:
            if n.path not in used and j.path not in used:
                used |= {n.path, j.path}
                pairs.append((n, j))
        for _d, _np, _jp, n, j in edges:   # confirmed but left over: variants
            for f, other in ((n, j), (j, n)):
                if f.path not in used and other.path in used and f.path not in variant_of:
                    variant_of[f.path] = other
                    variants.append(f)
        leftovers = [f for f in ns + js if f.path not in used and f.path not in variant_of]
        for f in leftovers:
            pairs.append((f, None) if f.kind == "nef" else (None, f))
        if ns and js:
            # A file whose name matches the other kind but whose capture data matches none of
            # them: a different shot with the same name (README pairing rule 4).
            in_edge = {e[1] for e in edges} | {e[2] for e in edges}
            odd = [f for f in ns + js if f.path not in in_edge]
            if odd:
                group = sorted(f.path for f in ns + js)
                no_data = any(f.capture_key()[0] is None for f in odd)
                (res.unconfirmed if no_data else res.collisions).append(group)

    captures: list[Capture] = []
    for n, j in pairs:
        status = "paired" if n and j else ("nef_only" if n else "jpg_only")
        captures.append(Capture("", status, nef=n, jpg=j))
    for f in singles:
        captures.append(Capture("", "single", jpg=f if f.kind == "jpg" else None,
                                heic=f if f.kind == "heic" else None))

    # capture_ids: deterministic; colliding ids get -b, -c in path order.
    by_id: dict[str, list[Capture]] = defaultdict(list)
    for c in captures:
        cid, fallback = make_capture_id(c.lead)
        c.capture_id, c.fallback_id = cid, fallback
        by_id[cid].append(c)
    for cid, same in by_id.items():
        if len(same) > 1:
            same.sort(key=lambda c: c.lead.path)
            for i, c in enumerate(same[1:]):
                c.capture_id = f"{cid}-{ID_SUFFIXES[i]}"
                res.suffixed_ids.append(c.capture_id)
    res.fallback_ids = sorted(c.capture_id for c in captures if c.fallback_id)

    cap_of_path = {f.path: c.capture_id for c in captures for f in c.files}
    res.variants = {v.path: cap_of_path[variant_of[v.path].path] for v in variants}
    return captures, res


# --------------------------------------------------------------------------- stage 2

def _sidecar_map(paths: list[str], pictures: list[str]) -> dict[str, str]:
    """picture path -> its sidecar path. 'X.NEF.xmp' first; else a unique 'X.xmp' beside it."""
    xmps = {p.lower(): p for p in paths if p.lower().endswith(".xmp")}
    by_stem_dir: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pic in pictures:
        pp = PurePosixPath(pic)
        by_stem_dir[(str(pp.parent).lower(), pp.stem.lower())].append(pic)
    out: dict[str, str] = {}
    for pic in pictures:
        full = xmps.get(pic.lower() + ".xmp")
        if full:
            out[pic] = full
            continue
        pp = PurePosixPath(pic)
        short = xmps.get(str(pp.with_suffix(".xmp")).lower())
        if short and len(by_stem_dir[(str(pp.parent).lower(), pp.stem.lower())]) == 1:
            out[pic] = short
    return out


def run_pair(root: Path, cfg: Config, conn: sqlite3.Connection, exif: Exif,
             out: Progress = print) -> PairResult:
    root = Path(root).resolve()
    scan_result = scan(root, cfg.scan, conn)
    hashes = fill_hashes(root, conn, exif, out, sha=False, image=True)
    meta, unreadable = load_metadata(root, conn, exif, out)

    rows = conn.execute(
        "SELECT path, kind, size, mtime_ns, sha256, image_hash FROM files WHERE present=1"
        " ORDER BY path").fetchall()
    all_paths = [r["path"] for r in rows]
    pic_paths = [r["path"] for r in rows if r["kind"] in PICTURE_KINDS]
    sidecars = _sidecar_map(all_paths, pic_paths)

    fmt = cfg.scan.format_subfolders
    pics: list[PicFile] = []
    for r in rows:
        if r["kind"] not in PICTURE_KINDS:
            continue
        m = meta.get(r["path"])
        xmp_rows = [m.raw] if m else []
        side = sidecars.get(r["path"])
        if side and side in meta:
            xmp_rows.append(meta[side].raw)
        pics.append(PicFile(
            path=r["path"], kind=r["kind"], size=r["size"], mtime_ns=r["mtime_ns"],
            image_hash=r["image_hash"] or None, sha256=r["sha256"], meta=m,
            decision=human_decision(xmp_rows) if xmp_rows else Decision(),
            folder=folder_hint(r["path"], fmt)))

    dd = find_duplicates(pics, cfg.preferred_roots)
    roles = dd.role_of()
    keepers = [f for f in pics if roles.get(f.path, ("keeper", None))[0] == "keeper"]
    captures, res = pair_keepers(keepers)
    res.scan, res.hashes, res.dedupe = scan_result, hashes, dd
    res.unreadable = sorted(set(unreadable) | set(hashes.errors))

    folder_of = {f.path: f.folder for f in pics}
    res.folders = folder_report(pics, roles, folder_of)
    res.captures = captures

    # ---- write the manifest (one transaction)
    cap_of_path = {f.path: c.capture_id for c in captures for f in c.files}
    file_rows: dict[str, tuple[str | None, str | None, str | None]] = {}
    for f in pics:
        role, of = roles.get(f.path, ("keeper", None))
        if f.path in res.variants:
            file_rows[f.path] = (res.variants[f.path], "variant", None)
        elif role == "keeper":
            file_rows[f.path] = (cap_of_path.get(f.path), "keeper", None)
        else:   # duplicate_copy / conflict: belongs to the keeper's capture
            file_rows[f.path] = (cap_of_path.get(of), role, of)
    for pic, side in sidecars.items():
        file_rows[side] = (file_rows.get(pic, (None,))[0], None, None)

    with conn:
        conn.execute("UPDATE files SET capture_id=NULL, role=NULL, duplicate_of=NULL"
                     " WHERE present=1")
        conn.executemany("UPDATE files SET capture_id=?, role=?, duplicate_of=? WHERE path=?",
                         [(cid, role, of, path) for path, (cid, role, of) in file_rows.items()])
        new_ids = {c.capture_id for c in captures}
        old_ids = {r[0] for r in conn.execute("SELECT capture_id FROM photos WHERE quarantined=0")}
        conn.executemany("DELETE FROM photos WHERE capture_id=?",
                         [(cid,) for cid in old_ids - new_ids])
        for c in captures:
            lead = c.lead.meta
            conn.execute(
                "INSERT INTO photos (capture_id, nef_path, jpg_path, heic_path, dto, model, serial,"
                " folder, status) VALUES (?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(capture_id) DO UPDATE SET nef_path=excluded.nef_path,"
                " jpg_path=excluded.jpg_path, heic_path=excluded.heic_path, dto=excluded.dto,"
                " model=excluded.model, serial=excluded.serial, folder=excluded.folder,"
                " status=excluded.status",
                (c.capture_id,
                 c.nef.path if c.nef else None, c.jpg.path if c.jpg else None,
                 c.heic.path if c.heic else None,
                 iso_time(lead), lead.model if lead else None, lead.serial if lead else None,
                 shared_folder([f.path for f in c.files], fmt), c.status))
    return res
