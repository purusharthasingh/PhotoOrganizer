"""Stage 5 (advisory): write Pending labels + reason keywords into XMP.

For every picture file the desired pipeline state is computed and compared with
what the file's XMP already holds; only the difference is written, so re-runs
are no-ops and stale pipeline labels are removed when a flag goes away.

Safety rules:
- Dry run unless apply=True. Applying takes a Stage 0 snapshot first.
- Raws are never written: a NEF's labels go to its `X.NEF.xmp` sidecar.
- A capture with any human decision (pick label, colour label or rating set by a
  person, on any of its files) is left completely alone.
- Only the pipeline's own namespaces are touched (`cull|`, `dupcopy|` in
  Milestone 1); every other keyword is preserved.
- After writing, each JPG/HEIC's picture-data hash is re-checked: labels must
  never change the picture.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .analysis import AnalyzeResult, run_analyze
from .config import Config
from .exif_io import Exif, ExifError
from .metadata import FileMeta, _as_list, human_decision, load_metadata
from .pair import _sidecar_map
from .scan import scan
from .snapshot import Progress, take_snapshot

PENDING_PICK, PENDING_LABEL = 2, "Yellow"
# Namespaces this stage reconciles. burst|/scene|/dup|/subject| arrive in Milestone 3.
M1_NAMESPACES = ("cull", "dupcopy")


@dataclass
class FileOp:
    target: str                    # file written: the JPG/HEIC itself, or the NEF's sidecar
    picture: str                   # the picture file the labels describe
    capture_id: str | None
    reasons: list[str]
    creates_sidecar: bool
    args: list[str]
    summary: str                   # human-readable description of the change


@dataclass
class FlagResult:
    analysis: AnalyzeResult | None = None
    applied: bool = False
    ops: list[FileOp] = field(default_factory=list)
    unchanged: int = 0
    human_skipped: list[str] = field(default_factory=list)    # capture ids left alone
    snapshot_path: Path | None = None
    written: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    picture_changed: list[str] = field(default_factory=list)  # must stay empty


def _ns(keyword: str, sep: str) -> str:
    return keyword.split(sep, 1)[0].lower()


def plan_file(picture: str, target: str, current: FileMeta | None, reasons: list[str],
              dupcopy_of: str | None) -> tuple[list[str], str] | None:
    """exiftool args turning `current` XMP into the desired pipeline state, or None."""
    raw = current.raw if current else {}
    hier = _as_list(raw.get("HierarchicalSubject"))
    tags = _as_list(raw.get("TagsList"))
    leaves = _as_list(raw.get("Subject"))

    want_hier = {f"cull|pending|{r}" for r in reasons}
    if dupcopy_of:
        want_hier.add(f"dupcopy|{dupcopy_of}")
    want_tags = {k.replace("|", "/") for k in want_hier}
    want_leaves = set(reasons)

    have_hier = {k for k in hier if _ns(k, "|") in M1_NAMESPACES}
    have_tags = {t for t in tags if _ns(t, "/") in M1_NAMESPACES}
    machine_leaves = {k.rsplit("|", 1)[-1] for k in have_hier if k.lower().startswith("cull|")}
    human_leaves = {k.rsplit("|", 1)[-1] for k in hier if _ns(k, "|") not in M1_NAMESPACES}
    have_leaves = {s for s in leaves if s in machine_leaves and s not in human_leaves}

    args: list[str] = []
    changes: list[str] = []
    for k in sorted(have_hier - want_hier):
        args.append(f"-XMP-lr:HierarchicalSubject-={k}")
        changes.append(f"-{k}")
    for k in sorted(want_hier - have_hier):
        args.append(f"-XMP-lr:HierarchicalSubject+={k}")
        changes.append(f"+{k}")
    for t in sorted(have_tags - want_tags):
        args.append(f"-XMP-digiKam:TagsList-={t}")
    for t in sorted(want_tags - have_tags):
        args.append(f"-XMP-digiKam:TagsList+={t}")
    for s in sorted(have_leaves - want_leaves):
        args.append(f"-XMP-dc:Subject-={s}")
    for s in sorted(want_leaves - set(leaves)):
        args.append(f"-XMP-dc:Subject+={s}")

    pick, label = raw.get("PickLabel"), raw.get("Label")
    pending_before = any(k.lower().startswith("cull|pending|") for k in have_hier)
    if want_hier and (reasons or dupcopy_of):
        if pick != PENDING_PICK:
            args.append(f"-XMP-digiKam:PickLabel={PENDING_PICK}")
            changes.append("pick=pending")
        if label != PENDING_LABEL:
            args.append(f"-XMP-xmp:Label={PENDING_LABEL}")
    elif pending_before:
        # The pipeline flagged this before and no longer does: undo its own labels.
        if pick == PENDING_PICK:
            args.append("-XMP-digiKam:PickLabel=")
            changes.append("pick cleared")
        if label == PENDING_LABEL:
            args.append("-XMP-xmp:Label=")
    if not args:
        return None
    return args, ", ".join(changes) or "keyword lists tidied"


def run_flag(root: Path, cfg: Config, conn: sqlite3.Connection, exif: Exif,
             apply: bool = False, out: Progress = print) -> FlagResult:
    root = Path(root).resolve()
    res = FlagResult(analysis=run_analyze(root, cfg, conn, exif, out))
    meta, _ = load_metadata(root, conn, exif, out)
    reasons_of = {s.capture_id: [r for r, _ in s.reasons] for s in res.analysis.scores}

    rows = [dict(r) for r in conn.execute(
        "SELECT path, kind, capture_id, role, duplicate_of FROM files"
        " WHERE present=1 ORDER BY path")]
    pictures = [r for r in rows if r["kind"] in ("nef", "jpg", "heic")]
    sidecars = _sidecar_map([r["path"] for r in rows], [r["path"] for r in pictures])
    capture_of = {r["path"]: r["capture_id"] for r in rows}

    def labels_source(pic: str) -> FileMeta | None:
        """Where this picture's labels live: the file itself, or a NEF's sidecar."""
        if pic.lower().endswith(".nef"):
            side = sidecars.get(pic)
            return meta.get(side) if side else None
        return meta.get(pic)

    # Human decisions win for the whole capture (pipeline's own labels excluded).
    human_caps = set()
    for r in pictures:
        rows_xmp = [m.raw for m in (meta.get(r["path"]), labels_source(r["path"])) if m]
        d = human_decision(rows_xmp)
        if d.pick is not None or d.label or d.rating is not None:
            human_caps.add(r["capture_id"])
    res.human_skipped = sorted(c for c in human_caps if c)

    for r in pictures:
        cid = r["capture_id"]
        if cid in human_caps:
            continue
        role = r["role"]
        if role == "keeper":
            reasons, dup_of = reasons_of.get(cid, []), None
        elif role == "duplicate_copy":
            reasons, dup_of = ["duplicate_copy"], capture_of.get(r["duplicate_of"])
        else:   # variant / conflict / unassigned: no pipeline labels
            reasons, dup_of = [], None
        if r["kind"] == "nef":
            target = sidecars.get(r["path"]) or f"{r['path']}.xmp"
        else:
            target = r["path"]
        planned = plan_file(r["path"], target, labels_source(r["path"]), reasons, dup_of)
        if planned is None:
            res.unchanged += 1
            continue
        args, summary = planned
        res.ops.append(FileOp(target, r["path"], cid, reasons,
                              not (root / target).exists(), args, summary))

    if not apply:
        return res

    res.applied = True
    snap = take_snapshot(root, cfg.scan, conn, exif, out)   # safety net before any write
    res.snapshot_path = snap.path
    before = {r["path"]: r["image_hash"] for r in conn.execute(
        "SELECT path, image_hash FROM files WHERE present=1")}

    for op in res.ops:
        try:
            exif.write(root / op.target, op.args)
            res.written.append(op.target)
        except ExifError as err:
            res.errors.append(str(err))

    # Labels must never change pictures: re-hash every JPG/HEIC we wrote.
    pics_written = [t for t in res.written if PurePosixPath(t).suffix.lower() != ".xmp"]
    after = exif.image_data_hashes([root / t for t in pics_written])
    for t in pics_written:
        if after.get((root / t).resolve().as_posix()) != before.get(t):
            res.picture_changed.append(t)

    # Record machine decisions per capture, and refresh the inventory (new sidecars).
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with conn:
        for s in res.analysis.scores:
            if s.capture_id in human_caps:
                continue
            reasons = [x for x, _ in s.reasons]
            if reasons:
                conn.execute(
                    "INSERT INTO decisions (capture_id, source, decision, reason, ts)"
                    " VALUES (?, 'machine', 'pending', ?, ?)"
                    " ON CONFLICT(capture_id, source) DO UPDATE SET decision=excluded.decision,"
                    " reason=excluded.reason, ts=excluded.ts"
                    " WHERE decisions.reason IS NOT excluded.reason",
                    (s.capture_id, ",".join(reasons), now))
            else:
                conn.execute("DELETE FROM decisions WHERE capture_id=? AND source='machine'",
                             (s.capture_id,))
    scan(root, cfg.scan, conn)
    return res
