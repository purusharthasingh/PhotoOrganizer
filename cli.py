"""PhotoOrganizer command line. Run `python cli.py --help`."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from photo_organizer import config as config_mod
from photo_organizer import manifest
from photo_organizer.exif_io import ExifError
from photo_organizer.scan import ScanResult, folder_hint, scan


def _root(args, cfg) -> Path:
    root = Path(args.root) if args.root else cfg.library_root
    if root is None:
        sys.exit("error: no root given and [library] root is not set in config.toml")
    return root.resolve()


def _print_list(title: str, items: list[str], limit: int) -> None:
    if not items:
        return
    print(f"\n{title} ({len(items)}):")
    for item in items[:limit]:
        print(f"  {item}")
    if len(items) > limit:
        print(f"  ... and {len(items) - limit} more (use --verbose)")


def report_scan(result: ScanResult, cfg, verbose: bool) -> None:
    limit = 10**9 if verbose else 10
    counts = result.kind_counts()
    kinds = ", ".join(f"{counts[k]} {k}" for k in ("nef", "jpg", "heic", "xmp") if counts[k])
    print(f"Scanned {result.root}")
    print(f"  {len(result.files)} files ({kinds or 'none'})")
    print(f"  new {len(result.new)}, changed {len(result.changed)}, unchanged {len(result.unchanged)},"
          f" reappeared {len(result.reappeared)}, missing {len(result.missing)}")
    if result.ignored:
        ignored = ", ".join(f"{n} {ext}" for ext, n in result.ignored.most_common())
        print(f"  ignored (not photos): {ignored}")

    by_folder: dict[str, int] = defaultdict(int)
    for f in result.files:
        if f.kind != "xmp":
            by_folder[folder_hint(f.path, cfg.scan.format_subfolders) or "."] += 1
    print("\nFolder hints (photo files per folder, format subfolders merged):")
    for folder in sorted(by_folder):
        print(f"  {by_folder[folder]:5d}  {folder}")

    if verbose:
        _print_list("New", result.new, limit)
        _print_list("Changed", result.changed, limit)
    else:
        _print_list("Changed", result.changed, limit)
    _print_list("Reappeared", result.reappeared, limit)
    _print_list("Missing (in manifest, gone from disk)", result.missing, limit)
    _print_list(f"WARNING: paths over {cfg.scan.path_warn_length} characters", result.long_paths, limit)
    _print_list("ERRORS (could not read)", result.errors, limit)


def cmd_scan(args) -> int:
    cfg = config_mod.load(args.config)
    root = _root(args, cfg)
    conn = manifest.connect(root)
    try:
        result = scan(root, cfg.scan, conn)
    finally:
        conn.close()
    report_scan(result, cfg, args.verbose)
    print(f"\nManifest: {manifest.db_path(root)}")
    return 1 if result.errors else 0


def cmd_snapshot(args) -> int:
    from photo_organizer.exif_io import Exif
    from photo_organizer.snapshot import take_snapshot

    cfg = config_mod.load(args.config)
    root = _root(args, cfg)
    conn = manifest.connect(root)
    try:
        with Exif() as exif:
            res = take_snapshot(root, cfg.scan, conn, exif)
    finally:
        conn.close()
    s = res.scan
    print(f"Scanned {s.root}: {len(s.files)} files"
          f" (new {len(s.new)}, changed {len(s.changed)}, missing {len(s.missing)})")
    print(f"Hashed {res.hashed_sha} files (sha256) and {res.hashed_image} pictures (image hash);"
          f" the rest were cached")
    print(f"Snapshot of {res.files} files: {res.path}")
    limit = 10**9 if args.verbose else 10
    _print_list("Changed while hashing (left out; run again)", res.unstable, limit)
    _print_list("No image hash (exiftool found no picture data)", res.no_image_hash, limit)
    _print_list("ERRORS", res.errors + s.errors, limit)
    return 1 if (res.errors or s.errors or res.unstable) else 0


def cmd_verify(args) -> int:
    from photo_organizer.exif_io import Exif
    from photo_organizer.snapshot import latest_snapshot, verify

    cfg = config_mod.load(args.config)
    root = _root(args, cfg)
    snap = Path(args.snapshot) if args.snapshot else latest_snapshot(root)
    if snap is None or not snap.is_file():
        print(f"error: no snapshot found (run: python cli.py snapshot {root})", file=sys.stderr)
        return 2
    against = Path(args.against).resolve() if args.against else root
    with Exif() as exif:
        res = verify(snap, against, cfg.scan, exif)

    print(f"Snapshot: {res.snapshot}")
    print(f"Checked:  {res.against}")
    print(f"  ok {len(res.ok)}, labels only {len(res.metadata_only)},"
          f" sidecar edited {len(res.sidecar_changed)}, CHANGED {len(res.changed)},"
          f" MISSING {len(res.missing)}, UNREADABLE {len(res.unreadable)}, extra {len(res.extra)}")
    limit = 10**9 if args.verbose else 10
    _print_list("CHANGED (picture data or raw differs)", res.changed, limit)
    _print_list("MISSING", res.missing, limit)
    _print_list("UNREADABLE", res.unreadable, limit)
    if args.verbose:
        _print_list("Labels only (metadata changed, picture identical)", res.metadata_only, limit)
        _print_list("Sidecar edited", res.sidecar_changed, limit)
    _print_list("Extra (not in snapshot)", res.extra, limit)
    print("\nResult:", "OK" if res.problems == 0 else f"{res.problems} PROBLEM(S)")
    return 1 if res.problems else 0


def cmd_pair(args) -> int:
    from collections import Counter

    from photo_organizer.exif_io import Exif
    from photo_organizer.pair import run_pair

    cfg = config_mod.load(args.config)
    root = _root(args, cfg)
    conn = manifest.connect(root)
    try:
        with Exif() as exif:
            res = run_pair(root, cfg, conn, exif)
    finally:
        conn.close()
    limit = 10**9 if args.verbose else 10
    s = res.scan
    print(f"Scanned {s.root}: {len(s.files)} files"
          f" (new {len(s.new)}, changed {len(s.changed)}, missing {len(s.missing)})")

    statuses = Counter(c.status for c in res.captures)
    print(f"\nCaptures: {len(res.captures)}  ("
          + ", ".join(f"{statuses[k]} {k}" for k in ("paired", "nef_only", "jpg_only", "single")
                      if statuses[k]) + ")")
    by_folder: dict[str, Counter] = {}
    from photo_organizer.scan import shared_folder
    for c in res.captures:
        folder = shared_folder([f.path for f in c.files], cfg.scan.format_subfolders) or "."
        by_folder.setdefault(folder, Counter())[c.status] += 1
    for folder in sorted(by_folder):
        cnt = by_folder[folder]
        print(f"  {folder}: " + ", ".join(f"{n} {k}" for k, n in sorted(cnt.items())))
    if args.verbose:
        for c in sorted(res.captures, key=lambda c: c.lead.path):
            print(f"    {c.capture_id:40s} {c.status:9s} " + " + ".join(f.path for f in c.files))

    dd = res.dedupe
    n_dups = sum(len(g.copies) for g in dd.groups)
    print(f"\nDuplicate copies: {n_dups} file(s) in {len(dd.groups)} group(s)"
          " (marked in the manifest only; nothing is moved)")
    for g in dd.groups[:limit]:
        print(f"  keep {g.keeper.path}   [{g.reason}]")
        for c in g.copies:
            print(f"     copy {c.path}")
    if len(dd.groups) > limit:
        print(f"  ... and {len(dd.groups) - limit} more groups (use --verbose)")
    for folder, n, total, keeper_folders in res.folders:
        where = ", ".join(f"{k or '.'} ({m})" for k, m in keeper_folders)
        if n == total:
            print(f"  DUPLICATE FOLDER: {folder} (100%, all {total} files kept in: {where})")
        else:
            print(f"  Partial overlap: {folder} is {100 * n // total}% duplicates"
                  f" ({n}/{total}) of: {where}")

    _print_list("CONFLICTS: copies with different human decisions (none marked)",
                [" vs ".join(f"{f.path} [{f.decision.describe()}]" for f in c.files)
                 for c in dd.conflicts], limit)
    _print_list("Duplicates carrying metadata the keeper lacks (not merged)",
                [f"{p}: {what}" for p, what in sorted(dd.extra_metadata.items())], limit)
    _print_list("Same picture data but different capture data (not treated as copies)",
                [" = ".join(g) for g in dd.odd_matches], limit)
    _print_list("Variants (same capture data, different picture; attached, not marked)",
                [f"{p} -> {cid}" for p, cid in sorted(res.variants.items())], limit)
    _print_list("Name collisions (same name, different shot; kept as separate captures)",
                [" / ".join(g) for g in res.collisions], limit)
    _print_list("Same name but no capture data to confirm a pair",
                [" / ".join(g) for g in res.unconfirmed], limit)
    _print_list("Fallback capture IDs (missing serial/shutter count/subsec)", res.fallback_ids, limit)
    _print_list("Capture IDs given a suffix to stay unique", res.suffixed_ids, limit)
    _print_list("ERRORS (unreadable)", res.unreadable + s.errors, limit)
    print(f"\nManifest: {manifest.db_path(root)}")
    return 1 if (res.unreadable or s.errors) else 0


def cmd_analyze(args) -> int:
    from collections import Counter

    from photo_organizer.analysis import run_analyze
    from photo_organizer.exif_io import Exif

    cfg = config_mod.load(args.config)
    root = _root(args, cfg)
    conn = manifest.connect(root)
    try:
        with Exif() as exif:
            res = run_analyze(root, cfg, conn, exif, workers=args.workers)
    finally:
        conn.close()

    print(f"Analyzed {len(res.scores)} captures: {res.computed} computed, {res.cached} cached"
          f" (advisory only; nothing is written to photo files)")
    if res.blur_skipped:
        print("  Blur: too few captures to compare sharpness; no blur flags.")
    counts = Counter(r for s in res.scores for r, _ in s.reasons)
    extreme = sum(1 for s in res.scores for _, x in s.reasons if x)
    print("Flags: " + (", ".join(f"{n} {r}" for r, n in counts.most_common()) or "none")
          + (f"  ({extreme} extreme)" if extreme else ""))

    rows = res.scores if args.verbose else [s for s in res.scores if s.reasons]
    if rows:
        print(f"\n{'capture':34s} {'mean':>5s} {'p50':>5s} {'clip_hi':>7s} {'clip_lo':>7s}"
              f" {'lap_var':>8s} {'z':>5s}  reasons")
        for s in sorted(rows, key=lambda s: (s.folder, s.source_path)):
            m = s.metrics
            if not m:
                continue
            z = f"{s.sharpness_z:5.2f}" if s.sharpness_z is not None else "    -"
            why = ", ".join(r + (" (extreme)" if x else "") for r, x in s.reasons) or "-"
            print(f"{s.source_path[:34]:34s} {m['mean']:5.0f} {m['p50']:5.0f} {m['clip_hi']:7.3f}"
                  f" {m['clip_lo']:7.3f} {m['lap_var']:8.0f} {z}  {why}")
    _print_list("Too dark to judge sharpness (no blur verdict)", res.no_detail, 10**9)
    _print_list("ERRORS (could not analyze)", res.errors, 10**9)
    print(f"\nFull report: {res.report_path}")
    return 1 if res.errors else 0


def cmd_flag(args) -> int:
    from collections import Counter

    from photo_organizer.exif_io import Exif
    from photo_organizer.flag import run_flag

    cfg = config_mod.load(args.config)
    root = _root(args, cfg)
    conn = manifest.connect(root)
    try:
        with Exif() as exif:
            res = run_flag(root, cfg, conn, exif, apply=args.apply)
    finally:
        conn.close()
    limit = 10**9 if args.verbose else 25

    flagged = Counter(r for op in res.ops for r in op.reasons)
    new_sidecars = sum(op.creates_sidecar for op in res.ops)
    mode = "APPLIED" if res.applied else "DRY RUN (nothing written; add --apply)"
    print(f"Flag, advisory mode: {mode}")
    print(f"  {len(res.ops)} file(s) to update ({new_sidecars} new NEF sidecar(s)),"
          f" {res.unchanged} already up to date")
    if flagged:
        print("  Pending reasons (per file): " + ", ".join(f"{n} {r}" for r, n in flagged.most_common()))
    _print_list("Changes", [f"{op.target}: {op.summary}" + (" [new sidecar]" if op.creates_sidecar else "")
                            for op in res.ops], limit)
    _print_list("Left alone: captures with a human decision", res.human_skipped, limit)
    if res.applied:
        print(f"\nSnapshot taken before writing: {res.snapshot_path}")
        print(f"Written: {len(res.written)} file(s)")
        _print_list("ERRORS", res.errors, 10**9)
        if res.picture_changed:
            _print_list("!!! PICTURE DATA CHANGED (should never happen) — restore these from"
                        " backup and report", res.picture_changed, 10**9)
        else:
            print("Picture data re-checked: unchanged in every written JPG/HEIC.")
        print("In digiKam: select the photos, then Item -> Reread Metadata From File.")
    return 1 if (res.errors or res.picture_changed) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cli.py", description="PhotoOrganizer culling pipeline")
    parser.add_argument("--config", help="path to config.toml (default: the project's)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scan", help="Stage 1: inventory photo files into the manifest (read-only)")
    p.add_argument("root", nargs="?", help="folder to scan (default: [library] root in config.toml)")
    p.add_argument("-v", "--verbose", action="store_true", help="list every file, not just the first 10")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("snapshot", help="Stage 0: record sha256 + image hash of every file")
    p.add_argument("root", nargs="?", help="library folder (default: [library] root in config.toml)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("verify", help="Stage 0: check the library or a backup against a snapshot")
    p.add_argument("root", nargs="?", help="library folder the snapshot was taken of")
    p.add_argument("--snapshot", help="snapshot TSV (default: the root's latest)")
    p.add_argument("--against", help="folder to check instead of the root, e.g. a backup copy")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("pair", help="Stage 2: find duplicate copies and pair NEF+JPG (read-only)")
    p.add_argument("root", nargs="?", help="folder (default: [library] root in config.toml)")
    p.add_argument("-v", "--verbose", action="store_true", help="list every capture")
    p.set_defaults(func=cmd_pair)

    p = sub.add_parser("analyze", help="Stage 3: exposure and blur scores (read-only, advisory)")
    p.add_argument("root", nargs="?", help="folder (default: [library] root in config.toml)")
    p.add_argument("-v", "--verbose", action="store_true", help="show every capture, not just flagged")
    p.add_argument("--workers", type=int, help="parallel decodes (default: min(8, CPUs))")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("flag", help="Stage 5: write Pending labels to XMP (dry run unless --apply)")
    p.add_argument("root", nargs="?", help="folder (default: [library] root in config.toml)")
    p.add_argument("--apply", action="store_true", help="actually write (takes a snapshot first)")
    p.add_argument("--advisory", action="store_true", default=True,
                   help="Pending labels only (the only mode until Milestone 5)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_flag)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (config_mod.ConfigError, manifest.ManifestError, ExifError, NotADirectoryError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
