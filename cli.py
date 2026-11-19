"""PhotoOrganizer command line. Run `python cli.py --help`."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from photo_organizer import config as config_mod
from photo_organizer import manifest
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cli.py", description="PhotoOrganizer culling pipeline")
    parser.add_argument("--config", help="path to config.toml (default: the project's)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scan", help="Stage 1: inventory photo files into the manifest (read-only)")
    p.add_argument("root", nargs="?", help="folder to scan (default: [library] root in config.toml)")
    p.add_argument("-v", "--verbose", action="store_true", help="list every file, not just the first 10")
    p.set_defaults(func=cmd_scan)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (config_mod.ConfigError, manifest.ManifestError, NotADirectoryError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
