"""Stage 3 runner: exposure + blur for every capture, cached in `scores`.

Cache key (scores.model_version) = analyser version | preview size | source picture
hash, so scores are recomputed only when the code, the preview size or the actual
picture changes (not when labels are written into a JPG). Read-only on photo files.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import manifest
from .analyze import blur, exposure
from .config import Config
from .exif_io import Exif
from .pair import PairResult, run_pair
from .preview import load_preview
from .snapshot import Progress, _Meter

ANALYSERS = (exposure, blur)
REPORT_DIRNAME = "reports"


@dataclass
class CaptureScore:
    capture_id: str
    status: str
    folder: str
    source: str                         # jpg | heic | nef
    source_path: str
    metrics: dict[str, float] = field(default_factory=dict)
    sharpness_z: float | None = None
    reasons: list[tuple[str, bool]] = field(default_factory=list)   # (reason, extreme)


@dataclass
class AnalyzeResult:
    pair: PairResult | None = None
    scores: list[CaptureScore] = field(default_factory=list)
    computed: int = 0
    cached: int = 0
    errors: list[str] = field(default_factory=list)
    report_path: Path | None = None
    blur_skipped: bool = False          # too few captures to compare sharpness
    no_detail: list[str] = field(default_factory=list)   # too dark to judge sharpness


def _version(analyser, long_edge: int, source_key: str) -> str:
    return f"{analyser.VERSION}|{long_edge}|{source_key}"


def _cached(conn: sqlite3.Connection, cid: str, version: str, metrics) -> dict[str, float] | None:
    rows = dict(conn.execute(
        "SELECT metric, value FROM scores WHERE capture_id=? AND model_version=?",
        (cid, version)).fetchall())
    return rows if all(m in rows for m in metrics) else None


def _measure(root: Path, row: dict, long_edge: int) -> dict[str, float]:
    preview = load_preview(root, row["jpg_path"], row["nef_path"], row["heic_path"],
                           nef_orientation=row["orientation"], long_edge=long_edge)
    out: dict[str, float] = {}
    for a in ANALYSERS:
        out.update(a.measure(preview.rgb))
    return out


def run_analyze(root: Path, cfg: Config, conn: sqlite3.Connection, exif: Exif,
                out: Progress = print, workers: int | None = None) -> AnalyzeResult:
    root = Path(root).resolve()
    res = AnalyzeResult(pair=run_pair(root, cfg, conn, exif, out))
    long_edge = cfg.analysis_long_edge

    rows = [dict(r) for r in conn.execute(
        "SELECT p.capture_id, p.status, p.folder, p.nef_path, p.jpg_path, p.heic_path,"
        "       f.image_hash, f.size, f.mtime_ns, m.data AS meta"
        " FROM photos p"
        " JOIN files f ON f.path = COALESCE(p.jpg_path, p.heic_path, p.nef_path)"
        " LEFT JOIN file_meta m ON m.path = p.nef_path"
        " WHERE p.quarantined = 0 ORDER BY p.capture_id")]

    todo = []
    for r in rows:
        r["orientation"] = json.loads(r["meta"]).get("Orientation") if r["meta"] else None
        source_key = r["image_hash"][:16] if r["image_hash"] else f"{r['size']}-{r['mtime_ns']}"
        r["versions"] = {a: _version(a, long_edge, source_key) for a in ANALYSERS}
        src = "jpg" if r["jpg_path"] else "heic" if r["heic_path"] else "nef"
        cs = CaptureScore(r["capture_id"], r["status"], r["folder"] or ".", src,
                          r["jpg_path"] or r["heic_path"] or r["nef_path"])
        cached = {}
        for a, version in r["versions"].items():
            got = _cached(conn, r["capture_id"], version, a.METRICS)
            if got is None:
                cached = None
                break
            cached.update(got)
        if cached is not None:
            cs.metrics = cached
            res.cached += 1
        else:
            todo.append((r, cs))
        res.scores.append(cs)

    if todo:
        meter = _Meter("analyze", len(todo), sum(r["size"] for r, _ in todo), out)
        with ThreadPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 2)) as pool:
            futures = {pool.submit(_measure, root, r, long_edge): (r, cs) for r, cs in todo}
            for fut in as_completed(futures):
                r, cs = futures[fut]
                meter.add(r["size"])
                try:
                    cs.metrics = fut.result()
                except Exception as err:      # one bad file must not stop the run
                    res.errors.append(f"{r['capture_id']}: {err}")
                    continue
                with conn:
                    for a, version in r["versions"].items():
                        conn.execute("DELETE FROM scores WHERE capture_id=? AND model_version LIKE ?",
                                     (r["capture_id"], a.VERSION.split("/")[0] + "/%"))
                        conn.executemany(
                            "INSERT INTO scores (capture_id, metric, value, model_version)"
                            " VALUES (?, ?, ?, ?)",
                            [(r["capture_id"], m, cs.metrics[m], version) for m in a.METRICS])
                res.computed += 1
        meter.done()

    _judge(res, cfg)
    res.report_path = _write_report(root, res)
    return res


def _judge(res: AnalyzeResult, cfg: Config) -> None:
    """Apply thresholds. Blur is relative: Milestone 1 compares against all captures."""
    te, tb = cfg.thresholds["exposure"], cfg.thresholds["blur"]
    scored = [s for s in res.scores if s.metrics]
    for s in scored:
        s.reasons = exposure.reasons(s.metrics, te)
    # Near-black frames carry no detail: sharpness is meaningless and would skew the z-scores.
    res.no_detail = sorted(s.source_path for s in scored if s.metrics["p99"] < tb["min_p99"])
    scored = [s for s in scored if s.metrics["p99"] >= tb["min_p99"]]
    if len(scored) < tb["min_group"]:
        res.blur_skipped = True
        return
    z = blur.sharpness_z({s.capture_id: s.metrics["lap_norm"] for s in scored},
                         {s.capture_id: s.metrics["ten_norm"] for s in scored})
    for s in scored:
        s.sharpness_z = z.get(s.capture_id)
        if s.sharpness_z is not None and s.sharpness_z < tb["z"]:
            s.reasons.append(("blur", False))


def _write_report(root: Path, res: AnalyzeResult) -> Path:
    folder = root / manifest.STATE_DIRNAME / REPORT_DIRNAME
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    path = folder / f"analyze-{stamp}.tsv"
    metric_names = [m for a in ANALYSERS for m in a.METRICS]
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["capture_id", "status", "folder", "source_path", *metric_names,
                    "sharpness_z", "reasons"])
        for s in sorted(res.scores, key=lambda s: (s.folder, s.source_path)):
            w.writerow([s.capture_id, s.status, s.folder, s.source_path,
                        *[f"{s.metrics[m]:.6g}" if m in s.metrics else "" for m in metric_names],
                        f"{s.sharpness_z:.2f}" if s.sharpness_z is not None else "",
                        ",".join(r + ("!" if extreme else "") for r, extreme in s.reasons)])
    os.replace(tmp, path)
    return path
