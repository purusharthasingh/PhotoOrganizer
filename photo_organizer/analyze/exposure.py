"""Exposure metrics from luminance (README: Stage 3 -> Exposure)."""

from __future__ import annotations

import numpy as np

VERSION = "exposure/1"
METRICS = ("mean", "p1", "p50", "p99", "clip_hi", "clip_lo")


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luma Y on 0-255 (what a JPEG's own Y channel is)."""
    rgb = rgb.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def measure(rgb: np.ndarray) -> dict[str, float]:
    y = luminance(rgb)
    p1, p50, p99 = np.percentile(y, [1, 50, 99])
    return {
        "mean": float(y.mean()),
        "p1": float(p1), "p50": float(p50), "p99": float(p99),
        "clip_hi": float((y >= 250).mean()),
        "clip_lo": float((y <= 5).mean()),
    }


def reasons(m: dict[str, float], t: dict) -> list[tuple[str, bool]]:
    """[(reason, extreme)] for one capture. `extreme` marks auto-reject candidates for
    Milestone 5. Until then every flag is advisory (Pending)."""
    out = []
    if m["clip_hi"] > t["over_clip_hi"] and m["p50"] > t["over_p50"]:
        out.append(("overexposed", m["clip_hi"] > t["extreme_clip_hi"]))
    elif m["clip_hi"] > t["extreme_clip_hi"]:
        out.append(("overexposed", True))
    if (m["mean"] < t["under_mean"] and m["p99"] < t["under_p99"]) or m["clip_lo"] > t["under_clip_lo"]:
        out.append(("underexposed", False))
    return out
