"""Sharpness metrics (README: Stage 3 -> Blur and shake).

Raw sharpness numbers depend heavily on the subject (a foggy landscape scores low
even when perfectly sharp), so blur is judged *relative to other frames*: z-scores
of log sharpness. Milestone 1 compares against the whole run; Milestone 3 switches
to bursts (>= 3 frames), then scenes, with global stats as the last fallback.

Raw gradient energy also scales with contrast, so a dark but sharp frame looks
"blurry". The z-scores therefore use contrast-normalised values (divided by the
frame's luminance variance); the raw values are kept for calibration.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

VERSION = "blur/2"
METRICS = ("lap_var", "tenengrad", "lap_norm", "ten_norm")


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def measure(rgb: np.ndarray) -> dict[str, float]:
    g = _gray(rgb)
    lap_var = float(cv2.Laplacian(g, cv2.CV_64F).var())
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx * gx + gy * gy))
    contrast = max(float(g.astype(np.float64).var()), 1e-6)
    return {"lap_var": lap_var, "tenengrad": tenengrad,
            "lap_norm": lap_var / contrast, "ten_norm": tenengrad / contrast}


def zscores(values: dict[str, float]) -> dict[str, float]:
    """z-score of log(value) per key; needs >= 3 values with some spread, else {}."""
    logs = {k: math.log(max(v, 1e-6)) for k, v in values.items()}
    if len(logs) < 3:
        return {}
    xs = list(logs.values())
    mean = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))
    if sd < 1e-9:
        return {k: 0.0 for k in logs}
    return {k: (x - mean) / sd for k, x in logs.items()}


def sharpness_z(lap: dict[str, float], ten: dict[str, float]) -> dict[str, float]:
    """Combined z: mean of the Laplacian and Tenengrad z-scores (they mostly agree)."""
    zl, zt = zscores(lap), zscores(ten)
    return {k: (zl[k] + zt[k]) / 2 for k in zl if k in zt}
