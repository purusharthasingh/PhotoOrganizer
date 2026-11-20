"""Stage 3 tests: preview loading, exposure and blur metrics, score caching.

Synthetic images are generated in temp folders; sample_photos/ is only read.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from photo_organizer import manifest
from photo_organizer.analysis import run_analyze
from photo_organizer.analyze import blur, exposure
from photo_organizer.config import DEFAULT_THRESHOLDS, PROJECT_DIR, Config
from photo_organizer.exif_io import Exif
from photo_organizer.preview import load_preview

SAMPLE = PROJECT_DIR / "sample_photos"
HAVE_EXIFTOOL = shutil.which("exiftool") is not None
TE = DEFAULT_THRESHOLDS["exposure"]


def quiet(_msg: str) -> None:
    pass


def texture(seed: int = 0, size=(600, 900), sigma: float = 0.0, gain: float = 1.0,
            offset: float = 0.0) -> np.ndarray:
    """Detailed RGB test image; sigma > 0 blurs it, gain/offset change exposure."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (size[0] // 6, size[1] // 6), dtype=np.uint8)
    img = cv2.resize(base, (size[1], size[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    if sigma:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    img = np.clip(img * gain + offset, 0, 255).astype(np.uint8)
    return np.dstack([img, img, img])


class ExposureTests(unittest.TestCase):
    def test_metrics(self):
        m = exposure.measure(np.full((10, 10, 3), 255, np.uint8))
        self.assertEqual((m["clip_hi"], m["clip_lo"], m["p50"]), (1.0, 0.0, 255.0))
        m = exposure.measure(np.zeros((10, 10, 3), np.uint8))
        self.assertEqual((m["clip_lo"], m["mean"]), (1.0, 0.0))

    def test_luma_weights(self):
        green = np.zeros((4, 4, 3), np.uint8)
        green[..., 1] = 100
        self.assertAlmostEqual(exposure.measure(green)["mean"], 58.7, places=3)

    def test_reasons(self):
        ok = exposure.measure(texture())
        self.assertEqual(exposure.reasons(ok, TE), [])
        bright = exposure.measure(texture(gain=0.5, offset=200))
        self.assertEqual([r for r, _ in exposure.reasons(bright, TE)], ["overexposed"])
        burnt = exposure.measure(np.full((10, 10, 3), 255, np.uint8))
        self.assertEqual(exposure.reasons(burnt, TE), [("overexposed", True)])
        dark = exposure.measure(texture(gain=0.15))
        self.assertEqual(exposure.reasons(dark, TE), [("underexposed", False)])


class BlurTests(unittest.TestCase):
    def test_blur_lowers_sharpness(self):
        sharp, soft = blur.measure(texture()), blur.measure(texture(sigma=3))
        for k in blur.METRICS:
            self.assertGreater(sharp[k], soft[k], k)

    def test_normalised_ignores_exposure(self):
        bright, dark = blur.measure(texture()), blur.measure(texture(gain=0.2))
        self.assertGreater(bright["lap_var"] / dark["lap_var"], 10)   # raw: dark looks blurry
        self.assertAlmostEqual(bright["lap_norm"] / dark["lap_norm"], 1.0, delta=0.1)

    def test_zscores(self):
        self.assertEqual(blur.zscores({"a": 1, "b": 2}), {})
        z = blur.zscores({"a": 10, "b": 10, "c": 10})
        self.assertEqual(z, {"a": 0.0, "b": 0.0, "c": 0.0})
        z = blur.zscores({"a": 100, "b": 100, "c": 100, "d": 1})
        self.assertLess(z["d"], -1.4)


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_jpg_shrunk_and_rotated(self):
        im = Image.fromarray(texture(size=(300, 600)))
        exif = im.getexif()
        exif[0x0112] = 6                       # rotate 90 CW to display
        im.save(self.tmp / "r.jpg", exif=exif)
        p = load_preview(self.tmp, "r.jpg", None, None, long_edge=200)
        self.assertEqual(p.source, "jpg")
        self.assertEqual(p.rgb.shape, (200, 100, 3))   # portrait after rotation

    @unittest.skipUnless((SAMPLE / "Birb" / "DSC_0770.NEF").exists(), "sample NEF missing")
    def test_nef_embedded_preview(self):
        p = load_preview(SAMPLE, None, "Birb/DSC_0770.NEF", None, nef_orientation=1)
        self.assertEqual(p.source, "nef_preview")
        self.assertEqual(max(p.rgb.shape[:2]), 2048)

    def test_missing_everything(self):
        from photo_organizer.preview import PreviewError
        with self.assertRaises(PreviewError):
            load_preview(self.tmp, "nope.jpg", None, None)


@unittest.skipUnless(HAVE_EXIFTOOL, "exiftool not on PATH")
class AnalyzeRunTests(unittest.TestCase):
    """End to end on a generated library: flags, caching, cache survives label writes."""

    @classmethod
    def setUpClass(cls):
        cls.exif = Exif()

    @classmethod
    def tearDownClass(cls):
        cls.exif.close()

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.lib = self.tmp / "lib"
        (self.lib / "Shoot").mkdir(parents=True)
        for i in range(6):
            Image.fromarray(texture(seed=i)).save(self.lib / "Shoot" / f"IMG_{i}.JPG", quality=95)
        Image.fromarray(texture(seed=9, sigma=4)).save(self.lib / "Shoot" / "IMG_soft.JPG", quality=95)
        Image.fromarray(texture(seed=8, gain=0.1)).save(self.lib / "Shoot" / "IMG_dark.JPG", quality=95)
        Image.fromarray(np.zeros((300, 400, 3), np.uint8)).save(self.lib / "Shoot" / "IMG_black.JPG")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_it(self):
        conn = manifest.connect(self.lib)
        try:
            return run_analyze(self.lib, Config(), conn, self.exif, out=quiet, workers=2)
        finally:
            conn.close()

    def test_flags_and_cache(self):
        r = self.run_it()
        self.assertEqual((r.computed, r.cached, r.errors), (9, 0, []))
        why = {s.source_path.rsplit("/", 1)[-1]: [x for x, _ in s.reasons] for s in r.scores}
        self.assertEqual(why["IMG_soft.JPG"], ["blur"])
        self.assertEqual(why["IMG_dark.JPG"], ["underexposed"])      # dark, but not blurry
        self.assertEqual(why["IMG_black.JPG"], ["underexposed"])
        self.assertEqual(r.no_detail, ["Shoot/IMG_black.JPG"])
        for i in range(6):
            self.assertEqual(why[f"IMG_{i}.JPG"], [])
        self.assertTrue(r.report_path.exists())

        self.assertEqual((self.run_it().computed, self.run_it().cached), (0, 9))

        # Writing a label changes the file but not the picture: still cached.
        subprocess.run(["exiftool", "-q", "-overwrite_original", "-XMP-digiKam:PickLabel=2",
                        str(self.lib / "Shoot" / "IMG_0.JPG")], check=True)
        self.assertEqual(self.run_it().computed, 0)

        # A changed picture is recomputed.
        Image.fromarray(texture(seed=42)).save(self.lib / "Shoot" / "IMG_1.JPG", quality=95)
        self.assertEqual(self.run_it().computed, 1)


if __name__ == "__main__":
    unittest.main()
