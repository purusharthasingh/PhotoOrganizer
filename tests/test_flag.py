"""Stage 5 (advisory flag) tests. Everything runs on generated files or temp copies."""

from __future__ import annotations

import copy
import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from photo_organizer import manifest
from photo_organizer.config import PROJECT_DIR, Config
from photo_organizer.exif_io import Exif, ExifError
from photo_organizer.flag import run_flag
from tests.test_analyze import texture

SAMPLE = PROJECT_DIR / "sample_photos"
HAVE_EXIFTOOL = shutil.which("exiftool") is not None


def quiet(_msg: str) -> None:
    pass


def xmp(path: Path) -> dict:
    out = subprocess.run(["exiftool", "-j", "-XMP-digiKam:PickLabel", "-XMP-xmp:Label",
                          "-XMP-lr:HierarchicalSubject", "-XMP-dc:Subject",
                          "-XMP-digiKam:TagsList", str(path)],
                         capture_output=True, text=True, check=True).stdout
    import json
    row = json.loads(out)[0]
    for k in ("HierarchicalSubject", "Subject", "TagsList"):
        if isinstance(row.get(k), str):
            row[k] = [row[k]]
    return row


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(HAVE_EXIFTOOL, "exiftool not on PATH")
class FlagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exif = Exif()

    @classmethod
    def tearDownClass(cls):
        cls.exif.close()

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.lib = self.tmp / "lib"
        (self.lib / "S").mkdir(parents=True)
        for i in range(5):
            Image.fromarray(texture(seed=i)).save(self.lib / "S" / f"IMG_{i}.JPG", quality=95)
        Image.fromarray(texture(seed=7, sigma=4)).save(self.lib / "S" / "IMG_soft.JPG", quality=95)
        self.soft = self.lib / "S" / "IMG_soft.JPG"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def flag(self, apply=True, cfg=None):
        conn = manifest.connect(self.lib)
        try:
            return run_flag(self.lib, cfg or Config(), conn, self.exif, apply=apply, out=quiet)
        finally:
            conn.close()

    def test_dry_run_writes_nothing(self):
        before = {p: sha(p) for p in self.lib.rglob("*.JPG")}
        res = self.flag(apply=False)
        self.assertEqual([op.picture for op in res.ops], ["S/IMG_soft.JPG"])
        self.assertEqual({p: sha(p) for p in self.lib.rglob("*.JPG")}, before)
        self.assertFalse(res.applied)

    def test_apply_then_idempotent(self):
        res = self.flag()
        self.assertEqual((res.written, res.errors, res.picture_changed), (["S/IMG_soft.JPG"], [], []))
        self.assertTrue(res.snapshot_path.exists())
        x = xmp(self.soft)
        self.assertEqual((x["PickLabel"], x["Label"]), (2, "Yellow"))
        self.assertEqual(x["HierarchicalSubject"], ["cull|pending|blur"])
        self.assertEqual(x["TagsList"], ["cull/pending/blur"])
        self.assertEqual(x["Subject"], ["blur"])
        again = self.flag()
        self.assertEqual((again.ops, again.human_skipped), ([], []))

    def test_human_decision_wins(self):
        self.flag()
        subprocess.run(["exiftool", "-q", "-overwrite_original", "-XMP-digiKam:PickLabel=3",
                        str(self.soft)], check=True)
        before = sha(self.soft)
        res = self.flag()
        self.assertEqual(len(res.human_skipped), 1)
        self.assertEqual(res.ops, [])
        self.assertEqual(sha(self.soft), before)

    def test_stale_flag_is_removed_but_human_keywords_kept(self):
        self.flag()
        subprocess.run(["exiftool", "-q", "-overwrite_original",
                        "-XMP-lr:HierarchicalSubject+=People|Mom", "-XMP-dc:Subject+=Mom",
                        str(self.soft)], check=True)
        lenient = Config()
        lenient = copy.replace(lenient, thresholds={**lenient.thresholds,
                                                    "blur": {**lenient.thresholds["blur"], "z": -99.0}})
        res = self.flag(cfg=lenient)
        self.assertEqual([op.picture for op in res.ops], ["S/IMG_soft.JPG"])
        x = xmp(self.soft)
        self.assertNotIn("PickLabel", x)
        self.assertNotIn("Label", x)
        self.assertEqual(x["HierarchicalSubject"], ["People|Mom"])
        self.assertEqual(x["Subject"], ["Mom"])
        self.assertNotIn("TagsList", x)

    def test_write_refuses_raw(self):
        with self.assertRaises(ExifError):
            self.exif.write(self.tmp / "x.NEF", ["-XMP-xmp:Label=Red"])


@unittest.skipUnless(HAVE_EXIFTOOL and (SAMPLE / "Birb" / "DSC_0770.NEF").exists(),
                     "needs exiftool and sample NEF")
class NefSidecarTests(unittest.TestCase):
    def test_nef_gets_sidecar_and_is_never_written(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            lib = tmp / "lib"
            (lib / "B").mkdir(parents=True)
            for name in ("DSC_0770.NEF", "DSC_0770.JPG"):
                shutil.copy2(SAMPLE / "Birb" / name, lib / "B" / name)
            nef_sha = sha(lib / "B" / "DSC_0770.NEF")
            # Force an underexposed verdict so the pair gets flagged.
            cfg = Config()
            cfg = copy.replace(cfg, thresholds={**cfg.thresholds, "exposure": {
                **cfg.thresholds["exposure"], "under_clip_lo": -1.0}})
            conn = manifest.connect(lib)
            with Exif() as exif:
                res = run_flag(lib, cfg, conn, exif, apply=True, out=quiet)
            conn.close()
            self.assertEqual(sorted(res.written), ["B/DSC_0770.JPG", "B/DSC_0770.NEF.xmp"])
            self.assertEqual(sha(lib / "B" / "DSC_0770.NEF"), nef_sha)
            x = xmp(lib / "B" / "DSC_0770.NEF.xmp")
            self.assertEqual(x["HierarchicalSubject"], ["cull|pending|underexposed"])
            self.assertEqual(res.picture_changed, [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
