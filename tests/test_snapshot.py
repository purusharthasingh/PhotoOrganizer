"""Stage 0 snapshot/verify tests. Needs exiftool on PATH and Pillow."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from photo_organizer import manifest
from photo_organizer.config import ScanConfig
from photo_organizer.exif_io import Exif
from photo_organizer.snapshot import latest_snapshot, read_snapshot, take_snapshot, verify

HAVE_EXIFTOOL = shutil.which("exiftool") is not None


def make_jpg(path: Path, color) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color).save(path, "JPEG", quality=90)


def quiet(_msg: str) -> None:
    pass


@unittest.skipUnless(HAVE_EXIFTOOL, "exiftool not on PATH")
class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exif = Exif()

    @classmethod
    def tearDownClass(cls):
        cls.exif.close()

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.lib = self.tmp / "lib"
        self.cfg = ScanConfig()
        make_jpg(self.lib / "A" / "a.JPG", (200, 10, 10))
        make_jpg(self.lib / "A" / "b.JPG", (10, 200, 10))
        make_jpg(self.lib / "B" / "c.JPG", (10, 10, 200))
        (self.lib / "A" / "a.JPG.xmp").write_text("<x/>")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def snap(self):
        conn = manifest.connect(self.lib)
        try:
            return take_snapshot(self.lib, self.cfg, conn, self.exif, out=quiet)
        finally:
            conn.close()

    def test_snapshot_contents_and_cache(self):
        r1 = self.snap()
        self.assertEqual((r1.files, r1.hashed_sha, r1.hashed_image), (4, 4, 3))
        entries = {e.path: e for e in read_snapshot(r1.path)}
        self.assertEqual(set(entries), {"A/a.JPG", "A/b.JPG", "B/c.JPG", "A/a.JPG.xmp"})
        self.assertEqual(len(entries["A/a.JPG"].sha256), 64)
        self.assertEqual(len(entries["A/a.JPG"].image_hash), 64)
        self.assertEqual(entries["A/a.JPG.xmp"].image_hash, "")
        self.assertNotEqual(entries["A/a.JPG"].image_hash, entries["A/b.JPG"].image_hash)

        r2 = self.snap()
        self.assertEqual((r2.hashed_sha, r2.hashed_image), (0, 0))   # all cached
        self.assertNotEqual(r1.path, r2.path)
        self.assertEqual(latest_snapshot(self.lib), r2.path)

    def test_verify_backup(self):
        snap = self.snap().path
        backup = self.tmp / "backup"
        shutil.copytree(self.lib, backup, ignore=shutil.ignore_patterns(".photoorganizer"))

        clean = verify(snap, backup, self.cfg, self.exif, out=quiet)
        self.assertEqual((len(clean.ok), clean.problems, clean.extra), (4, 0, []))

        # Label written into a JPG: picture identical -> metadata only.
        subprocess.run(["exiftool", "-q", "-overwrite_original", "-XMP-digiKam:PickLabel=2",
                        str(backup / "A" / "a.JPG")], check=True)
        # Picture replaced -> changed. File removed -> missing. New file -> extra.
        make_jpg(backup / "A" / "b.JPG", (0, 0, 0))
        (backup / "B" / "c.JPG").unlink()
        make_jpg(backup / "B" / "new.JPG", (1, 2, 3))
        (backup / "A" / "a.JPG.xmp").write_text("<edited/>")

        r = verify(snap, backup, self.cfg, self.exif, out=quiet)
        self.assertEqual(r.metadata_only, ["A/a.JPG"])
        self.assertEqual(r.changed, ["A/b.JPG"])
        self.assertEqual(r.missing, ["B/c.JPG"])
        self.assertEqual(r.sidecar_changed, ["A/a.JPG.xmp"])
        self.assertEqual(r.extra, ["B/new.JPG"])
        self.assertEqual(r.problems, 2)

    def test_changed_file_is_rehashed(self):
        self.snap()
        make_jpg(self.lib / "A" / "a.JPG", (0, 0, 0))
        r = self.snap()
        self.assertEqual((r.hashed_sha, r.hashed_image), (1, 1))


if __name__ == "__main__":
    unittest.main()
