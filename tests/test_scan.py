"""Stage 1 scan tests. Run with: python -m unittest discover -s tests"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from photo_organizer import manifest
from photo_organizer.config import PROJECT_DIR, ScanConfig, load
from photo_organizer.scan import folder_hint, scan, shared_folder, walk

FMT = ScanConfig().format_subfolders
SAMPLE = PROJECT_DIR / "sample_photos"


def touch(root: Path, rel: str, data: bytes = b"x") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


class FolderHintTests(unittest.TestCase):
    def test_same_folder(self):
        self.assertEqual(folder_hint("Spring Musings/DSC_0771.NEF", FMT), "Spring Musings")

    def test_format_subfolder_dropped(self):
        self.assertEqual(folder_hint("Puerto Rico 26/Raw/DSC_9474.NEF", FMT), "Puerto Rico 26")
        self.assertEqual(folder_hint("Shoot/RAW/NEF/x.NEF", FMT), "Shoot")

    def test_format_name_only_dropped_at_end(self):
        self.assertEqual(folder_hint("Raw/Trip/x.NEF", FMT), "Raw/Trip")

    def test_root_file(self):
        self.assertEqual(folder_hint("x.JPG", FMT), "")

    def test_shared_folder(self):
        self.assertEqual(shared_folder(["Shoot/RAW/x.NEF", "Shoot/JPG/x.JPG"], FMT), "Shoot")
        self.assertEqual(shared_folder(["PR/Raw/x.NEF", "PR/x.JPG"], FMT), "PR")
        self.assertEqual(shared_folder(["A/x.NEF", "A/sub/x.JPG"], FMT), "A")
        self.assertEqual(shared_folder(["A/x.NEF", "B/x.JPG"], FMT), "")
        self.assertEqual(shared_folder(["Cheesecake/x.JPG"], FMT), "Cheesecake")

    def test_shared_folder_case_insensitive(self):
        self.assertEqual(shared_folder(["Trip/x.NEF", "trip/x.JPG"], FMT), "Trip")


class ScanTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.cfg = ScanConfig()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def run_scan(self, cfg=None):
        conn = manifest.connect(self.root)
        try:
            return scan(self.root, cfg or self.cfg, conn)
        finally:
            conn.close()

    def files_table(self):
        conn = manifest.connect(self.root)
        try:
            return {r["path"]: dict(r) for r in conn.execute("SELECT * FROM files")}
        finally:
            conn.close()

    def test_kinds_skips_and_ignored(self):
        touch(self.root, "A/DSC_1.NEF")
        touch(self.root, "A/DSC_1.jpg")
        touch(self.root, "A/DSC_1.NEF.xmp")
        touch(self.root, "Phone/IMG_1.HEIC")
        touch(self.root, "A/notes.txt")
        touch(self.root, "_quarantine/A/DSC_2.NEF")
        touch(self.root, "A/_Quarantine/DSC_3.NEF")   # skip is case-insensitive, any depth
        r = self.run_scan()
        kinds = {f.path: f.kind for f in r.files}
        self.assertEqual(kinds, {
            "A/DSC_1.NEF": "nef", "A/DSC_1.jpg": "jpg",
            "A/DSC_1.NEF.xmp": "xmp", "Phone/IMG_1.HEIC": "heic",
        })
        self.assertEqual(r.ignored[".txt"], 1)
        self.assertFalse(any(p.startswith(".photoorganizer") for p in kinds))

    def test_rescan_new_changed_missing_reappeared(self):
        a = touch(self.root, "A/a.JPG")
        b = touch(self.root, "A/b.JPG")
        r1 = self.run_scan()
        self.assertEqual(sorted(r1.new), ["A/a.JPG", "A/b.JPG"])

        r2 = self.run_scan()
        self.assertEqual(r2.new, [])
        self.assertEqual(sorted(r2.unchanged), ["A/a.JPG", "A/b.JPG"])

        # Pretend Stage 2 cached hashes, then change a and remove b.
        conn = manifest.connect(self.root)
        with conn:
            conn.execute("UPDATE files SET sha256='s', image_hash='i'")
        conn.close()
        a.write_bytes(b"longer content")
        st = a.stat()
        os.utime(a, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        moved = b.read_bytes()
        b.unlink()

        r3 = self.run_scan()
        self.assertEqual(r3.changed, ["A/a.JPG"])
        self.assertEqual(r3.missing, ["A/b.JPG"])
        rows = self.files_table()
        self.assertIsNone(rows["A/a.JPG"]["sha256"])
        self.assertIsNone(rows["A/a.JPG"]["image_hash"])
        self.assertEqual(rows["A/b.JPG"]["present"], 0)
        self.assertEqual(rows["A/b.JPG"]["sha256"], "s")   # kept for later reconciliation

        # Missing files are reported once, not on every scan.
        self.assertEqual(self.run_scan().missing, [])

        b.write_bytes(moved)
        r5 = self.run_scan()
        self.assertEqual(r5.reappeared, ["A/b.JPG"])
        self.assertEqual(self.files_table()["A/b.JPG"]["present"], 1)

    def test_long_path_warning(self):
        touch(self.root, "A/x.JPG")
        cfg = ScanConfig(path_warn_length=5)
        self.assertEqual(len(walk(self.root, cfg).long_paths), 1)
        self.assertEqual(walk(self.root, self.cfg).long_paths, [])

    def test_not_a_folder(self):
        with self.assertRaises(NotADirectoryError):
            walk(self.root / "nope", self.cfg)


class ConfigTests(unittest.TestCase):
    def test_project_config_loads(self):
        cfg = load()
        self.assertEqual(cfg.library_root, SAMPLE.resolve())
        self.assertEqual(cfg.scan.extensions[".jpeg"], "jpg")
        self.assertIn("raw", cfg.scan.format_subfolders)


@unittest.skipUnless(SAMPLE.is_dir(), "sample_photos/ not present")
class SampleLibraryTests(unittest.TestCase):
    """Read-only walk of the real sample set (no manifest written)."""

    def test_folder_hints(self):
        r = walk(SAMPLE, ScanConfig())
        hints = {folder_hint(f.path, FMT) for f in r.files}
        self.assertIn("Puerto Rico 26", hints)
        self.assertNotIn("Puerto Rico 26/Raw", hints)
        self.assertIn("Cheesecake/Also Cheesecake", hints)
        self.assertEqual(r.errors, [])
        self.assertEqual(r.long_paths, [])
        nef = {f.path for f in r.files if f.kind == "nef"}
        self.assertIn("Puerto Rico 26/Raw/DSC_9474.NEF", nef)


if __name__ == "__main__":
    unittest.main()
