"""Stage 2 tests: duplicate copies and pairing.

Edge cases are built in a temp folder from copies of sample files (sample_photos/
itself is never modified). Needs exiftool on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

from photo_organizer import manifest
from photo_organizer.config import PROJECT_DIR, Config
from photo_organizer.exif_io import Exif
from photo_organizer.metadata import Decision, FileMeta, human_decision, keys_agree
from photo_organizer.pair import make_capture_id, run_pair, time_stamp
from photo_organizer.dedupe import PicFile

SAMPLE = PROJECT_DIR / "sample_photos"
HAVE_EXIFTOOL = shutil.which("exiftool") is not None


def quiet(_msg: str) -> None:
    pass


def exiftool(*args: str) -> None:
    subprocess.run(["exiftool", "-q", "-overwrite_original", *args], check=True)


class PureFunctionTests(unittest.TestCase):
    def test_time_stamp_subsec(self):
        m = FileMeta("x", {"DateTimeOriginal": "2020:12:05 20:42:01", "SubSecTimeOriginal": "07"})
        self.assertEqual(time_stamp(m), "20201205-204201.07")
        m = FileMeta("x", {"DateTimeOriginal": "2020:12:05 20:42:01", "SubSecTimeOriginal": 5})
        self.assertEqual(time_stamp(m), "20201205-204201.50")   # '5' = .5 s
        m = FileMeta("x", {"DateTimeOriginal": "2020:12:05 20:42:01", "SubSecTimeOriginal": "123"})
        self.assertEqual(time_stamp(m), "20201205-204201.12")
        self.assertIsNone(time_stamp(FileMeta("x", {})))

    def test_keys_agree(self):
        self.assertTrue(keys_agree(("d", "1", 5, "s"), ("d", "1", 5, "s")))
        self.assertTrue(keys_agree(("d", None, 5, "s"), ("d", None, 5, "s")))   # missing from both
        self.assertFalse(keys_agree(("d", "1", 5, "s"), ("d", None, 5, "s")))   # missing from one
        self.assertFalse(keys_agree(("d", "1", 5, "s"), ("d", "1", 6, "s")))

    def test_capture_id(self):
        camera = FileMeta("x", {"DateTimeOriginal": "2020:12:05 20:41:45", "SubSecTimeOriginal": "96",
                                "SerialNumber": 3004822, "ShutterCount": 614, "Model": "NIKON D780"})
        pf = PicFile("x", "jpg", 1, 1, "ab" * 32, None, camera, Decision(), "")
        self.assertEqual(make_capture_id(pf), ("3004822_614_20201205-204145.96", False))
        phone = FileMeta("x", {"DateTimeOriginal": "2024:01:02 03:04:05", "SubSecTimeOriginal": "5",
                               "Model": "Pixel 8"})
        pf = PicFile("x", "jpg", 1, 1, "cd" * 32, None, phone, Decision(), "")
        self.assertEqual(make_capture_id(pf), ("Pixel8_20240102-030405.50_cdcdcdcd", True))

    def test_human_decision_ignores_pipeline_labels(self):
        machine = {"PickLabel": 2, "Label": "Yellow", "Rating": 0,
                   "HierarchicalSubject": ["cull|pending|blur", "burst|x"],
                   "TagsList": ["cull/pending/blur"], "Subject": ["blur"]}
        self.assertFalse(human_decision([machine]))
        human = dict(machine, PickLabel=3)
        self.assertEqual(human_decision([human]).pick, 3)
        tagged = {"HierarchicalSubject": ["People|Mom", "cull|pending|blur"],
                  "Subject": ["Mom", "blur"]}
        self.assertEqual(human_decision([tagged]).keywords, frozenset({"People|Mom"}))
        self.assertEqual(human_decision([{"Rating": 4}]).rating, 4)


@unittest.skipUnless(HAVE_EXIFTOOL and SAMPLE.is_dir(), "needs exiftool and sample_photos/")
class Stage2FixtureTests(unittest.TestCase):
    """Builds every README edge case in a temp library, runs Stage 2 once."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        lib = cls.lib = cls.tmp / "lib"
        src = SAMPLE / "Cheesecake"

        def cp(name: str, dest: str) -> Path:
            d = lib / dest
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / name, d)
            return d

        # Same-folder pair; sibling-folder pair (format subfolders).
        cp("DSC_0614.NEF", "A/DSC_0614.NEF")
        cp("DSC_0614.JPG", "A/DSC_0614.JPG")
        cp("DSC_0615.NEF", "Shoot/RAW/DSC_0615.NEF")
        cp("DSC_0615.JPG", "Shoot/JPG/DSC_0615.JPG")
        # nef_only, plus partial overlap: P holds a copy of B's NEF and one unique photo.
        cp("DSC_0616.NEF", "B/DSC_0616.NEF")
        cp("DSC_0621.NEF", "B/DSC_0621.NEF")
        cp("DSC_0616.NEF", "P/DSC_0616.NEF")
        Image.new("RGB", (40, 30), (9, 9, 9)).save(lib / "P" / "IMG_0002.JPG", "JPEG")
        # jpg_only, and a name collision: a different shot saved as DSC_0614.JPG.
        cp("DSC_0617.JPG", "C/DSC_0617.JPG")
        shutil.copy2(SAMPLE / "Spring Musings" / "DSC_0771.JPG", lib / "C" / "DSC_0614.JPG")
        # Exact copy whose metadata was edited by a person -> that copy becomes the keeper.
        cp("DSC_0618.JPG", "E/DSC_0618.JPG")
        cp("DSC_0618.JPG", "E2/DSC_0618.JPG")
        exiftool("-XMP-xmp:Rating=4", str(lib / "E2" / "DSC_0618.JPG"))
        # Conflict: two copies with different human pick labels.
        cp("DSC_0619.JPG", "F/DSC_0619.JPG")
        cp("DSC_0619.JPG", "G/DSC_0619.JPG")
        exiftool("-XMP-digiKam:PickLabel=3", str(lib / "F" / "DSC_0619.JPG"))
        exiftool("-XMP-digiKam:PickLabel=1", str(lib / "G" / "DSC_0619.JPG"))
        # Variant: an edited export with the same capture data but different pixels.
        cp("DSC_0620.NEF", "H/DSC_0620.NEF")
        cp("DSC_0620.JPG", "H/DSC_0620.JPG")
        (lib / "H" / "edit").mkdir()
        Image.new("RGB", (60, 40), (200, 100, 50)).save(lib / "H" / "edit" / "DSC_0620.JPG", "JPEG")
        exiftool("-tagsfromfile", str(src / "DSC_0620.JPG"), "-all:all",
                 str(lib / "H" / "edit" / "DSC_0620.JPG"))
        # Phone photo: no Nikon make, no shutter count -> single with a fallback id.
        (lib / "Phone").mkdir()
        Image.new("RGB", (40, 30), (1, 2, 3)).save(lib / "Phone" / "IMG_0001.JPG", "JPEG")
        exiftool("-DateTimeOriginal=2024:01:02 03:04:05", "-SubSecTimeOriginal=50",
                 "-Make=Google", "-Model=Pixel 8", str(lib / "Phone" / "IMG_0001.JPG"))
        # A sidecar for a NEF.
        (lib / "A" / "DSC_0614.NEF.xmp").write_text(
            '<x:xmpmeta xmlns:x="adobe:ns:meta/"/>', encoding="utf-8")

        cfg = Config(preferred_roots=("B",))
        conn = manifest.connect(lib)
        with Exif() as exif:
            cls.res = run_pair(lib, cfg, conn, exif, out=quiet)
        cls.files = {r["path"]: dict(r) for r in conn.execute("SELECT * FROM files")}
        cls.photos = {r["capture_id"]: dict(r) for r in conn.execute("SELECT * FROM photos")}
        conn.close()
        cls.by_lead = {c.lead.path: c for c in cls.res.captures}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def cap(self, path: str):
        cid = self.files[path]["capture_id"]
        return self.photos[cid]

    def test_same_folder_pair_and_sidecar(self):
        p = self.cap("A/DSC_0614.NEF")
        self.assertEqual((p["status"], p["jpg_path"], p["folder"]), ("paired", "A/DSC_0614.JPG", "A"))
        self.assertEqual(p["capture_id"], "3004822_610_20201205-203835.73")
        self.assertEqual(self.files["A/DSC_0614.NEF.xmp"]["capture_id"], p["capture_id"])

    def test_sibling_format_folders_pair(self):
        p = self.cap("Shoot/RAW/DSC_0615.NEF")
        self.assertEqual((p["status"], p["jpg_path"], p["folder"]),
                         ("paired", "Shoot/JPG/DSC_0615.JPG", "Shoot"))

    def test_nef_only_and_jpg_only(self):
        self.assertEqual(self.cap("B/DSC_0621.NEF")["status"], "nef_only")
        self.assertEqual(self.cap("C/DSC_0617.JPG")["status"], "jpg_only")

    def test_name_collision_is_separate_capture(self):
        p = self.cap("C/DSC_0614.JPG")
        self.assertEqual(p["status"], "jpg_only")
        self.assertNotEqual(p["capture_id"], self.cap("A/DSC_0614.NEF")["capture_id"])
        self.assertIn(["A/DSC_0614.JPG", "A/DSC_0614.NEF", "C/DSC_0614.JPG"], self.res.collisions)

    def test_partial_overlap_uses_preferred_root(self):
        self.assertEqual(self.files["P/DSC_0616.NEF"]["role"], "duplicate_copy")
        self.assertEqual(self.files["P/DSC_0616.NEF"]["duplicate_of"], "B/DSC_0616.NEF")
        report = {f: (n, total) for f, n, total, _ in self.res.folders}
        self.assertEqual(report["P"], (1, 2))

    def test_human_edited_copy_is_keeper(self):
        self.assertEqual(self.files["E2/DSC_0618.JPG"]["role"], "keeper")
        self.assertEqual(self.files["E/DSC_0618.JPG"]["role"], "duplicate_copy")
        g = [g for g in self.res.dedupe.groups if g.keeper.path == "E2/DSC_0618.JPG"][0]
        self.assertEqual(g.reason, "human decision")

    def test_conflict_marks_nothing(self):
        self.assertEqual(len(self.res.dedupe.conflicts), 1)
        roles = {self.files[p]["role"] for p in ("F/DSC_0619.JPG", "G/DSC_0619.JPG")}
        self.assertEqual(roles, {"keeper", "conflict"})
        self.assertNotIn("duplicate_copy", roles)

    def test_variant_attached_not_marked(self):
        cid = self.cap("H/DSC_0620.NEF")["capture_id"]
        self.assertEqual(self.cap("H/DSC_0620.NEF")["jpg_path"], "H/DSC_0620.JPG")
        row = self.files["H/edit/DSC_0620.JPG"]
        self.assertEqual((row["role"], row["capture_id"]), ("variant", cid))

    def test_phone_single_with_fallback_id(self):
        p = self.cap("Phone/IMG_0001.JPG")
        self.assertEqual(p["status"], "single")
        self.assertTrue(p["capture_id"].startswith("Pixel8_20240102-030405.50_"))
        self.assertIn(p["capture_id"], self.res.fallback_ids)

    def test_every_picture_has_a_capture(self):
        for path, row in self.files.items():
            self.assertIsNotNone(row["capture_id"], path)

    def test_status_counts(self):
        counts = Counter(c.status for c in self.res.captures)
        # A, Shoot, H paired; B/0616, B/0621 nef_only; C x2, E2, F jpg_only; P/IMG, Phone single
        self.assertEqual(counts, Counter(paired=3, nef_only=2, jpg_only=4, single=2))


@unittest.skipUnless(HAVE_EXIFTOOL and SAMPLE.is_dir(), "needs exiftool and sample_photos/")
class SampleExpectationTests(unittest.TestCase):
    """README Milestone 1 expected results, on a temp COPY of sample_photos/."""

    def test_expected_results(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            lib = tmp / "lib"
            shutil.copytree(SAMPLE, lib, ignore=shutil.ignore_patterns(".photoorganizer"))
            conn = manifest.connect(lib)
            with Exif() as exif:
                res = run_pair(lib, Config(), conn, exif, out=quiet)
            files = {r["path"]: dict(r) for r in conn.execute("SELECT * FROM files")}
            photos = {r["capture_id"]: dict(r) for r in conn.execute("SELECT * FROM photos")}
            conn.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        self.assertEqual(Counter(p["status"] for p in photos.values()),
                         Counter(paired=18, jpg_only=3))
        dups = sorted(p for p, r in files.items() if r["role"] == "duplicate_copy")
        self.assertEqual(len(dups), 6)
        self.assertTrue(all(p.startswith("Cheesecake/Also Cheesecake/") for p in dups))
        self.assertTrue(all(files[p]["duplicate_of"].startswith("Cheesecake/DSC_") for p in dups))
        self.assertIn(("Cheesecake/Also Cheesecake", 6, 6, [("Cheesecake", 6)]), res.folders)
        pr = [p for p in photos.values() if p["folder"] == "Puerto Rico 26"]
        self.assertEqual(Counter(p["status"] for p in pr), Counter(paired=4, jpg_only=3))
        self.assertEqual(photos["3004822_9571_20251029-113911.12"]["nef_path"],
                         "Puerto Rico 26/Raw/DSC_9474.NEF")


if __name__ == "__main__":
    unittest.main()
