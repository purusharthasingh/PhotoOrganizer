# PLAN.md — Photo Culling Pipeline

> **README.md is the authoritative reference.** This file is a compact implementation summary of it. If the two disagree, the README wins: fix this file.

## Context

- Library: Nikon D780 RAW+JPEG pairs (.NEF + .JPG) plus phone photos (HEIC/JPEG). Hundreds of GB, under 0.5 TB, in **whatever folder layout already exists**.
- D780 shoots lossless compressed or compressed NEF (12/14-bit). It has no High Efficiency mode, so LibRaw, rawpy, digiKam and darktable can all decode the raws.
- Hardware: Windows PC with an RTX 5060 Ti (Blackwell, sm\_120). The GPU stage (M4) needs PyTorch ≥ 2.7 built for CUDA ≥ 12.8 (cu128 or newer, no higher than the driver's CUDA version). cu118/cu121/cu124 builds fail at runtime.
- Python: the project venv is Python 3.14. Keep it. Check wheel availability for `mediapipe`, `rawpy` and `pyiqa` when the milestone that needs them starts.
- Hobby/convenience project: free tools only.

## Goals

1. Group similar photos by time (bursts, events) and by subject.
2. Flag blurry, shaky and missed-focus photos.
3. Flag badly over- or under-exposed photos.
4. Bind each NEF+JPG pair so that a decision on one applies to both, and cleanup moves both.
5. Detect exact duplicate copies and keep one.

## Core design decisions

- **Work with any folder layout.** Never reorganize, never rename. The folder only hints at the scene.
- **Flag with XMP metadata.** JPG and HEIC get embedded XMP. NEF gets a `DSC_1234.NEF.xmp` sidecar (digiKam's default naming). **Raws are read-only.**
- **Encoding:**
  - Reject: `digiKam:PickLabel=1`, `xmp:Rating=-1`, keyword `cull|reject|<reason>`.
  - Pending/review: `PickLabel=2`, `xmp:Label=Yellow`, keyword `cull|pending|<reason>`.
  - Pick: `PickLabel=3` plus stars.
  - Reasons: `blur`, `shake`, `missed_focus`, `overexposed`, `underexposed`, `duplicate_copy`, `near_duplicate`, `not_best_in_burst`.
  - Group keywords: `burst|<id>`, `scene|<id>`, `dup|<id>`, `subject|<id>`, `dupcopy|<keeper capture_id>`.
  - Write each keyword to `XMP-lr:HierarchicalSubject` (`|`-separated), `XMP-digiKam:TagsList` (`/`-separated) and `XMP-dc:Subject` (leaf only). JPGs also get `EXIF:Rating` (0–5; write 0 for reject/pending). Confirm these formats with a digiKam round-trip in M1.
- **The SQLite manifest holds the pair binding, the file inventory and the score cache.** XMP is the source of truth for decisions. Human decisions read back from XMP override machine decisions.
- **Analyze previews, not raws:** the JPG, or the NEF's embedded preview (`JpgFromRaw`/`PreviewImage`). Decode with `rawpy` only as a last resort.
- **Never delete.** Rejected captures are moved to `_quarantine/`, which mirrors the library tree. Each move is logged to TSV with its sha256 and can be restored.
- **Pairs never split.** All keeper files of a `capture_id` are either in the library or in quarantine.
- **Advisory first.** Until thresholds are calibrated, scripts only write Pending (`PickLabel=2`). Destructive commands default to dry-run.
- **Hashes:**
  - `ImageDataHash` (SHA-256 of the image data only, from `exiftool -api ImageHashType=SHA256 -ImageDataHash`) identifies image content. It is stable when metadata changes, so it is used for duplicate detection, phone `capture_id`s and JPG/HEIC integrity checks against the Stage 0 manifest.
  - The full-file sha256 is used for backup and move verification. It **changes by design** on JPG/HEIC once embedded XMP is written, so it is never used as an identity.

## Tools

- digiKam 9.x: review UI (pick labels, Survey tool, light table). Set sidecars to read/write with `filename.ext.xmp` naming.
- exiftool (latest), on PATH, called through `pyexiftool` in `-stay_open` mode.
- Core Python (`requirements.txt`): `numpy`, `opencv-python`, `pillow`, `pillow-heif`, `pyexiftool`, `imagehash`, `rawpy` (fallback), `mediapipe` (M3+).
- GPU Python (`requirements-gpu.txt`, M4): `torch`/`torchvision` (cu128+, installed first), `pyiqa`, `open_clip_torch`, `scikit-learn>=1.3` (use `sklearn.cluster.HDBSCAN`, not the standalone package).
- Optional: Krokiet (cross-check for non-photo duplicates), darktable (editing picks only).
- Sanity check for M4: `torch.cuda.get_arch_list()` must include `sm_120`.

## Pipeline

0. **Backup:** 3-2-1 backups plus a manifest of every file, recording both the full sha256 and the `ImageDataHash`. Do this before anything else.

1. **Scan:** take any root folder and scan it recursively for NEF/JPG/JPEG/HEIC.
   - A capture's `folder` is the shared parent of its files, ignoring format subfolders named `Raw`, `RAW`, `NEF`, `JPG` or `JPEG` (configurable). So `Puerto Rico 26/Raw/DSC_9474.NEF` + `Puerto Rico 26/DSC_9474.JPG` → folder `Puerto Rico 26`.
   - Ordering and grouping come from capture metadata, never from paths.

2. **Pair and deduplicate.** Collect metadata with one recursive `exiftool -json -n` call:

   `FileName, Directory, DateTimeOriginal, SubSecTimeOriginal, OffsetTimeOriginal, Model, SerialNumber, ShutterCount, AFArea*, AFCoordinatesAvailable, FocusResult, Orientation, ImageWidth, ImageHeight, FileSize, FileModifyDate, ImageDataHash`

   Cache hashes by `(path, size, mtime)`. The order of steps is: scan → hash → duplicate groups → choose keepers → pair the keepers.

   **2a. Duplicate copies**
   - An *exact copy* has the same file type and the same `ImageDataHash`, and its capture metadata (DTO, SubSec, Serial, ShutterCount) matches where present.
   - A *variant* has the same capture metadata but different image data. Variants are reported, never marked.
   - Keeper rules, applied in order until one copy wins:
     1. Human decisions in XMP. If two copies carry *different* human decisions, this is a conflict: report it and mark nothing. The best-ranked copy stands in as keeper; the others get role `conflict`.
     2. A paired copy beats a lone copy.
     3. The copy in the fuller folder.
     4. Preferred roots from `config.toml`.
     5. The oldest mtime, then the shortest path.
   - Stage 2 records non-keepers as role `duplicate_copy` + `duplicate_of` in `files` (manifest only; no file is touched). Stage 5 labels each copy's own file with reject/`duplicate_copy` + `dupcopy|<keeper capture_id>` (pending in advisory mode). Copies share the keeper's capture_id, so this per-file label is not a `decisions` row.
   - If a duplicate carries metadata the keeper lacks, report it. Never merge metadata.
   - A folder whose every capture is an exact copy of a capture kept elsewhere is reported as a *duplicate folder*. Partial overlaps are reported with a percentage.
   - Duplicate copies never take part in pairing or analysis. They attach to the keeper's capture.

   **2b. Pairing** (keepers only)
   1. Index NEF/JPG files by stem (case-insensitive) across the whole scanned root. A partner can be in the same folder, a sibling folder or a subfolder.
   2. Confirm a pair only if DTO, SubSec, ShutterCount and SerialNumber all match. A field missing from both files is ignored. (ShutterCount is not the file number: in `Cheesecake/`, `DSC_0614` has ShutterCount 610.)
   3. If several files confirm, choose the nearest in the folder tree (same folder, then sibling/subfolder, then elsewhere). Report the others as role `variant` (attached to the capture, never analyzed or labeled). Never pair silently.
   4. If the stem matches but the capture data doesn't, these are separate captures. `check` reports them as a name collision, which is not an error.
   5. Statuses: `paired` (NEF + JPG), `nef_only` (analyzed from the embedded preview), `jpg_only`, `single` (phone HEIC/JPEG). A missing partner is normal, not an error. A JPG is a camera JPG if it has a `ShutterCount` or a Nikon `Make`. Any other JPG, and every HEIC, is `single`.
   6. `capture_id`:
      - Camera: `SerialNumber_ShutterCount_<YYYYMMDD-HHMMSS.ss>`, built from whichever file exists.
      - Phone (no shutter count): `Model_<YYYYMMDD-HHMMSS.ss>_<first 8 chars of ImageDataHash>`, with spaces removed from the model name.
      - The phone form is also the fallback for a camera file missing Serial, ShutterCount or SubSec. `check` warns about these.
   7. Re-scans are stable. A late partner upgrades `nef_only`/`jpg_only` to `paired` under the same `capture_id`, and a lost partner downgrades it. Existing decisions are kept.

3. **Analyze.** Results are cached in `scores`, keyed by `(capture_id, metric, model_version)`. Bump `model_version` or the config hash to recompute.
   - **Exposure (CPU):** luminance Y on 0–255. Record `clip_hi` (fraction ≥ 250), `clip_lo` (fraction ≤ 5), mean, p1, p50 and p99.
     - Overexposed: `clip_hi > 0.05 and p50 > 170`.
     - Underexposed: `(mean < 50 and p99 < 180) or clip_lo > 0.30`.
     - Captures with a NEF (`paired`/`nef_only`) have raw headroom, so they are flagged for review only. Auto-reject applies only in extreme cases (`clip_hi > 0.25`).
   - **Blur and shake (CPU):** resize to about 2048 px on the long edge, then compute Laplacian variance and Tenengrad, and both divided by luminance variance (`lap_norm`, `ten_norm`). Judge on the normalised values (z of log, mean of both), so dark-but-sharp frames aren't called blurry. Near-black frames (p99 < 20) get no blur verdict and are left out of the comparison. Whole-frame scores can't tell bokeh from a miss; the AF-point check (M3) fixes that.
     - Score relative to the burst with z-scores (flag at z < −1.5) **only when the burst has 3 or more frames**.
     - Otherwise fall back to scene-level z-scores. If the scene also has fewer than 3 frames, use global percentiles.
     - M1 uses global stats only, until bursts exist (M3).
     - Detect shake with FFT or gradient anisotropy, since motion blur is directional.
   - **Missed focus:**
     - AF area source, in order: (1) pixel coordinates when `AFCoordinatesAvailable` = Yes (live view); (2) named viewfinder points when it is No (phase detect, as on every current sample photo): `PrimaryAFPoint` / `AFPointsUsed` on the `FocusPointSchema` grid (D780: `51-point`, rows A–E x columns 1–11, C6 = centre), mapped by a per-schema table to approximate frame boxes; (3) the fallback below. Map the AF box to JPG pixels (scaling and orientation), then compute `focus_ratio = sharpness(AF box) / sharpness(frame)`. A ratio below about 1 suggests missed focus.
     - Validate first by drawing AF boxes on 20 images (`tools/draw_af_boxes.py`).
     - Fallback: mediapipe face/object detection, then sharpness inside the detected region.
   - **Learned quality (GPU, M4):** pyiqa `topiq_nr` in fp16, batches of 32–128.

4. **Group:**
   - **Bursts and scenes:**
     - Sort by clock-offset-corrected capture time.
     - Bursts are per camera serial; a gap over 2 s starts a new burst.
     - Scenes span all cameras and phones. A gap over 30–60 min starts a new scene, and so does a folder change (configurable).
   - **Near-duplicates:** `imagehash.phash`, with Hamming distance ≤ 6–8.
   - **Subject clusters (GPU):** open\_clip or DINOv2 embeddings, then HDBSCAN within each scene (`min_cluster_size=3`, cosine).
   - **Best-in-burst:**
     - Weighted z-score of subject sharpness, IQA, an optional aesthetic score (e.g. `laion_aes`) and eyes-open (only frames with faces).
     - A missing component gets weight 0.
     - Keep the top 1–2 frames and mark the rest Pending (`not_best_in_burst`).
   - **Group IDs** are deterministic, human-readable and never counters or UUIDs:

     | Group | Format | Example |
     |---|---|---|
     | Burst | `<YYYYMMDD-HHMMSS.ss first frame>-<model, no spaces>` | `burst\|20201205-204145.96-D780` |
     | Scene | `<YYYYMMDD-HHMM first frame>-<folder slug>` | `scene\|20251029-1139-puerto-rico-26` |
     | Near-dup set | `capture_id` of earliest member | `dup\|3004822_614_20201205-204145.96` |
     | Subject | `<scene_id>/<n>`, n ordered by each cluster's earliest frame | `subject\|20251029-1139-puerto-rico-26/2` |
     | Duplicate copy | keeper's `capture_id` | `dupcopy\|3004822_614_20201205-204145.96` |

     - Folder slug: lowercase, with spaces and punctuation replaced by `-`.
     - If two IDs ever collide, add a suffix `-b`, `-c`.
     - Reconcile with the previous run: a new group inherits an old ID if Jaccard overlap ≥ 0.5 (best match wins, each old ID used at most once). Only unmatched groups get fresh IDs.
     - The pipeline owns the `burst|`, `scene|`, `dup|`, `subject|` and `dupcopy|` namespaces. It rewrites them on every write and never touches other keywords.

5. **Flag:** compute one decision per `capture_id` and write it to every file that exists for that capture: the JPG/HEIC embedded XMP and/or the NEF sidecar. Advisory mode (M1–M4) writes only Pending plus reason keywords.
   - Dry run by default; `--apply` writes, after taking a snapshot.
   - Writes the diff between desired and current XMP only (idempotent), and removes its own stale labels.
   - Skips any capture with a human pick label, colour label or rating.
   - Reconciles only `cull|` and `dupcopy|` in M1.
   - Never writes raws.
   - Re-hashes the picture data of written JPG/HEIC files afterwards.
   - Duplicate copies get `cull|pending|duplicate_copy` + `dupcopy|<keeper>`; variants and conflicts get nothing.
   - No `EXIF:Rating` in advisory mode.

6. **Review in digiKam:**
   - Filter by Pick Label, using the Survey tool or the light table.
   - Run *Write Metadata to Files*.
   - `sync` re-reads the XMP into `decisions` with `source=human`.

7. **Cleanup:**
   - Dry-run by default; `--apply` actually moves files.
   - Move every keeper file of a rejected capture, plus its sidecars, to `_quarantine/<mirrored path>`.
   - Duplicate copies:
     - Before each move, verify that the keeper still exists and that its `ImageDataHash` still matches.
     - **Never quarantine the last remaining copy.**
     - A duplicate folder moves as one operation, but each file is still logged on its own line.
   - Logging and verification:
     - Append `timestamp  capture_id  src_path  dst_path  sha256  reason` to `_quarantine/moves.tsv`.
     - Verify the full sha256 before and after each move.
     - Integrity check: fail if any `capture_id` has *keeper* files on both sides.
   - `restore` reverses moves from `moves.tsv` and re-verifies the hashes.
   - `check` (run by hand; nightly scheduling optional):
     - Reports untracked and missing files, and refreshes pair statuses.
     - Reports name collisions, duplicate copies, keeper conflicts, and duplicate or overlapping folders.
     - Warns about paths over 240 characters.
     - Does **not** report `nef_only`/`jpg_only` as problems.
   - Empty the quarantine only after 30–90 days, once backups are confirmed.

## SQLite schema

```sql
CREATE TABLE photos (
  capture_id  TEXT PRIMARY KEY,
  nef_path    TEXT,                 -- keeper copies only
  jpg_path    TEXT,
  heic_path   TEXT,
  dto         TEXT,                 -- DTO + SubSec, clock-offset corrected
  model       TEXT,
  serial      TEXT,
  af_x REAL, af_y REAL, af_w REAL, af_h REAL,   -- AF box in JPG pixel space
  folder      TEXT,                 -- shared parent, format subfolders ignored
  status      TEXT,                 -- paired | nef_only | jpg_only | single
  quarantined INTEGER DEFAULT 0
);

CREATE TABLE files (                -- every file on disk, incl. duplicate copies and sidecars
  path         TEXT PRIMARY KEY,    -- relative to the scanned root, '/'-separated
  capture_id   TEXT,
  kind         TEXT,                -- nef | jpg | heic | xmp
  size         INTEGER,
  mtime_ns     INTEGER,             -- hashes are cleared when size or mtime change
  sha256       TEXT,                -- full-file hash (backup / move verification)
  image_hash   TEXT,                -- ImageDataHash (identity, dedupe)
  role         TEXT,                -- keeper | duplicate_copy | conflict | variant (NULL for sidecars)
  duplicate_of TEXT,                -- keeper path when role = duplicate_copy
  present      INTEGER,             -- 0 = missing on the last scan
  first_seen   TEXT,
  last_seen    TEXT
);

-- Cached exiftool metadata (capture fields + XMP decisions), valid while size/mtime match.
CREATE TABLE file_meta (
  path     TEXT PRIMARY KEY,
  size     INTEGER,
  mtime_ns INTEGER,
  data     TEXT                     -- exiftool JSON row
);

CREATE TABLE scores (
  capture_id    TEXT,
  metric        TEXT,
  value         REAL,
  model_version TEXT,
  PRIMARY KEY (capture_id, metric, model_version)
);

CREATE TABLE groups (
  capture_id TEXT PRIMARY KEY,
  burst_id   TEXT,
  scene_id   TEXT,
  dup_id     TEXT,
  subject_id TEXT
);

CREATE TABLE decisions (
  capture_id TEXT,
  source     TEXT,                  -- machine | human
  decision   TEXT,                  -- reject | pending | pick
  reason     TEXT,
  ts         TEXT,                  -- time of the latest change
  PRIMARY KEY (capture_id, source)
);
-- Effective decision = the human row if present, else the machine row.
```

## Project layout and CLI

```
cli.py                       # snapshot | verify | scan | pair | analyze | group | flag | sync | cleanup | restore | check
config.toml                  # in the project: library root, thresholds, clock offsets, format-subfolder names, preferred roots
<root>/.photoorganizer/      # per scanned root: manifest.sqlite, snapshots/<timestamp>.tsv
photo_organizer/
  config.py  manifest.py  exif_io.py  dedupe.py  pair.py  preview.py
  analyze/{exposure,blur,focus,iqa}.py
  group/{bursts,dupes,clusters,rank,ids}.py
  flag.py  sync_xmp.py  cleanup.py
tools/draw_af_boxes.py
tests/
```

## Build order

1. **M1: pairing, manifest and classical pass (advisory only)**
   - `config.toml` + loader.
   - `snapshot` (Stage 0 manifest: full sha256 + `ImageDataHash` per file) and `verify` (library or backup vs a snapshot: ok / labels only / sidecar edited / changed / missing / unreadable / extra).
   - Stage 1 `scan`: inventory into `files`, folder hint, new/changed/missing detection, path-length warnings.
   - `exif_io` (stay_open; batch read; XMP write to JPG/HEIC/NEF sidecar), plus a digiKam keyword round-trip test.
   - `manifest`, `dedupe`, `pair`.
   - Preview loader (JPG → HEIC via pillow-heif → NEF embedded preview via `rawpy.extract_thumb` → rawpy half-size decode; orientation).
   - `analyze`: exposure and global-stats blur, with score caching keyed by analyser version | preview size | source picture hash, and a TSV report per run.
   - `flag --advisory`.
   - Run on a **copy** (`sandbox/flag_test/`, git-ignored; `sample_photos/` stays unmodified) and check the result in digiKam.
   - Expected results on `sample_photos/`:
     - `Birb`, `Cheesecake`, `Spring Musings`: 14 captures, all `paired` in the same folder.
     - `Cheesecake/Also Cheesecake`: `DSC_0618`–`0620` NEF+JPG are exact copies of those in `Cheesecake/`. All 6 files are `duplicate_copy`, the keepers stay in `Cheesecake/`, and the folder is reported as a 100% duplicate folder.
     - `Puerto Rico 26`: `DSC_9474`–`9476` and `9480` are `paired` (NEF in `Raw/`), `DSC_9477`–`9479` are `jpg_only`, and all 7 have folder `Puerto Rico 26`.
     - Still to add (as copies inside `sample_photos/`): a sibling-folder pair, a `nef_only` capture, a stem collision, a JPG copy with edited metadata, a partly overlapping folder, and conflicting human labels.
2. **M2: calibration.**
   - Hand-label about 200 photos (the current sample is about 22 captures, so more are needed).
   - `sync`, then a precision/recall report for each flag.
   - Tune `config.toml`, favoring high precision for rejects.
3. **M3: bursts and focus.**
   - Bursts and scenes, then switch to burst-relative blur with the small-burst fallback.
   - Shake detection.
   - AF-box mapping + `draw_af_boxes`, then the focus ratio and the mediapipe fallback.
   - pHash near-dups.
   - Group IDs + reconciliation, and writing the group keywords.
4. **M4: GPU stage.** CUDA check, `topiq_nr`, CLIP/DINOv2 + HDBSCAN subject IDs, eyes-open, best-in-burst ranking.
5. **M5: cleanup.**
   - `cleanup` (dry-run/apply), `moves.tsv` + hash verification, integrity check, `restore`, `check`.
   - Auto-reject only high-confidence cases (`clip_hi > 0.25`, extreme blur outliers, exact duplicate copies).

## Gotchas

- Don't let darktable and digiKam write the same sidecar. Use darktable only for editing picks.
- Nikon's file counter rolls over (DSC\_9999 back to DSC\_0001), so never pair on filename alone.
- Phone and camera clocks drift. Store per-camera time offsets in `config.toml` and apply them before grouping.
- After flagging, JPG/HEIC full sha256 values no longer match the Stage 0 manifest. That is expected, so compare `ImageDataHash` instead.
- Long paths are unlikely to be a problem. Enabling Windows long-path support is optional insurance, and `check` warns over 240 characters.
- Develop and test every stage only in `sample_photos/`, on copies. Touch the real library only once a stage behaves correctly there.
