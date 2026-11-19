# PLAN.md — Photo Culling Pipeline

## Context

- Library: Nikon D780 RAW+JPEG pairs (.NEF + .JPG) plus phone photos (HEIC/JPEG). Hundreds of GB, under 0.5 TB.
- D780 shoots lossless compressed or compressed NEF (12/14-bit). It has no High Efficiency mode, so LibRaw, rawpy, digiKam and darktable can all decode the raws.
- Hardware: Windows PC with an RTX 5060 Ti (Blackwell, sm\_120). PyTorch must be 2.7 or newer, built for CUDA 12.8 or newer (cu128 index). Older cu118/cu121/cu124 builds will fail.
- Hobby/convenience project: free tools only.

## Goals

1. Group similar photos by time (bursts, events) and by subject.
2. Flag blurry, shaky and missed-focus photos.
3. Flag badly over- or under-exposed photos.
4. Bind each NEF+JPG pair so that a decision on one applies to both, and cleanup removes both.

## Core design decisions

- **Flag with XMP metadata, never by renaming.** JPG and HEIC get embedded XMP; NEF gets a `DSC_1234.NEF.xmp` sidecar (digiKam's default naming). Raws stay read-only.
- **Encoding:**
  - Reject: `digiKam:PickLabel=1`, `xmp:Rating=-1`, plus the keyword `cull|reject|<reason>` (e.g. blur, shake, overexposed).
  - Pending/review: `PickLabel=2`, `xmp:Label=Yellow`, plus a reason keyword.
  - Pick: `PickLabel=3` plus stars.
  - Groups: the keywords `burst|<id>` and `scene|<id>`.
  - Write each keyword to `lr:HierarchicalSubject`, `dc:Subject` and `digiKam:TagsList`. For JPGs, also write `EXIF:Rating` so Windows Explorer can see it.
- **The SQLite manifest is the pair binding and the score cache.** XMP is the source of truth for decisions.
- **Analyze the JPG (or the NEF's embedded preview), never the full raw.** This is much faster.
- **Never delete.** Rejected pairs get moved into `_quarantine/`, which mirrors the library's folder tree, with a TSV log.
- **Humans make the final call** in digiKam. Scripts only suggest decisions until the thresholds have been tuned.

## Tools

- digiKam 9.x: catalog, review, faces, Image Quality Scanner. Set sidecars to read/write.
- exiftool (latest), called through `pyexiftool` with `-stay_open` for batch speed.
- Python 3.11/3.12 with these packages: `torch`/`torchvision` (cu128), `opencv-python`, `numpy`, `pillow`, `pillow-heif`, `imagehash`, `pyiqa`, `open_clip_torch`, `hdbscan` (or scikit-learn's HDBSCAN), `rawpy` (fallback only), `mediapipe`.
- Krokiet (the Czkawka successor) for library-wide duplicate sweeps.
- Sanity check: `torch.cuda.get_arch_list()` must include `sm_120`.

## Pipeline

0. **Backup** (3-2-1 rule) plus a sha256 manifest of every file. Do this before anything else.
1. **Ingest:** use `YYYY/YYYY-MM-DD[_event]/` folders, with phone and camera photos in the same tree. Optionally rename both members of a pair to the same stem based on capture data (date-time, SubSec, ShutterCount).
2. **Pair:** run a single recursive `exiftool -json` call that collects FileName, Directory, DateTimeOriginal, SubSecTimeOriginal, OffsetTimeOriginal, Model, SerialNumber, ShutterCount, the AFArea\* fields, FocusResult, Orientation and ImageWidth/Height. Match on (dir, stem). Confirm the match by checking that DateTimeOriginal, SubSec and ShutterCount are equal. If they differ, set status to `name_collision`. Orphans (a NEF or JPG with no partner) get status `orphan` and are never deleted automatically. Build `capture_id = SerialNumber_ShutterCount_DTO+SubSec`.
3. **Analyze** (results cached in SQLite, keyed by capture\_id and model version):
   - Exposure (CPU): luminance Y; `clip_hi` = fraction of pixels ≥250; `clip_lo` = fraction ≤5; mean and p1/p50/p99. Starting thresholds:
     - Overexposed if clip\_hi > 0.05 and p50 > 170.
     - Underexposed if mean < 50 and p99 < 180, or if clip\_lo > 0.30.
     - Paired photos (which have raw headroom) only get flagged for review. Auto-reject only extreme cases (clip\_hi > 0.25).
   - Blur (CPU): variance of the Laplacian plus Tenengrad, at around 2048 px on the long edge. Compare within a burst (z-scores), never against one global threshold. Use FFT or gradient anisotropy to detect shake.
   - Missed focus: if `AFCoordinatesAvailable`, crop the AF-area box (scaled to the JPG and corrected for orientation) and compute sharpness inside the box divided by whole-frame sharpness. A ratio below about 1 suggests missed focus. Verify by drawing the AF box on 20 test images. Fallback for photos without AF data: face or object detection, then sharpness inside the detected region.
   - Learned quality (GPU, later): pyiqa `topiq_nr` in fp16, batches of 32–128.
4. **Group:**
   - Bursts: a time gap over 2 s starts a new burst; over 30–60 min starts a new scene/event.
   - Near-duplicates: `imagehash.phash`; Hamming distance ≤6–8 means near-duplicate.
   - Subject (GPU, later): open\_clip or DINOv2 embeddings, then HDBSCAN (min\_cluster\_size=3, cosine) within each event.
   - Best-in-burst: a weighted z-score of subject sharpness, IQA score, aesthetic score and eyes-open. Keep the top 1–2; mark the rest as pending.
5. **Flag:** compute one decision per capture\_id and write it to both pair members.
6. **Review in digiKam:** filter by Pick Label (Survey tool or light table). Write the metadata to files, then have the script re-read the XMP into the manifest so human decisions override machine scores.
7. **Cleanup:** dry-run by default. Move every member of a rejected pair, plus its sidecars, to `_quarantine/` (mirrored tree), log each move to TSV with its sha256, and verify the hashes afterwards. Integrity check: a capture\_id can never be split between the library and quarantine. Run a nightly orphan check. Empty the quarantine only after 30–90 days, once backups are confirmed.

## SQLite schema (starting point)

- `photos(capture_id PK, nef_path, jpg_path, heic_path, xmp_path, dto, model, af_x, af_y, af_w, af_h, sha256_nef, sha256_jpg, status)`
- `scores(capture_id, metric, value, model_version)`
- `groups(capture_id, burst_id, scene_id, dup_id)`
- `decisions(capture_id, decision, reason, source, ts)`

## Build order

1. **First script (next step):** pairing, the manifest, and the classical blur and exposure pass, writing only `PickLabel=2` (pending) plus reason keywords. Advisory mode only, with no moves.
2. Run it on a sample folder and calibrate the thresholds against about 200 hand-labeled photos.
3. Add burst grouping and the AF-box focus check.
4. Add the GPU stage: TOPIQ scoring, CLIP/DINOv2 clustering, best-in-burst ranking.
5. Add the quarantine cleanup script. Enable auto-reject only for high-confidence cases.

## Gotchas

- Don't let darktable and digiKam both write the same sidecar. Use darktable only for editing chosen picks.
- Nikon's file counter rolls over (DSC\_9999 back to DSC\_0001), so never pair on filename alone.
- Phone and camera clocks drift. Store a per-camera time offset.
- Enable Windows long-path support.
- Test everything on a copy of a sample folder first.
