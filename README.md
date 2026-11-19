# PhotoOrganizer — Photo Culling Pipeline

PhotoOrganizer is a local photo culling pipeline built from free tools. It works on a mixed library of **Nikon D780 RAW+JPEG pairs** (`.NEF` + `.JPG`) and **phone photos** (HEIC/JPEG). It does four things:

- groups similar shots (bursts, events, subjects)
- flags blurry, shaky, missed-focus and badly exposed frames
- writes those suggestions as **XMP metadata** you can review in digiKam
- moves rejected photos to a quarantine folder, never deleting anything and always keeping NEF+JPG pairs together

> **Status: pre-alpha.** The design is done (see [`PLAN.md`](PLAN.md)). No code exists yet.
> Work through the [Roadmap](#roadmap--build-checklist) in order.

---

## Contents

- [Goals](#goals)
- [Principles and safety guarantees](#principles-and-safety-guarantees)
- [Requirements](#requirements)
- [Setup](#setup)
- [Pipeline overview](#pipeline-overview)
- [Metadata encoding reference](#metadata-encoding-reference)
- [Default thresholds](#default-thresholds)
- [SQLite manifest](#sqlite-manifest)
- [Proposed project layout and CLI](#proposed-project-layout-and-cli)
- [Roadmap / build checklist](#roadmap--build-checklist)
- [Calibration workflow](#calibration-workflow)
- [Gotchas](#gotchas)
- [Backup and recovery](#backup-and-recovery)
- [Acknowledgements](#acknowledgements)

---

## Goals

1. **Group** similar photos by time (bursts, events) and by subject.
2. **Flag** blurry, shaky and missed-focus photos.
3. **Flag** badly over- or under-exposed photos.
4. **Bind** each NEF+JPG pair, so that a decision on one applies to both and cleanup removes both.

## Principles and safety guarantees

| Rule | What it means in practice |
|---|---|
| **Flag with metadata, never rename** | Decisions live in XMP. JPG and HEIC get embedded XMP. Each NEF gets a `DSC_1234.NEF.xmp` sidecar, which is digiKam's default naming. |
| **Raws are read-only** | No tool in this project ever writes to a `.NEF`. |
| **One copy of each image** | Exact duplicate copies (identical image data, wherever they are) are detected. One copy is kept and the others are marked `duplicate_copy`. See [Stage 2](#duplicate-copies). |
| **Never delete** | Rejected pairs are *moved* to `_quarantine/`. That folder mirrors the library tree, and a TSV log records each move with its sha256. |
| **Humans decide** | Scripts only *suggest* (Pending / Reject) until the thresholds are calibrated. You make the final call in digiKam. |
| **Safe by default** | Every destructive command defaults to `--dry-run`. Analysis starts in advisory mode, writing only `PickLabel=2`. |
| **Pairs never split** | One `capture_id` maps to every file of a capture. All of those files are either in the library or all in quarantine, never split between the two. |
| **XMP is the source of truth for decisions** | The SQLite manifest holds the pair bindings and cached scores. Human edits made in digiKam are read back and override machine scores. |
| **Analyze previews, not raws** | Analysis reads the JPG, or the NEF's embedded preview. It decodes the full raw (`rawpy`) only as a fallback. |

---

## Requirements

Legend: **Required** = needed from Milestone 1 · **Later** = needed only from the milestone named · **Optional** = not needed by the pipeline at all.
Items marked 🛠 are **installed manually**, outside `pip`. See [Manual installs checklist](#manual-installs-checklist).

### Hardware

| Item | Status | Notes |
|---|---|---|
| Windows PC | Required | Any PC that can run Python and exiftool. Milestones 1–3 use only the CPU. |
| NVIDIA GPU (RTX 5060 Ti, Blackwell `sm_120`) | Later (M4) | Only needed for IQA scoring and embeddings. |
| Free space on the **library's volume** | Later (M5) | Keeps `_quarantine/` on the same drive, so moves are instant renames and don't need double the space. |
| Backup media (external drive + offsite/cloud) | Required | For Stage 0. 3-2-1 backups need storage beyond the library drive. |

### Software

| Tool | Status | Version | Purpose |
|---|---|---|---|
| 🛠 Python | Required | 3.11 or 3.12 recommended | Runs the pipeline scripts. Newer versions (3.13+) are fine for the core packages. Before M3/M4, check that `mediapipe`, `rawpy` and `pyiqa` publish wheels for your Python version. If they don't, recreate the venv with 3.12. |
| 🛠 [exiftool](https://exiftool.org/) | Required | latest | Handles every metadata read and write, through `pyexiftool` in `-stay_open` mode. |
| 🛠 [digiKam](https://www.digikam.org/) | Required | 9.x | Review UI: pick labels, Survey tool, light table. You also use it to hand-label photos for calibration. |
| 🛠 NVIDIA driver | Later (M4) | must report CUDA ≥ 12.8 in `nvidia-smi` | Lets PyTorch use the GPU. The CUDA Toolkit is **not** needed, because PyTorch wheels bundle their own runtime. |
| PyTorch + torchvision (pip) | Later (M4) | ≥ 2.7, CUDA build `cu128` or newer | GPU scoring and embeddings. The cu118/cu121/cu124 builds do **not** support `sm_120`. |
| 🛠 Backup software | Required | any | Stage 0. Use whatever tool you like (e.g. Windows File History, FreeFileSync, Restic, or a cloud client). The pipeline doesn't depend on it. |
| 🛠 [Krokiet](https://github.com/qarmin/czkawka) | Optional | latest | Optional cross-check (the Czkawka successor). The pipeline detects exact duplicate copies itself (Stage 2) and near-duplicates (Stage 4). Krokiet is still handy for non-photo files. |
| 🛠 darktable | Optional | latest | Editing chosen picks only. See [Gotchas](#gotchas). |
| 🛠 git | Optional | any | Version control for the project code. |

### Python packages

The packages are split so Milestones 1–3 don't pull in the multi-GB GPU stack.

```text
# requirements.txt  (core, Milestones 1–3, CPU only)
numpy
opencv-python
pillow
pillow-heif        # HEIC phone photos
pyexiftool         # needs exiftool.exe on PATH
imagehash          # M3: near-duplicate pHash
rawpy              # optional fallback: decode the raw only when the NEF has no usable preview
mediapipe          # M3: face fallback for missed-focus; M4: eyes-open check
```

```text
# requirements-gpu.txt  (Milestone 4; install torch/torchvision first from a CUDA >= 12.8 index)
pyiqa              # topiq_nr quality score
open_clip_torch    # CLIP embeddings (or DINOv2 via torch.hub, no extra package)
scikit-learn       # >= 1.3, provides sklearn.cluster.HDBSCAN
```

> Use scikit-learn's built-in `HDBSCAN` rather than the standalone `hdbscan` package. On Windows, the standalone package sometimes has no prebuilt wheel, and building it from source needs the 🛠 Microsoft C++ Build Tools.

---

## Setup

### Manual installs checklist

These can't be installed through `pip`. Install them once per machine.

- [ ] **Python** (3.11/3.12 recommended), from python.org. Tick *Add python.exe to PATH*.
- [ ] **exiftool**: see [step 3](#3-exiftool).
- [ ] **digiKam 9.x**, from digikam.org. Then apply the [settings below](#5-digikam-settings).
- [ ] **Backup tooling + media** for Stage 0.
- [ ] *(M4)* **NVIDIA driver**, recent enough that `nvidia-smi` shows CUDA ≥ 12.8.
- [ ] *(Optional)* **Krokiet**, **darktable**, **git**.
- [ ] *(Only if a pip install fails while building a package)* **Microsoft C++ Build Tools**, for packages that have no Windows wheel.
- [ ] *(Usually already present)* **Microsoft Visual C++ Redistributable**. PyTorch and OpenCV need it. If `import torch` or `import cv2` fails with a DLL error, install the latest x64 redistributable.

### 1. Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # core: enough for Milestones 1–3
```

When you reach Milestone 4:

```powershell
# PyTorch for Blackwell (sm_120): use a CUDA build of 12.8 or newer.
# Pick the newest cuXXX build your PyTorch release offers (see pytorch.org/get-started),
# as long as it is <= the "CUDA Version" that `nvidia-smi` reports for your driver.
# Example (replace cu128 with e.g. cu130 / cu134 if available):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-gpu.txt
```

### 2. CUDA sanity check (Milestone 4)

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

The arch list **must include `sm_120`**. If it doesn't, you have the wrong torch build.

### 3. exiftool

1. Download the Windows package from exiftool.org.
2. Extract it. Rename `exiftool(-k).exe` to `exiftool.exe`, and keep the `exiftool_files` folder next to it.
3. Put that folder on `PATH`.
4. Verify with `exiftool -ver` in a new terminal.

### 4. Windows long paths (optional)

Most programs on Windows can't use paths longer than 260 characters. Typical paths such as `D:\Photos\Spring Musings\DSC_0771.NEF.xmp` are well under 100 characters, even under `_quarantine/`, so you're unlikely to hit the limit. Turn this on as cheap insurance in case your library root is deep, your event names are long, or restored or exported files have long names. Run it in an elevated PowerShell, then reboot:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
```

The setting only helps programs that support long paths, such as Python. Some tools can still fail on very long paths, so keeping paths short matters more. The `check` command warns about any path over 240 characters.

### 5. digiKam settings

- *Settings → Configure digiKam → Metadata → Behavior*: enable writing ratings, pick labels, color labels and tags **to sidecar files**. Use **read/write** sidecar mode.
- *Sidecars*: keep the default `filename.ext.xmp` naming, so the sidecar for `DSC_1234.NEF` is `DSC_1234.NEF.xmp`.
- After the pipeline writes metadata, use *Item → Reread Metadata From File* so digiKam picks it up.
- Optional: digiKam's own *Image Quality Scanner* and face recognition are useful for comparison during calibration, but the pipeline doesn't need them.

---

## Pipeline overview

```
 ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐
 │0 Backup  │──▶│1 Scan    │──▶│2 Pair    │──▶│3 Analyze  │──▶│4 Group   │
 │sha256 mfst│  │any layout│   │exiftool  │   │exposure,  │   │bursts,   │
 └──────────┘   └──────────┘   │→ SQLite  │   │blur, focus│   │dupes,    │
                               └──────────┘   │IQA (GPU)  │   │subjects  │
                                              └───────────┘   └────┬─────┘
                                                                   ▼
                 ┌────────────┐   ┌──────────────────┐   ┌──────────────┐
                 │7 Cleanup   │◀──│6 Review (digiKam)│◀──│5 Flag (XMP)  │
                 │_quarantine/│   │human decisions → │   │one decision  │
                 │TSV + sha256│   │re-read into DB   │   │per capture_id│
                 └────────────┘   └──────────────────┘   └──────────────┘
```

### Stage 0 — Backup

- Follow the 3-2-1 rule: 3 copies, on 2 kinds of media, with 1 offsite. **Do this before running anything else.**
- Take a **snapshot**: `python cli.py snapshot <root>`. It scans the root and writes `<root>/.photoorganizer/snapshots/<UTC time>.tsv`, with one line per file: `path, kind, size, mtime_ns, sha256, image_hash`.
  - `sha256` covers the whole file. `image_hash` is exiftool's `ImageDataHash` (SHA-256 of the picture data only), so it stays the same when labels are written into a JPG/HEIC.
  - Fingerprints are cached in the manifest, so later snapshots only re-read files whose size or mtime changed.
- **Verify** with `python cli.py verify <root>` (checks the library), or add `--against E:\Backup\Photos` to check a backup copy. It re-reads every file and reports:

  | Result | Meaning | Problem? |
  |---|---|---|
  | ok | whole file identical | no |
  | labels only | JPG/HEIC metadata changed, picture identical (e.g. pipeline or digiKam labels) | no |
  | sidecar edited | `.xmp` changed | no |
  | **changed** | picture data differs, or a NEF changed at all | **yes** |
  | **missing** | in the snapshot, not on disk | **yes** |
  | **unreadable** | could not be read | **yes** |
  | extra | on disk, not in the snapshot | no (reported) |

  It exits with code 1 if there are any problems. Take a fresh snapshot before each pipeline stage that writes files, and verify your backup against it.

### Stage 1 — Scan (any folder layout)

The pipeline **works with whatever folder structure you already have**. It never reorganizes or renames your library. For example, all of these are valid:

```
sample_photos/Spring Musings/DSC_0771.NEF        event-named folders (current layout)
Photos/2021/2021-04-15/DSC_0771.NEF        date folders
Photos/Phone/Camera/PXL_20210415_...jpg    phone dumps, nested to any depth
```

- Point the CLI at any root folder. It scans recursively.
- Grouping and ordering come from **capture metadata** (DateTimeOriginal, SubSec, clock-offset corrections), never from folder names or dates in paths.
- Each capture's folder is recorded in the manifest as a `folder` hint. By default a new scene starts whenever the folder changes (configurable), because your folders usually reflect events you've already grouped by hand.
  - A capture's folder is the **shared parent folder of its files**, ignoring format subfolders named like `Raw`, `RAW`, `NEF`, `JPG` or `JPEG` (the list is configurable). That way a NEF in `Puerto Rico 26/Raw/` and a JPG in `Puerto Rico 26/` count as the same event, `Puerto Rico 26`.
- `_quarantine/` mirrors whatever layout exists, so restoring puts files back exactly where they were.
- **Not planned:** reorganizing into a date tree or renaming files to capture-based stems. It isn't needed for any stage, and it would conflict with "flag, never rename". If you ever want it, it would be a separate, opt-in, dry-run-first tool.

### Stage 2 — Pair and deduplicate

One recursive `exiftool -json` call collects these fields:

`FileName, Directory, DateTimeOriginal, SubSecTimeOriginal, OffsetTimeOriginal, Model, SerialNumber, ShutterCount, AFArea*, AFCoordinatesAvailable, FocusResult, Orientation, ImageWidth, ImageHeight`

```powershell
exiftool -r -json -n -ext NEF -ext JPG -ext JPEG -ext HEIC `
  -FileName -Directory -DateTimeOriginal -SubSecTimeOriginal -OffsetTimeOriginal `
  -Model -SerialNumber -ShutterCount "-AFArea*" -AFCoordinatesAvailable -FocusResult `
  -Orientation -ImageWidth -ImageHeight -FileSize -FileModifyDate `
  -api ImageHashType=SHA256 -ImageDataHash  D:\Photos\2024 > scan.json
```

#### What a pair is

- The two files of a pair **always share the same stem** (`DSC_0771.NEF` + `DSC_0771.JPG`). The stem is compared case-insensitively.
- They **don't have to be in the same folder.** The partner can sit in:
  - the same folder: `Spring Musings/DSC_0771.NEF` + `Spring Musings/DSC_0771.JPG`
  - a sibling folder: `Shoot/RAW/DSC_0771.NEF` + `Shoot/JPG/DSC_0771.JPG`
  - a subfolder: `Puerto Rico 26/DSC_9474.JPG` + `Puerto Rico 26/Raw/DSC_9474.NEF`
- **A matching name alone doesn't make a pair.** Nikon's file counter rolls over, so different shots can share a stem. A pair is only confirmed when the capture data matches.
- **A missing partner is normal.** One file of a pair can be gone, for example a JPG deleted after export, or a NEF that was never copied over. The remaining file is a complete, valid capture. It is not an error and it isn't flagged as one.

#### Pairing rules

1. **Index by stem.** Build a map from stem to every NEF/JPG with that stem anywhere under the scanned root.
2. **Confirm by capture data.** A NEF and a JPG with the same stem are paired only if `DateTimeOriginal`, `SubSecTimeOriginal`, `ShutterCount` and `SerialNumber` all match. Ignore any field that's missing from both files.
   - ShutterCount is not the file number. In `sample_photos/Cheesecake/`, `DSC_0614` has ShutterCount 610.
3. **Choose the nearest match.** Pairing runs on keeper copies only (see [Duplicate copies](#duplicate-copies)). If several files still confirm, for example *variants* with the same capture data, pair with the one closest in the folder tree: same folder, then sibling or subfolder, then anything else. Report the rest. Never pair them silently. The rest get role `variant`: they are attached to that capture but never analyzed or labeled.
4. **Same name, different shot.** If the stem matches but the capture data doesn't, the files are **not** paired. Each becomes its own capture, and `check` reports it as a name collision. It isn't treated as a problem with either file.
5. **Statuses:**

   | Status | Meaning | Treated as |
   |---|---|---|
   | `paired` | NEF + JPG confirmed | One capture, two files |
   | `nef_only` | No confirmed JPG partner | One capture, one file. Analyzed from the NEF's embedded preview. |
   | `jpg_only` | No confirmed NEF partner | One capture, one file |
   | `single` | Phone photo (HEIC/JPEG), never paired | One capture, one file |

   A JPG counts as a **camera** JPG (so it can be `paired` or `jpg_only`) if it has a `ShutterCount` or its `Make` is Nikon. Every other JPG, and every HEIC, is `single`.

   Every status flows through Analyze → Group → Flag → Review → Cleanup in the same way. A decision applies to all files that exist for the capture, whether that's one or two.
6. **`capture_id`** = `SerialNumber_ShutterCount_DTO+SubSec`. It is computed from whichever file exists, so a pair and a lone file are handled the same way.
   - Fallback for phones, which have no shutter count: `Model_DTO+SubSec_<first 8 chars of ImageDataHash>`. It uses the image-data hash, not the full-file sha256, because the full-file hash changes once XMP is written into the file.
   - The same fallback applies to a camera file missing `SerialNumber`, `ShutterCount` or SubSec (e.g. a JPG re-saved by an editor). `check` warns about every capture that needed it.
7. **Re-scans stay stable.** If a partner shows up later (copied back in, say), the next scan upgrades `nef_only`/`jpg_only` to `paired` under the same `capture_id`. If one file of a pair disappears, it downgrades the capture the same way. Existing decisions are kept.

#### Duplicate copies

The same file can turn up in several places, for example an import run twice, a backup copied back into the library, or a "To edit" folder. If two copies contain **exactly the same image**, keeping both makes no sense. One copy is kept and the others are marked as duplicates.

**Order of operations:** scan → hash → group duplicate copies → choose a keeper → pair the keepers (rules above). Duplicate copies never take part in pairing or analysis. They are attached to the keeper's capture as extra copies.

**How duplicates are detected.** Matching names or matching dates are never enough:

| Level | Test | Result |
|---|---|---|
| **Exact copy** | Same file type **and** the same `ImageDataHash`: a SHA-256 of the image data with all metadata excluded, from `exiftool -api ImageHashType=SHA256 -ImageDataHash`. The capture metadata (DTO, SubSec, Serial, ShutterCount) must match too, when present. | Marked as a duplicate |
| **Variant** | Same capture metadata but different image data, for example a resized export or an edited JPG | Reported only, never marked. These are different files. |
| **Near-duplicate** | Visually similar shots (pHash, Stage 4) | Handled in Stage 4 as similar shots, not copies |

Hashing only the image data means a copy still matches after its metadata changes. This matters because the pipeline itself writes XMP into JPGs, and digiKam may too. Full-file sha256 values are still recorded separately for backup and move verification. Hashes are cached by `(path, size, mtime)`, so the whole library is read in full only once.

**Choosing the keeper.** Rules are applied in order until one copy wins:

1. **Human decisions.** Keep the copy whose XMP or sidecar has ratings, pick labels or tags you set. If two copies carry *different* human decisions, it's a **conflict**: nothing is marked, and the case is reported for you to resolve. Until then the best-ranked copy stands in as the keeper for pairing and analysis, and the others get role `conflict`.
2. **Paired beats lone.** Keep the copy whose partner sits next to it. For a NEF, that's the copy beside its JPG.
3. **Fuller folder.** Keep the copy in the folder that holds more of that event's captures, rather than a stray copy.
4. **Preferred roots** (optional, in `config.toml`). For example, prefer `D:\Photos` over `D:\Imports`.
5. **Tie-break:** the oldest file modified time, then the shortest path.

**Marking duplicates.**
- The keeper is unchanged.
- Stage 2 only records the role `duplicate_copy` (and `duplicate_of`) in the manifest's `files` table. It never moves, renames or writes to any file.
- Stage 5 then labels each duplicate copy's own file: decision `reject` with reason `duplicate_copy` and the keyword `dupcopy|<keeper capture_id>`, or `pending` in advisory mode (Milestones 1–4). A copy shares the keeper's `capture_id`, so this per-file label is not stored in the `decisions` table.
- Exact copies are one of the few cases that can be auto-rejected in Milestone 5, because the image data is provably identical and the keeper is verified to exist.
- If a duplicate carries metadata the keeper lacks (keywords, GPS, rating), the report says so. Nothing is merged automatically.

**Duplicate folders.**
- If **every** capture in folder A is an exact copy of a capture kept elsewhere, folder A is reported as a **duplicate folder**.
- Cleanup can then quarantine the whole folder in one operation. Each file is still logged on its own line.
- Partial overlaps are reported with an overlap percentage, for example "`Imports/2025-10` is 85% duplicates of `Puerto Rico 26`". Only the overlapping files are marked.

### Stage 3 — Analyze

Results are cached in SQLite, keyed by `(capture_id, metric, model_version)`. Re-runs skip work that's already cached.

- **Exposure (CPU):**
  - Compute luminance Y on a 0–255 scale.
  - `clip_hi` = fraction of pixels ≥ 250. `clip_lo` = fraction of pixels ≤ 5.
  - Also record the mean, p1, p50 and p99.
- **Blur and shake (CPU):**
  - Resize to about 2048 px on the long edge.
  - Compute variance of the Laplacian and Tenengrad.
  - Score each frame **relative to its burst** (z-scores), never against a single global threshold.
  - Detect shake with FFT or gradient anisotropy: motion blur is directional.
- **Missed focus:**
  - Where the AF area comes from, in order:
    1. **Pixel coordinates** (`AFCoordinatesAvailable` = Yes; live view / contrast-detect): `AFAreaXPosition/YPosition/Width/Height`.
    2. **Named viewfinder points** (`AFCoordinatesAvailable` = No, `AFDetectionMethod` = Phase Detect). This is every photo in the current `sample_photos/` set. The camera records `PrimaryAFPoint` (e.g. `C6 (Center)`) and `AFPointsUsed` (e.g. `B5,C5,C6`) on the grid named by `FocusPointSchema` (`51-point` on the D780: rows A–E, columns 1–11, C6 = centre). A per-schema table maps each point to an approximate box in the frame. Use the primary point, or the bounding box of the points used.
    3. Otherwise, the face/object fallback below.
  - `FocusDistance` is recorded too and can serve as an extra signal.
  - With an AF box from 1 or 2:
    1. Scale the AF-area box to the JPG's dimensions and correct it for orientation.
    2. Compute `focus_ratio = sharpness(AF box) / sharpness(whole frame)`.
    3. A ratio below about 1 suggests missed focus.
  - **Validate** first by drawing the AF box on 20 test images.
  - Fallback for photos without AF data: detect a face or object (mediapipe), then measure sharpness inside the detected region.
- **Learned quality (GPU, Milestone 4):**
  - pyiqa `topiq_nr` in fp16, with batches of 32–128.

### Stage 4 — Group

- **Bursts and scenes:**
  - Sort by corrected capture time.
  - A gap of more than 2 s starts a new burst. Bursts are computed per camera.
  - A gap of more than 30–60 min starts a new scene/event. A change of folder also starts a new scene (configurable; see Stage 1).
- **Near-duplicates:** `imagehash.phash`. A Hamming distance of 6–8 or less counts as a near-duplicate.
- **Subject clusters (GPU):** open_clip or DINOv2 embeddings, then HDBSCAN within each scene (`min_cluster_size=3`, cosine distance).
- **Best-in-burst:**
  - Score each frame with a weighted z-score of subject sharpness, IQA score, aesthetic score and eyes-open.
  - The aesthetic score is optional (e.g. a pyiqa aesthetic metric such as `laion_aes`). Eyes-open applies only to frames with detected faces. A missing component gets weight 0.
  - Keep the top 1–2 frames. Mark the rest as Pending.

#### Group IDs

Group IDs are written into XMP keywords, so they must be **stable across re-runs**. Otherwise every run would leave stale keywords behind. They are never sequential counters or random UUIDs. Every ID is built from the data and is readable on its own:

| Group | ID format | Example (from `sample_photos/`) |
|---|---|---|
| Burst (per camera) | `<YYYYMMDD-HHMMSS.ss of first frame>-<camera model>` | `burst\|20201205-204145.96-D780` |
| Scene | `<YYYYMMDD-HHMM of first frame>-<folder slug>` | `scene\|20251029-1139-puerto-rico-26` |
| Near-duplicate set | `capture_id` of the earliest member | `dup\|3004822_614_20201205-204145.96` |
| Subject cluster | `<scene_id>/<n>`, with `n` numbered by each cluster's earliest frame (HDBSCAN labels themselves are arbitrary) | `subject\|20251029-1139-puerto-rico-26/2` |
| Duplicate copy | `capture_id` of the keeper | `dupcopy\|3004822_614_20201205-204145.96` |

Notes on the formats:
- Times are the clock-offset-corrected capture times.
- The camera model is the short model name with spaces removed. For a scene, the folder slug is the capture's `folder` (see Stage 1), lowercased, with spaces and punctuation replaced by `-`.
- Timestamps go down to hundredths of a second, so two different groups essentially never get the same ID. If they ever do, a suffix (`-b`, `-c`) is added.

**Rules that keep IDs stable:**
1. **Deterministic.** The same library and config always produce the same IDs. Run order and processing order don't affect them.
2. **Reconcile with the previous run.** When groups are recomputed, a new group inherits an existing ID if it shares at least half of that old group's members (Jaccard overlap ≥ 0.5, best match wins, each old ID used at most once). Only unmatched groups get a fresh ID. So adding a late photo at the start of a burst, or changing a clock offset slightly, doesn't rename the group.
3. **Owned namespaces.** On each run the pipeline rewrites every keyword under `burst|`, `scene|`, `dup|`, `subject|` and `dupcopy|` on each file it writes, which removes stale IDs. It **never** touches any other keyword.
4. **Per-camera bursts, cross-camera scenes.** Bursts are segmented per camera (serial number), so two cameras firing at the same moment form separate bursts. Scenes span all cameras and phones.

### Stage 5 — Flag

- Compute **one decision per `capture_id`** and write it to every file that exists for that capture: the JPG's embedded XMP and/or the NEF's `.NEF.xmp` sidecar. A `nef_only` or `jpg_only` capture gets just its one file written.
- In advisory mode (Milestones 1–4), the script writes only Pending plus reason keywords.

### Stage 6 — Review in digiKam

1. Filter by Pick Label, and review with the Survey tool or the light table.
2. Change labels as needed, then run *Write Metadata to Files*.
3. Run the `sync` step. It re-reads the XMP into the manifest's `decisions` table with `source=human`. Human decisions always override machine scores.

### Stage 7 — Cleanup

- Dry-run by default. `--apply` is required to actually move anything.
- Move **every member** of each rejected capture, including its sidecars, to `_quarantine/<mirrored path>`.
- **Duplicate copies:** move each `duplicate_copy` file and its sidecar to quarantine. Before each move, verify that the keeper still exists and that its `ImageDataHash` still matches. **Never quarantine the last remaining copy.** A duplicate folder is moved as one operation, with every file logged.
- Append one line per file to `_quarantine/moves.tsv`:

  ```
  timestamp	capture_id	src_path	dst_path	sha256	reason
  ```

- Verify the hashes after each move.
- **Integrity check:** fail loudly if any `capture_id` has *keeper* files in both the library and quarantine. Duplicate copies in quarantine while the keeper stays in the library are expected.
- Run the `check` command:
  - It reports **untracked files** (on disk but not in the manifest) and **missing files** (in the manifest but gone from disk), and updates `paired` ↔ `nef_only`/`jpg_only` accordingly.
  - It reports name collisions, duplicate copies, keeper conflicts, and duplicate folders or partial overlaps.
  - `nef_only` and `jpg_only` captures are **not** reported as problems.
  - It warns about any library or quarantine path over 240 characters.
  - Run it by hand. Scheduling it nightly with Windows Task Scheduler is optional.
- Empty the quarantine only after 30–90 days, and only once backups are confirmed.

---

## Metadata encoding reference

| Decision | `XMP-digiKam:PickLabel` | `XMP-xmp:Rating` | `XMP-xmp:Label` | Keywords |
|---|---|---|---|---|
| **Reject** | `1` | `-1` | — | `cull\|reject\|<reason>` |
| **Pending / review** | `2` | (unchanged) | `Yellow` | `cull\|pending\|<reason>` |
| **Pick** | `3` | `1`–`5` stars | — | — |

**Reason keywords:** `blur`, `shake`, `missed_focus`, `overexposed`, `underexposed`, `duplicate_copy` (exact copy of a kept file), `near_duplicate` (similar shot), `not_best_in_burst`

**Group keywords:** `burst|<id>`, `scene|<id>`, `dup|<id>` (near-duplicate set), `subject|<id>` (subject cluster), and `dupcopy|<keeper capture_id>` on duplicate copies. See [Group IDs](#group-ids) for how IDs are generated. The pipeline owns these namespaces and rewrites them on every run.

**Where keywords are written.** Each field uses its own separator convention. Confirm the formats below with a digiKam round-trip during Milestone 1.

| Field | Format | Example |
|---|---|---|
| `XMP-lr:HierarchicalSubject` | `\|`-separated path | `cull\|reject\|blur` |
| `XMP-digiKam:TagsList` | `/`-separated path | `cull/reject/blur` |
| `XMP-dc:Subject` | flat leaf keyword | `blur` |
| `EXIF:Rating` (JPG only) | 0–5 star count | lets Windows Explorer show stars. EXIF can't store `-1`, so write `0` for reject/pending. |

Example exiftool write. The pipeline will do this via `pyexiftool`:

```powershell
# JPG: embedded XMP
exiftool -overwrite_original -XMP-digiKam:PickLabel=2 -XMP-xmp:Label=Yellow `
  "-XMP-lr:HierarchicalSubject+=cull|pending|blur" "-XMP-digiKam:TagsList+=cull/pending/blur" `
  "-XMP-dc:Subject+=blur" DSC_1234.JPG

# NEF: write to the sidecar, never the raw. Create it first if missing:
exiftool -tagsfromfile DSC_1234.NEF -o DSC_1234.NEF.xmp
exiftool -overwrite_original -XMP-digiKam:PickLabel=2 ... DSC_1234.NEF.xmp
```

---

## Default thresholds

Every value below is a **starting point to be calibrated** in Milestone 2. They all live in `config.toml`.

| Check | Rule | Default |
|---|---|---|
| Overexposed | `clip_hi > A and p50 > B` | A = 0.05, B = 170 |
| Underexposed | `(mean < C and p99 < D) or clip_lo > E` | C = 50, D = 180, E = 0.30 |
| Exposure auto-reject (captures with a NEF) | `paired` and `nef_only` captures have raw headroom, so they're flagged for review only, except in extreme cases. `jpg_only` and `single` have no raw to recover from. | `clip_hi > 0.25` |
| Blur | Burst-relative z-score of Laplacian variance / Tenengrad. Bursts with fewer than 3 frames fall back to scene-level z-scores, then to global percentiles. | z < −1.5 (proposed) |
| Missed focus | `sharpness(AF box) / sharpness(frame)` | < ~1.0 |
| Exact duplicate copy | same file type + identical `ImageDataHash` (SHA-256 of image data) + matching capture metadata | exact match; no threshold |
| Near-duplicate | pHash Hamming distance | ≤ 6–8 |
| Burst gap | time between consecutive frames | > 2 s |
| Scene gap | time between consecutive frames | > 30–60 min |
| Subject clustering | HDBSCAN | `min_cluster_size=3`, cosine |
| Best-in-burst | frames kept per burst | top 1–2 |
| IQA batch | `topiq_nr`, fp16 | 32–128 |

---

## SQLite manifest

The manifest holds the pair bindings and the score cache. Decisions are mirrored here from XMP.

```sql
CREATE TABLE photos (
  capture_id  TEXT PRIMARY KEY,   -- SerialNumber_ShutterCount_DTO+SubSec
  nef_path    TEXT,
  jpg_path    TEXT,
  heic_path   TEXT,
  dto         TEXT,               -- DateTimeOriginal + SubSec, clock-offset corrected
  model       TEXT,
  serial      TEXT,
  af_x REAL, af_y REAL, af_w REAL, af_h REAL,   -- AF box in JPG pixel space
  folder      TEXT,               -- shared parent of the capture's files, format subfolders (Raw/JPG) ignored; scene hint only
  status      TEXT,               -- paired | nef_only | jpg_only | single
  quarantined INTEGER DEFAULT 0   -- 1 once Stage 7 has moved the capture's files
);
-- photos.*_path columns point at the KEEPER copy of each file.

-- Every file on disk, including duplicate copies.
CREATE TABLE files (
  path         TEXT PRIMARY KEY,  -- relative to the scanned root, '/'-separated
  capture_id   TEXT,              -- capture this file belongs to
  kind         TEXT,              -- nef | jpg | heic | xmp
  size         INTEGER,
  mtime_ns     INTEGER,           -- hashes are cleared when size or mtime change
  sha256       TEXT,              -- full-file hash (backup / move verification)
  image_hash   TEXT,              -- ImageDataHash: SHA-256 of image data, metadata excluded
  role         TEXT,              -- keeper | duplicate_copy | conflict | variant (NULL for sidecars)
  duplicate_of TEXT,              -- keeper path when role = duplicate_copy
  present      INTEGER,           -- 0 = in the manifest but missing on the last scan
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
  metric        TEXT,             -- clip_hi, clip_lo, lap_var, tenengrad, focus_ratio, topiq, ...
  value         REAL,
  model_version TEXT,
  PRIMARY KEY (capture_id, metric, model_version)
);

CREATE TABLE groups (
  capture_id TEXT PRIMARY KEY,
  burst_id   TEXT,              -- e.g. 20201205-204145.96-D780
  scene_id   TEXT,              -- e.g. 20251029-1139-puerto-rico-26
  dup_id     TEXT,              -- capture_id of the near-duplicate set's earliest member
  subject_id TEXT               -- <scene_id>/<n>
);

CREATE TABLE decisions (
  capture_id TEXT,
  source     TEXT,                -- machine | human
  decision   TEXT,                -- reject | pending | pick
  reason     TEXT,
  ts         TEXT,                -- time of the latest change
  PRIMARY KEY (capture_id, source)
);
-- Effective decision = the human row if present, else the machine row.
```

---

## Proposed project layout and CLI

*Proposed layout. Adjust it as the code takes shape.*

```
PhotoOrganizer/
├── README.md
├── PLAN.md
├── config.toml               # thresholds, library root, per-camera clock offsets (lives in the project)
├── requirements.txt          # core (M1–M3, CPU)
├── requirements-gpu.txt      # M4 GPU extras
├── cli.py                    # entry point (subcommands below)
├── photo_organizer/
│   ├── config.py             # load/validate config.toml
│   ├── manifest.py           # SQLite schema, migrations, queries
│   ├── exif_io.py            # pyexiftool stay_open wrapper; XMP read/write helpers
│   ├── dedupe.py             # Stage 2: ImageDataHash, duplicate copies, keeper choice, duplicate folders
│   ├── pair.py               # Stage 2: pairing of keeper copies
│   ├── analyze/
│   │   ├── exposure.py
│   │   ├── blur.py           # Laplacian, Tenengrad, shake (FFT / anisotropy)
│   │   ├── focus.py          # AF-box mapping + focus ratio, face fallback
│   │   └── iqa.py            # pyiqa topiq_nr (GPU)
│   ├── group/
│   │   ├── bursts.py         # burst / scene segmentation
│   │   ├── dupes.py          # pHash near-duplicates
│   │   ├── clusters.py       # CLIP/DINOv2 + HDBSCAN (GPU)
│   │   └── rank.py           # best-in-burst
│   ├── flag.py               # Stage 5: decisions → XMP
│   ├── sync_xmp.py           # Stage 6: XMP → decisions (human overrides)
│   └── cleanup.py            # Stage 7: quarantine, TSV log, integrity checks
├── tools/
│   └── draw_af_boxes.py      # debug overlay for AF-box validation
├── tests/
└── sample_photos/            # test library: copies of real shoots, one folder per event (git-ignored)
```

Planned commands:

```powershell
python cli.py snapshot D:\Photos\2024                 # Stage 0: fingerprint every file
python cli.py verify   D:\Photos\2024 [--against E:\Backup] # Stage 0: check library or backup
python cli.py scan     D:\Photos\2024                 # Stage 1: inventory files (new/changed/missing)
python cli.py pair     D:\Photos\2024                 # Stage 2: hash, dedupe, pair
python cli.py analyze  D:\Photos\2024 [--gpu]         # exposure, blur, focus (+ IQA)
python cli.py group    D:\Photos\2024 [--gpu]         # bursts, scenes, dupes, clusters
python cli.py flag     D:\Photos\2024 [--advisory]    # write XMP (advisory = pending only)
python cli.py sync     D:\Photos\2024                 # read human decisions back from XMP
python cli.py cleanup  D:\Photos\2024 [--apply]       # dry-run unless --apply
python cli.py check    D:\Photos\2024                 # untracked/missing files, collisions, integrity, path length
```

---

## Roadmap / build checklist

### Milestone 1: pairing, manifest and classical pass (advisory only)

- [ ] `config.toml` + loader (library root, thresholds, clock offsets). The config lives in the project. Each scanned root gets its own manifest at `<root>/.photoorganizer/manifest.sqlite`, so `sample_photos/` and the real library never share state.
- [x] `snapshot` command (Stage 0): write a manifest of every file under the root, with full sha256 and `ImageDataHash`, to `<root>/.photoorganizer/snapshots/<timestamp>.tsv`
- [x] `verify` command (Stage 0): check the library or a backup copy against a snapshot
- [x] Stage 1 scan: recursive inventory into the `files` table (size, mtime, kind), the folder hint (format subfolders ignored), new/changed/missing detection between scans, and warnings for paths over 240 characters
- [x] `exif_io.py`: pyexiftool `stay_open` wrapper; batch read
- [ ] `exif_io.py`: XMP write for JPG, HEIC and the NEF sidecar (Stage 5)
- [ ] Verify the keyword field formats with a digiKam round-trip (write → reread in digiKam → write back → diff)
- [x] `manifest.py`: create the schema; upsert photos
- [x] `pair.py`: recursive scan; stem index across folders (same, sibling and subfolder); confirm on DTO/SubSec/ShutterCount/Serial; nearest-match tie-break; `paired` / `nef_only` / `jpg_only` / `single` statuses; collision report; stable `capture_id` across re-scans
- [x] `dedupe.py`: ImageDataHash (SHA-256, metadata excluded) cached by path/size/mtime; exact-copy groups; keeper rules (human decisions → paired → fuller folder → preferred roots → oldest/shortest); conflict report; duplicate-folder and overlap report; mark copies `duplicate_copy` (pending in advisory mode)
- [ ] Pairing test cases in `sample_photos/`. Expected results for the current set:
  - `Birb`, `Cheesecake`, `Spring Musings`: 14 captures, all `paired` in the same folder
  - `Cheesecake/Also Cheesecake`: `DSC_0618`–`0620` (NEF + JPG) have the same image data as the copies in `Cheesecake/`. All 6 files are `duplicate_copy`, the keepers stay in `Cheesecake/` (the fuller folder), and the folder is reported as a 100% duplicate folder.
  - `Puerto Rico 26`: `DSC_9474`–`9476` and `9480` are `paired` (NEF in the `Raw/` subfolder); `DSC_9477`–`9479` are `jpg_only`. All seven get folder `Puerto Rico 26`.
  - Still to add (on copies, inside `sample_photos/`): a sibling-folder pair, a `nef_only` capture, the same stem with different capture data, a JPG copy with edited metadata, a partly overlapping folder, and two copies with conflicting human labels
- [ ] Preview loader: JPG first, then the NEF embedded preview (`exiftool -b -JpgFromRaw` / `PreviewImage`), with `rawpy` as a last resort; HEIC via `pillow-heif`; apply orientation
- [ ] `exposure.py`: Y histogram stats + flags
- [ ] `blur.py`: Laplacian variance + Tenengrad at 2048 px (global stats for now; burst-relative z-scores in M3)
- [ ] Score caching by `(capture_id, metric, model_version)`
- [ ] `flag.py --advisory`: write `PickLabel=2`, `Label=Yellow` and reason keywords to every existing file of each capture
- [ ] Run on a **copy** of one sample folder and check the results in digiKam

### Milestone 2: calibration

- [ ] Hand-label about 200 photos in digiKam (see [Calibration workflow](#calibration-workflow))
- [ ] `sync_xmp.py`: import the labels into `decisions` with `source=human`
- [ ] Calibration report: precision and recall per flag at different thresholds
- [ ] Update the defaults in `config.toml`

### Milestone 3: bursts and focus

- [ ] `bursts.py`: burst (> 2 s) and scene (> 30–60 min) segmentation using clock-corrected times
- [ ] Switch blur to burst-relative z-scores
- [ ] Shake detection (FFT / gradient anisotropy)
- [ ] `focus.py`: map the AF area to JPG pixels (scaling + orientation)
- [ ] `tools/draw_af_boxes.py`: overlay AF boxes on 20 images and verify them by eye
- [ ] Focus ratio + missed-focus flag; mediapipe face fallback
- [ ] `dupes.py`: pHash near-duplicate sets
- [ ] Group IDs (see [Group IDs](#group-ids)): deterministic ID builders, reconciliation with the previous run (Jaccard ≥ 0.5), and rewriting of the owned keyword namespaces
- [ ] Write the `burst|id`, `scene|id` and `dup|id` keywords

### Milestone 4: GPU stage

- [ ] CUDA sanity check built into `cli.py analyze --gpu`
- [ ] `iqa.py`: `topiq_nr` in fp16 with batching
- [ ] `clusters.py`: open_clip or DINOv2 embeddings + HDBSCAN within each scene; `subject|<scene_id>/<n>` IDs
- [ ] Eyes-open score (mediapipe face mesh)
- [ ] `rank.py`: weighted z-score best-in-burst; keep the top 1–2 and mark the rest Pending

### Milestone 5: cleanup and auto-reject

- [ ] `cleanup.py`: dry-run plan output; `--apply` moves all members + sidecars to the mirrored `_quarantine/`
- [ ] `moves.tsv` log + post-move sha256 verification
- [ ] Integrity check: no `capture_id` split between the library and quarantine
- [ ] `check` command: untracked/missing files, pair status refresh, collisions/duplicates, and warnings for paths over 240 characters
- [ ] *(Optional)* schedule `check` nightly with Windows Task Scheduler
- [ ] Restore command that reverses moves from `moves.tsv`
- [ ] Enable auto-reject **only** for high-confidence cases (e.g. `clip_hi > 0.25`, extreme blur outliers)

---

## Calibration workflow

1. Copy a representative sample of about 200 photos into `sample_photos/`, in any folder layout. Include bursts, low light, backlit scenes and phone shots.
2. Run `pair`, `analyze` and `flag --advisory` on the sample.
3. In digiKam, **ignore the machine labels** and label each photo yourself:
   - Pick: Accepted
   - Reject: Rejected, with a tag such as `cull/reject/blur`
   - Unsure: Pending
4. Run *Write Metadata to Files*, then run `python cli.py sync sample_photos/`.
5. Compare human decisions with machine scores for each metric. Choose thresholds that give **high precision for rejects**, because a missed reject costs less than a wrongly rejected keeper.
6. Update `config.toml`. Bump the `model_version` or config hash so cached flags are recomputed.
7. Repeat on a second, different sample before trusting the thresholds library-wide.

---

## Gotchas

- **Sidecar conflicts:** don't let darktable and digiKam write the same `.xmp` sidecar. Use darktable only for editing chosen picks, and configure it to write its own sidecars, or none.
- **Counter rollover:** Nikon's file counter wraps from `DSC_9999` back to `DSC_0001`. Never pair on filename alone; always confirm with DTO/SubSec/ShutterCount.
- **Clock drift:** phone and camera clocks drift apart. Store a per-camera time offset in `config.toml` and apply it before burst/scene grouping.
- **Long paths:** Windows' 260-character path limit is unlikely to matter for typical layouts (`_quarantine/` adds only about 12 characters). Keep the library root shallow and event names short, and turn on long-path support as optional insurance (see [Setup](#4-windows-long-paths-optional)). The `check` command warns about any path over 240 characters.
- **Test in `sample_photos/` only:** develop and run every new stage against `sample_photos/`, which holds copies of real shoots (e.g. `sample_photos/Birb/`). Point the pipeline at the real library only after a stage behaves correctly there.
- **Torch builds:** the cu118, cu121 and cu124 wheels install fine, then fail at runtime on `sm_120`. Use cu128 or newer, no higher than the CUDA version your driver reports in `nvidia-smi`. Wheels bundle their own CUDA runtime, so the CUDA Toolkit doesn't need to be installed.

---

## Backup and recovery

- **Before anything else:** set up 3-2-1 backups and generate a sha256 manifest of the whole library.
- **Quarantine is reversible:** every move is in `_quarantine/moves.tsv`. Restoring means moving each `dst_path` back to its `src_path` and re-verifying the sha256 (planned `restore` command).
- **Retention:** keep the quarantine for 30–90 days, and empty it only after confirming that a backup taken *before* the cleanup is intact.
- **Manifest backup:** the SQLite manifest can be rebuilt from the files and their XMP, but back it up anyway. Scores are expensive to recompute.

---

## Acknowledgements

This is a personal hobby project built entirely on free tools: [exiftool](https://exiftool.org/), [digiKam](https://www.digikam.org/), [Krokiet/Czkawka](https://github.com/qarmin/czkawka), [OpenCV](https://opencv.org/), [pyiqa](https://github.com/chaofengc/IQA-PyTorch), [OpenCLIP](https://github.com/mlfoundations/open_clip), [DINOv2](https://github.com/facebookresearch/dinov2), [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan), [imagehash](https://github.com/JohannesBuchner/imagehash), [MediaPipe](https://developers.google.com/mediapipe), [rawpy/LibRaw](https://github.com/letmaik/rawpy), and [PyTorch](https://pytorch.org/).

[`PLAN.md`](PLAN.md) is a compact implementation summary of this README. If the two disagree, this README wins.
