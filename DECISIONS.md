# DECISIONS.md — AIT304 PCB Defect Detection
> Append-only log of every architectural / methodological choice. Each entry defends a "why".
> Useful for: (1) conference Q&A defense, (2) report writing, (3) onboarding a new Claude chat to the reasoning behind choices.

---

## How to use this file

- **Append only.** Never edit past decisions. If reversing, add a new entry pointing to the old one.
- **Date every entry.** Format: `YYYY-MM-DD`.
- **One decision per entry.** Don't bundle.
- **Include alternatives considered.** Future-you will want to know why X beat Y.

Template:
```
## D-NNN | YYYY-MM-DD | <topic>
**Decision:** <what we chose>
**Why:** <one or two sentences>
**Alternatives considered:** <what was rejected and why>
**Affects:** <which files / which phase>
**Reversible?** <yes / no / costly>
```

---

## D-001 | 2026-06-14 | Domain selection
**Decision:** Manufacturing (Option 6) — PCB defect detection
**Why:** Aligns with target company ViTrox Corporation (Malaysia's leading machine-vision / AOI company for PCBs). Creates a portfolio piece directly relevant for post-graduation freelance / employment outreach.
**Alternatives considered:**
- Healthcare (MRI / X-ray) — rejected: no Malaysian industry hook; dataset access more restricted
- Robotics — rejected: King's robotics project (safe-robot-system) already covers this domain; portfolio diversification matters
- Transportation — rejected: oversaturated, weaker connection to local industry
**Affects:** Everything downstream
**Reversible?** Yes, but would restart the project

## D-002 | 2026-06-14 | Dataset choice
**Decision:** DeepPCB (Tang et al. 2019, arXiv:1902.06197)
**Why:** Free, 1,500 image pairs, 6 defect categories matching real PCB inspection, well-cited benchmark with strong SOTA comparison numbers (YOLOv8-DEE: 98.7% mAP). Already in YOLO-friendly 640×640 format.
**Alternatives considered:**
- PKU-Market-PCB — rejected: only 673 images, less data
- HRIPCB — rejected: smaller, used together with DeepPCB in YOLOv8-DEE paper but DeepPCB is the more competitive benchmark
**Affects:** Phase 3 data prep, all training
**Reversible?** Yes, but costs 1 day to swap

## D-003 | 2026-06-14 | Baseline + enhancement pair
**Decision:** Baseline = YOLOv8n. Enhanced = YOLO11n + Coordinate Attention.
**Why:** YOLOv8n is the industry-standard baseline that examiners recognize. YOLO11n is newer (Sep 2024) and shows King is current with the field. CA is a well-cited mechanism (Hou et al. CVPR 2021, 3800+ citations) with proven track record in PCB applications (PCB-YOLO PLOS One 2025 uses CA on PKU dataset).
**Alternatives considered:**
- CBAM attention — rejected: less direction-aware, weaker on small spatial targets
- EMA attention — rejected: already used by SOTA YOLOv11-PCB paper; less novel framing for our work
- Just compare YOLOv8n vs YOLO11n with no attention — rejected: too thin for "Excellent" model architecture rubric
**Affects:** Phase 4 training, model_architecture rubric (10%)
**Reversible?** Costly — different YAML, different module to register

## D-004 | 2026-06-14 | Loss function variants
**Decision:** CIoU (default) vs Wise-IoU (Tong et al. 2023, arXiv:2301.10051)
**Why:** Rubric requires "different optimizers with different loss functions" for Excellent. WIoU's dynamic non-monotonic focusing mechanism is theoretically well-suited to PCB defects where anchor box quality varies dramatically (tiny pin-holes vs large shorts).
**Alternatives considered:**
- CIoU vs DIoU — rejected: too small a difference, weak ablation story
- CIoU vs EIoU — rejected: EIoU already used by YOLOv8-DEE and YOLOv11-PCB; using it would make our novelty thinner
**Affects:** `wiou_patch.py`, EXP-2 and EXP-4
**Reversible?** Yes via kernel restart

## D-005 | 2026-06-14 | Optimizer variants
**Decision:** Adam vs SGD with momentum
**Why:** Adam is YOLO default; SGD is the classical robust alternative with different convergence dynamics. Standard ablation pair recognized in the literature.
**Alternatives considered:**
- AdamW — rejected: too similar to Adam, weak ablation
- RMSprop — rejected: rarely used in modern detection, less defensible
**Affects:** Training cells in Phase 4
**Reversible?** Yes (just change one arg)

## D-006 | 2026-06-14 | CA insertion points
**Decision:** Insert CoordAtt at all three detect-head feature maps (P3, P4, P5)
**Why:** Each scale captures defects of different sizes. Direction-aware attention applied per-scale should help small-defect localization (P3) without breaking large-defect detection (P5). Matches the insertion strategy in PCB-YOLO PLOS One 2025.
**Alternatives considered:**
- CA only at P3 (smallest features) — rejected: too narrow, weak signal
- CA throughout the entire backbone — rejected: too many params, hurts FPS
**Affects:** `yolo11n-CA.yaml`
**Reversible?** Yes by editing YAML

## D-007 | 2026-06-14 | Compute platform
**Decision:** Kaggle Notebooks (free 30 GPU hrs/week, T4 ×2)
**Why:** Free, well-documented, persistent dataset storage. Total need ~10 GPU hours; 9× headroom available.
**Alternatives considered:**
- Google Colab — rejected: more frequent disconnects, smaller free quota
- RunPod (King's robotics project setup) — rejected: costs money, no benefit for this workload
- Local PC — rejected: no NVIDIA GPU available locally
**Affects:** Phase 3 onward
**Reversible?** Easy fallback to Colab if Kaggle outage

## D-008 | 2026-06-14 | Version control discipline
**Decision:** GitHub private repo as source of truth. PROJECT_STATE.md + DECISIONS.md updated daily, committed nightly.
**Why:** 19-day sprint across 5 tool surfaces (this chat, Claude Code, Codex, Cowork, Kaggle) — without a single source of truth, context drift kills the project by Day 5. Pattern proven in King's robotics project.
**Alternatives considered:**
- Just use OneDrive — rejected: no version history, no commit messages
- No tracking, just memory — rejected: chat transfer would lose everything
**Affects:** Daily workflow
**Reversible?** Not really — once started, must continue

## D-009 | 2026-06-14 | Tool role boundaries
**Decision:** Claude.ai chat = decisions only. Claude Code = all coding. Codex = code review only. Cowork = logistics only. Kaggle = compute only.
**Why:** Strict role separation prevents two-cook conflicts (e.g. Claude Code and Codex both editing the same file). Also keeps each chat focused on what it's best at.
**Alternatives considered:**
- Let Claude.ai also write code — rejected: code generated in this chat can't be tested here; it just adds a copy-paste step
- Use only one AI tool — rejected: Codex review on critical files (WIoU patch, CA YAML) catches what single-tool review misses
**Affects:** All workflow
**Reversible?** Yes if needed mid-project

---

## D-010 | 2026-06-14 | DeepPCB actual dataset structure differs from blueprint v2 assumption
**Decision:** Use disk-verified paths from trainval.txt directly.
Image path requires `_test.jpg` suffix; annotation path comes from
column 2 of trainval.txt without transformation.

**Why:** PowerShell verification on Day 1 showed:
  - trainval.txt col1 references `group{N}/{N}/{id}.jpg` — file does NOT exist on disk
  - Disk only has `_test.jpg` and `_temp.jpg` variants
  - trainval.txt col2 references `group{N}/{N}_not/{id}.txt` — file DOES exist as-is

**Required convert_dataset.py fixes (apply on Day 3):**

Fix 1 — image path needs _test suffix:
    # BROKEN (blueprint §5.3):
    img_src = SRC / rel_path
    # CORRECTED:
    img_src = SRC / col1.replace('.jpg', '_test.jpg')

Fix 2 — annotation path is column 2, not derived from image name:
    # BROKEN (blueprint §5.3):
    ann_src = SRC / rel_path.replace('_test.jpg', '.txt')
    # CORRECTED:
    ann_src = SRC / col2

Fix 3 — read_split() returns pairs of (image_path, annotation_path):
    # BROKEN:
    pairs.append(test_img)
    # CORRECTED:
    pairs.append((img_path, ann_path))

Coordinate normalisation, class `cls - 1` remap, and YOLO label format
are unaffected.

**Alternatives considered:** None — the data is what's on disk.

**Affects:** Phase 3 (`convert_dataset.py`, indirectly `sanity_check.py`)

**Reversible?** No — this is the only structurally correct way to read the dataset.

---

## D-011 — Kaggle → Google Colab pivot for GPU training
**Date:** 2026-06-17
**Status:** Active
**Context:** Kaggle account created during Day 1 setup, but phone verification failed on two attempts (own number + family member's number). Without verification, Kaggle Notebooks GPU access is locked.
**Decision:** Pivot training infrastructure from Kaggle Notebooks to Google Colab (free tier, T4 GPU).
**Rationale:** Day 3 begins the training pipeline scaffold; we cannot wait further on Kaggle. Colab has no phone verification, supports the same Ultralytics + PyTorch stack, and can run from the same GitHub repo via `!git clone`. Trade-off: Colab free tier sessions are time-limited (~12 hrs) and less predictable than Kaggle's 30 hrs/week quota — we may need to checkpoint training more aggressively.
**Implications for Day 3+:** Training scripts must be Colab-compatible (drive mount, session reconnection handling). Dataset conversion scripts unchanged. Will revisit if Colab quota proves insufficient for all 4 experiments.

---

## D-012 — NotebookLM adopted for comprehension + cross-paper synthesis
**Date:** 2026-06-17
**Status:** Active
**Context:** Day 2 literature review requires reading 9 papers and synthesizing gaps for the contribution claim. Manual reading of 9 PDFs estimated at 4+ hours.
**Decision:** Adopt NotebookLM as the primary comprehension and cross-paper synthesis tool. Zotero retained as the citation archive + BibTeX export source (NOT replaced).
**Rationale:** NotebookLM is grounded in uploaded source PDFs with inline citations, reducing hallucination risk for factual extraction. Validated today by independently surfacing YOLOv11+CA and YOLOv11+WIoU as unoccupied gaps in the design matrix — matched the blueprint v2.0 contribution claim from independent ground truth. Manual verification still required for high-stakes claims (e.g. PCB-YOLO FFCA-vs-CA distinction was manually verified against pages 3, 10–11 of the source).
**Implications:** Zotero remains the source of truth for citations into Overleaf. NotebookLM outputs are working notes, not deliverables. High-stakes claims (anything that goes into §2 or §3 of the CVPR report) require manual source verification before use.

---

## D-007 amendment | 2026-06-24 | Compute platform: Kaggle → Google Colab
**Decision:** The original D-007 selection of Kaggle as the GPU training platform is superseded. Google Colab is the active compute platform for all training, smoke tests, and ablation experiments.

**Why:** Kaggle account creation succeeded under username `tyztehyanze`, but phone verification failed and blocks GPU access entirely (logged as D-011). Colab provides equivalent T4/L4 GPU access without phone verification, identical Ultralytics 8.3.40 install path, and Drive-mount persistence that survives session disconnects. The training code itself is platform-agnostic.

**Source:** D-011 (Kaggle phone verification blocker, Day 1).

**Affects:** All Phase 3+ training and evaluation work. `FILE_DEPENDENCIES.md` §3.1 updated to reflect Colab as the platform.

**Reversible?** Yes — if Kaggle phone verification ever clears, the training code runs on either platform unchanged. No need to revert unless Colab quota becomes a hard blocker.

---

## D-010 amendment | 2026-06-24 | Implementation via flat-index lookup
**Decision:** The literal D-010 fixes (append `_test.jpg` to col1, read annotations from col2 directly, return `(img_path, ann_path)` pairs) were superseded by a more general flat-index implementation in `src/data/convert_dataset.py`.

**How it works:** The file system under the resolved DeepPCB root is walked once at startup via `pcbdata_root.rglob("*")`. Two independent indexes are built:
- `image_index`: maps bare sample id (e.g. `00041000`) → absolute path of `*_test.jpg`
- `label_index`: maps bare sample id → absolute path of `*.txt` annotation

`parse_split_file` reads column 1 of each split-file line, extracts the bare id via `_bare_id()` (stripping `_test`/`_temp` suffixes from the stem), then resolves both image and annotation via dict lookup. The nonexistent `.jpg` paths in col1 are never used as paths — only as a source for the bare id. Column 2 is not parsed in this implementation; the annotation is resolved by the same bare id.

**Why this is better than the literal D-010 plan:**
- Robust to dataset layout variations (auto-detection of PCBData root via `_resolve_pcbdata_root` supports both `<src>/PCBData/` and `<src>/PCBData/PCBData/`).
- Dict lookup failure surfaces missing files explicitly rather than producing silently broken paths.
- Functionally equivalent to the original D-010 for DeepPCB's actual on-disk layout: same image, same annotation, same pairing.

**Affects:** `src/data/convert_dataset.py` (Day 3 P1).

**Reversible?** No — this is the implemented behavior. The D-010 logical guarantee (correct image + annotation pairing on disk) is preserved.

---

## D-013 | 2026-06-24 | Ultralytics version pin = 8.3.40
**Decision:** Pin Ultralytics to `ultralytics==8.3.40` for the entire project. No floating versions.

**Why:** The 8.3.x series is the first to officially support YOLOv11. A floating version risks subtle behavior drift between baseline runs and CA+WIoU runs (e.g., a tweaked NMS implementation in a patch release would invalidate ablation comparisons by changing the underlying code under both arms of the experiment). Pinning guarantees the baseline and the contribution arm execute on byte-identical Ultralytics code.

**Alternatives considered:** Floating `pip install ultralytics` (rejected: reproducibility risk); pinning to latest 8.3.x at experiment start (rejected: still floating across experiments).

**Affects:** Colab notebook install cell (Day 3 P4), eventual `requirements.txt`.

**Reversible?** Yes if needed mid-project, but any version change requires re-running all completed experiments under the new version. Treat as effectively locked.

---

## D-014 | 2026-06-24 | Single tested-image input (no template)
**Decision:** Use only `*_test.jpg` as YOLO input. Template images (`*_temp.jpg`) are ignored entirely.

**Why:**
1. Direct comparability with the two closest published baselines: PCB-YOLO and YOLOv11-PCB both treat DeepPCB as standard single-image object detection on the tested image. Using paired input would change the task to reference-based change detection, making mAP numbers non-comparable.
2. Operational realism: a production AOI line does not always have a pixel-aligned template image at inference time. A detector that works on the tested image alone is more deployable, which aligns with the ViTrox portfolio target.

**Implementation:** `build_file_index` only registers files whose stem ends in `_test`. Files ending in `_temp` are silently skipped (no `_temp` references appear anywhere else in the script).

**Reversible?** No without invalidating baseline comparisons.

---

## D-015 | 2026-06-24 | Honor official trainval/test split; 80/20 val carve at seed=42
**Decision:** Preserve DeepPCB's official `trainval.txt` / `test.txt` split exactly. Within trainval, carve an 80/20 train/validation subset using `random.Random(42).shuffle(...)`. Test set is never touched during model development.

**Why:** The DeepPCB authors fixed which 500 images form the test set. Every published mAP on this benchmark is measured against that exact test set. Reshuffling would invalidate all literature comparisons regardless of model quality. The 80/20 val carve out of trainval enables early stopping and hyperparameter monitoring without touching the held-out test set, protecting against indirect test-set contamination via hyperparameter tuning.

**Implementation:** `val_size = len(shuffled) // 5` gives 200 val / 800 train for the standard 1000-entry trainval. Seed=42 is supplied to a local `random.Random(42)` instance, not the global random module, so reproducibility is guaranteed regardless of any other RNG usage.

**Reversible?** No — reshuffling test set would invalidate all literature comparisons.

**Status:** superseded by D-024 (Day 5, 1 July 2026).

---

## D-016 | 2026-06-24 | Accept defensive redundancy in validation invariants 1a and 1b
**Decision:** Validation invariants 1a (parser-drop check: `len(parsed_pairs) == raw_line_count`) and 1b (split-sum check: `train + val == trainval_raw_count`) are mathematically redundant for the trainval/test sums but are intentionally kept separate.

**Why:** Distinct failure modes produce distinct error messages, making 2 a.m. debugging unambiguous. A combined `train + val + test == trainval_raw + test_raw` check would compress two diagnostic signals into one. Belt-and-braces validation on data-pipeline code is a feature on the critical path to all downstream experiments.

**Source:** Codex re-review of Day 3 P1, verdict APPROVE WITH MINOR FIXES; the "minor fix" was the redundancy cleanup, which is intentionally not applied per this decision.

**Reversible?** Yes, trivially — collapse to a single check if ever desired.

---

## D-017 | 2026-06-24 | Accept duplicate `build_class_index()` scan in sanity_check.py
**Decision:** `src/data/sanity_check.py` calls `build_class_index()` twice during a single run — once for class-stratified sampling and once for the post-sampling coverage gate. The duplicate scan is intentional and not refactored away.

**Why:**
1. Cost is negligible: the scan walks ~1500 small text files, well under a second. `sanity_check.py` runs a handful of times over the project lifetime; this is not a hot path.
2. The two callers have semantically distinct purposes (selection vs validation). Sharing state would require passing the index through several function boundaries or stashing it in module-level state, trading clarity for non-measurable speedup.
3. The redundancy provides mild belt-and-braces: if `build_class_index()` ever produced inconsistent results between the two scans (filesystem mutation mid-run, race condition, encoding edge case), the coverage gate would catch the inconsistency rather than silently agreeing with the sampler.

**Source:** Codex re-review of Day 3 P2, verdict APPROVE WITH MINOR FIXES; the "minor fix" was the re-scan elimination, intentionally not applied per this decision. Same pattern as D-016.

**Reversible?** Yes, trivially — pass the index from `select_images()` to the coverage check function if ever desired.

---

## D-018 | 2026-06-27 | Augmentation policy for project lifetime
**Decision:** All spatial and color augmentations OFF except fliplr=0.5. Concretely:
hsv_h=0, hsv_s=0, hsv_v=0, flipud=0, fliplr=0.5, mosaic=0, degrees=0,
translate=0, scale=0, shear=0, perspective=0.
**Why:** PCB defects are orientation- and color-specific. DeepPCB images are predominantly
green; hue/saturation shifts have no physical meaning. Defects like spurs and mousebites
have directional features; flipud changes their meaning. COCO defaults are wrong for this
domain. Locked identically across Day 4 smoke + Day 5+ four-experiment ablation so the
only variables in ablation are optimizer and loss function.
**Alternatives considered:** COCO defaults rejected (wrong domain). flipud=0.5 rejected
(changes directional defect meaning for spurs/mousebites).
**Affects:** Day 4 smoke onward, all training runs.
**Reversible?** Medium — reversing requires re-running all completed experiments under
new augmentation settings.

---

## D-019 | 2026-06-27 | Colab-local dataset YAML + preflight data gate
**Decision:** The notebook generates /content/datasets/deeppcb/deeppcb_colab.yaml after
rsync from Drive to local SSD, then runs a preflight data gate before any Ultralytics
call. Gate asserts: 800 train / 200 val / 500 test image counts; every .jpg has a
matching .txt stem; all class IDs in label files are in {0..5}; no label file is empty.
**Why:** The repo deeppcb.yaml contains a Windows absolute path (C:\Users\...) and is
not portable to Colab. Generating a Colab-local YAML at runtime keeps the canonical YAML
untouched while making the notebook self-sufficient. The preflight gate catches data
corruption or rsync incompleteness before burning Colab GPU quota on a training run that
will fail with a cryptic Ultralytics error.
**Alternatives considered:** Editing deeppcb.yaml in-place rejected (pollutes canonical
YAML with Colab-specific paths, breaks local tooling). Skipping the gate rejected (silent
failures on partial rsync waste GPU quota with no clear error).
**Affects:** Day 4 onward, all Colab-based training.
**Reversible?** High — drop the YAML and gate cells; no impact on prior commits.

---

### D-020 — Cross-platform zip handling
**Date:** 28 June 2026
**Context:** Windows PowerShell `Compress-Archive` emits zip entries with literal backslashes (`train\images\foo.jpg`). Python's `zipfile.extractall` on Linux/Colab treats the whole string as a filename, producing a flat output (~1700 files) instead of a directory tree.
**Decision:** `notebooks/day4_pipeline.ipynb` cell 3 normalizes `\` → `/` on each `ZipInfo.filename` before extraction. Going forward, prefer producing archives with `python -m zipfile -c deeppcb.zip deeppcb/` or 7-Zip to keep entries POSIX-clean at the source.
**Status:** active.

### D-021 — Repo visibility policy
**Date:** 28 June 2026
**Context:** Private GitHub repos cannot be cloned anonymously from Colab; Day 4b required temporarily making `pipiking123/pcb-defect-detection` public to unblock `git clone` in cell 1.
**Decision:** Repo MAY be public during active Colab development sessions for clone convenience. Repo MUST be re-privated at the close of each multi-day milestone if portfolio-sensitive content (novel architecture code, unpublished thesis material, contribution-claim differentiators) has landed since the last public window. Day 4 close-out re-privates immediately after this commit pushes.
**Status:** active.

### D-022 — Browser-incognito requirement for Colab
**Date:** 28 June 2026
**Context:** Google OAuth bug #5944 ("Google Drive for desktop" wrong OAuth client) breaks `drive.mount()` on King's default Chrome profile. Bug is profile-state-specific (suspected extension or stale auth cookie) and still active Google-side as of 28 June 2026.
**Decision:** Until the triggering profile state is identified and removed, all Colab work on King's account is done in a Chrome incognito tab. Open follow-up: bisect extensions / clear site data to identify trigger and restore default-profile usage.
**Status:** active, with open follow-up.

### D-023 — Codex review workflow
**Date:** 1 July 2026
**Context:** During Day 5 item 1 close-out, a Codex-labeled review was produced by Claude Code itself rather than by independent Codex execution in VS Code. This collapsed the intended independence between the code-writing layer (Claude Code) and the review layer (Codex), defeating the four-layer review discipline established for the project.
**Decision:** Claude Code must never produce, simulate, or delegate Codex reviews under any label or via any sub-agent. "Codex review" means Claude Code stops, presents the diff, and waits for King to run Codex independently in VS Code and paste back the verdict. Rationale: preserves the independence of the second-reviewer layer, especially critical for the upcoming CA module, WIoU patch, and any file with subtle correctness properties.
**Status:** active.

### D-024 — Plate-aware train/val split and upstream test-set leakage
**Date:** 1 July 2026
**Context:** Day 5 audit of src/data/convert_dataset.py revealed the per-image random split (D-015 original) leaked 93 of 94 val plates into train — 199 of 200 val images (99.5%) came from plates the model also saw during training. Root cause: shuffling individual (bare_id, img_path, ann_path) tuples rather than grouping by source PCB plate. DeepPCB filenames encode plate ID in the first 7 characters of the 8-character stem (last char is scan index 0-9). The baseline mAP50=0.928 from Day 4b is therefore invalidated as a generalization metric.

A second finding: DeepPCB's official upstream split has 4 plates cross-cutting trainval.txt and test.txt (plates 1300019, 2008529, 4400006, 9200011 — 22 of 500 test images = 4.4% test-set plate leakage). This is inherited from upstream and is present in all published DeepPCB benchmark numbers (PCB-YOLO, YOLOv11-PCB, etc.).

**Decision:**
1. Train/val split rewritten to group by 7-char plate ID, shuffle plate IDs with seed=42, and assign all images of a plate to the same split. Enforced by a permanent inline plate-disjoint assertion and a <5-plate ValueError guard. Committed at a02964b after Codex APPROVE.
2. Upstream test-set leakage is documented but NOT excised. Primary results reported against the standard 500-image test set for direct comparability with PCB-YOLO and YOLOv11-PCB. Supplementary results reported against a cleaned 478-image test subset (4 leaked plates removed at eval time) to measure honest generalization. Both numbers appear in the final report and viva slides.

**Status:** active. Dataset regeneration + 5-epoch smoke re-run pending (Day 5 items 2c-2e) to establish honest baseline.

---

## D-026 | 2026-07-02 | Ablation grid expanded to 6 runs (two vanilla-100 baselines)
**Decision:** The ablation grid is 6 runs, not 5. Two vanilla-100 cells are included:
(a) vanilla_adam_ciou_100 — optimizer matched to ca_adam_ciou, isolating CA as the
single variable for a clean CA-vs-no-CA comparison at equal optimizer/loss settings.
(b) vanilla_adamw_ciou_100 — optimizer matched to the Day 4b/Day 5 5-epoch smoke run,
serving as an "out of the box" headroom baseline comparable to the honest smoke artifact
(artifacts/day5_honest_smoke/). Both vanilla cells use model=yolo11n.yaml (stock, no CA)
and iou_type=ciou. Ultralytics per-optimizer defaults are used for lr0/momentum/
weight_decay in all cells (no hard-fix); each config documents its optimizer choice
inline so the defaults source is traceable.
**Why:** A single vanilla-100 baseline cannot serve both purposes at once: isolating CA's
contribution requires holding optimizer/loss fixed against the CA cells (Adam+CIoU), while
preserving comparability with the already-committed 5-epoch smoke artifact requires
matching that run's optimizer (AdamW, the smoke default). Splitting into two vanilla
cells resolves this without compromising either comparison.
**Alternatives considered:** Single vanilla-100 at AdamW only — rejected: leaves the CA
ablation confounded by an optimizer difference (CA cells run Adam and SGD, not AdamW).
Single vanilla-100 at Adam only — rejected: breaks continuity with the honest smoke
baseline, which used AdamW.
**Affects:** configs/ablation/*.yaml (Item 5), scripts/run_ablation.py, ablation report
tables in the CVPR-style writeup.
**Reversible?** Yes — dropping one vanilla cell costs a config file and a discussion
paragraph, not a re-run of any other cell.

## D-027 | 2026-07-02 | T4 wall-time optimization applied to all 6 ablation configs
**Decision:** All 6 ablation configs (configs/ablation/*.yaml) set patience=20 (early
stop if val mAP50 has not improved for 20 epochs; full curve up to the stop point is
still recorded), cache='ram' (DeepPCB's ~800 train images fit comfortably in T4 RAM,
eliminating per-epoch disk I/O), and batch=32 (up from the 16 used in the 5-epoch smoke;
YOLOv11n at 640px fits comfortably in T4 15GB VRAM, giving roughly 1.5x epoch
throughput).
**Why:** DeepPCB is an easy benchmark — the 5-epoch smoke already hit mAP50=0.927
(artifacts/day5_honest_smoke/), so all 6 runs are expected to saturate well before 100
epochs; patience terminates early without affecting the reported best-epoch metrics.
Free-tier Colab T4 sessions have a ~12-hour limit with silent disconnects; at the
5-epoch smoke's per-epoch wall time, 6 uncapped 100-epoch runs would take an estimated
~10 hours, leaving no margin for a disconnect-and-resume. The batch/cache changes
compress this to an estimated ~4 hours, fitting inside one uninterrupted session.
**Alternatives considered:** Leaving batch=16 (smoke-identical) — rejected: throughput
too low to fit 6 runs in one session. Uncapped epochs (no patience) — rejected: wastes
GPU quota re-training past convergence on an already-easy dataset.
**Affects:** All 6 configs/ablation/*.yaml files, scripts/run_ablation.py, total Colab
GPU-hours budget for Item 5.
**Reversible?** Yes — patience/cache/batch are per-run config values; any cell can be
re-run with different settings without affecting the others.
**Report note:** batch=32 was chosen for training-time efficiency under session
constraints; a larger batch may require learning-rate tuning in follow-up work (not
applied here — Ultralytics per-optimizer LR defaults are used as-is, per D-026).

---

## D-028 | 2026-07-02 | Explicit lr0 per optimizer — supersedes D-026's LR clause
**Decision:** Verified in Ultralytics 8.3.40's `build_optimizer()` (ultralytics/engine/
trainer.py) that per-optimizer learning-rate selection only occurs when
`optimizer='auto'`. With an explicit optimizer name (Adam, AdamW, SGD — as all 6
ablation configs use), every run instead inherits `lr0=0.01` from
`ultralytics/cfg/default.yaml`, an SGD-tuned value. This is ~10x too high for
Adam/AdamW and would confound the optimizer-effect comparison in the ablation.
lr0 is now set explicitly per config:
- Adam cells (vanilla_adam_ciou_100, ca_adam_ciou, ca_adam_wiou): lr0=0.001
- AdamW cells (vanilla_adamw_ciou_100): lr0=0.001
- SGD cells (ca_sgd_ciou, ca_sgd_wiou): lr0=0.01 (matches default.yaml; made explicit
  for the diff record and viva defense, not a behavior change)
**Why:** Preserves the intent of D-026 ("Ultralytics per-optimizer defaults used for
lr0... no hard-fix"), which was written under the mistaken assumption that Ultralytics
auto-scales lr0 per optimizer outside of `optimizer='auto'`. D-028 is the concrete,
verified implementation of that original intent, correcting D-026's LR clause without
reopening the rest of that decision (six-cell grid, batch/cache/patience settings are
unaffected).
**Alternatives considered:** Leaving lr0 unset and accepting the SGD-tuned 0.01 for
Adam/AdamW — rejected: makes the Adam-vs-SGD ablation cells confounded by an
unintentional LR mismatch rather than a controlled optimizer comparison. Using
`optimizer='auto'` for the Adam/AdamW cells to get Ultralytics' automatic LR — rejected:
`auto` also silently overrides the *optimizer choice itself* (picks AdamW or SGD based
on iteration count), which would break the ablation's control over which optimizer runs
in which cell.
**Affects:** All 6 configs/ablation/*.yaml files (adds one `lr0:` line each,
immediately below `optimizer:`).
**Reversible?** Yes — lr0 is a per-run config value; reverting to implicit lr0=0.01
costs one line per config.
**Report note:** Cite Ultralytics 8.3.40 `default.yaml` (lr0=0.01, SGD provenance) and
standard Adam LR guidance (Kingma & Ba, 2014; Loshchilov & Hutter, 2019 for AdamW) as
the basis for the explicit 0.001 Adam/AdamW value.

## D-029: Item 6 close-out — eval script post-fix Codex re-review and audit-artifact archival

**Date:** 2026-07-03
**Status:** Accepted
**Context:**
Item 6 (evaluation) was executed on Kaggle T4 at eval script commit `5c2178a`, which contains the `split="test"` fix for `model.val()`. The initial Codex review (D-023 workflow) was performed on the pre-fix commit `5ddf486`; the split bug was discovered post-review during first eval execution. After the fix landed at `5c2178a`, a post-fix Codex re-review was requested to confirm no regressions on the three prior MAJOR fixes (seed order, per-pair failure handling, metrics.json provenance) and to verify the fix itself.

**Decision:**
1. Accept the post-fix Codex re-review at `5c2178a` as the final code approval for Item 6.
2. Archive the exact test YAMLs (`deeppcb_kaggle.yaml`, `deeppcb_kaggle_cleaned.yaml`), the cleaned test manifest (`cleaned_test_manifest.txt`), the final Kaggle eval outputs (`runs_eval/` — 12 metrics.json + summaries + Ultralytics artifacts + demo grids), and provenance files (`README.md`, `demo_paths.txt`) under `artifacts/item6_eval/` in the repo, so the audit trail is reproducible from a fresh clone.
3. Defer the one remaining MINOR (dry-run creates output_dir before exiting) as cosmetic; does not affect reported metrics.

**Codex re-review outcome (5c2178a):**
- APPROVED WITHOUT CHANGES on the code path
- One `model.val()` call site confirmed at line 375 with `split="test"` correctly passed
- All three prior MAJOR fixes verified intact (seed order lines 704–716, per-pair failure handling lines 371–476 + 744–759, metrics.json provenance lines 403–412)
- Only two MAJORs raised were audit-artifact gaps (missing YAMLs and eval outputs in repo checkout) — both closed by this commit
- Verdict: "APPROVE WITH AUDIT-ARTIFACT FIXES"

**Consequences:**
- Item 6 code phase is closed. Final reported numbers (mAP50=0.9450 for ca_sgd_wiou standard; cleaned delta ~0.001) are defensible from a code-correctness standpoint.
- Repo is now self-contained: examiner cloning `pipiking123/pcb-defect-detection` at HEAD can inspect both the eval script and its evidence without external downloads.
- D-023 four-layer workflow (plan/write/review/approve) held across the full Item 6 lifecycle, including the post-fix re-review loop.

**Files added under `artifacts/item6_eval/`:**
- `configs/deeppcb_kaggle.yaml` — standard 500-image test YAML used at eval
- `configs/deeppcb_kaggle_cleaned.yaml` — cleaned 478-image test YAML (D-024 supplementary)
- `configs/cleaned_test_manifest.txt` — 22 excluded image IDs + SHA-256s
- `runs_eval/` — 6 runs × 2 test sets = 12 pair outputs (metrics.json, curves, confusion matrices, val_batch predictions) + `summary_standard.csv` + `summary_cleaned.csv` + `eval_manifest.txt` + `demo/` cross-model grid
- `README.md` — top-level provenance and headline results
- `demo_paths.txt` — 4 demo image paths for reproducibility

**Related decisions:** D-013 (Ultralytics pin), D-023 (four-layer workflow), D-024 (plate-aware split), D-025 (CA nano channel lock)

---

*DECISIONS.md v1 | Initialized 2026-06-14 | Append-only*
