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

---

## D-016 | 2026-06-24 | Accept defensive redundancy in validation invariants 1a and 1b
**Decision:** Validation invariants 1a (parser-drop check: `len(parsed_pairs) == raw_line_count`) and 1b (split-sum check: `train + val == trainval_raw_count`) are mathematically redundant for the trainval/test sums but are intentionally kept separate.

**Why:** Distinct failure modes produce distinct error messages, making 2 a.m. debugging unambiguous. A combined `train + val + test == trainval_raw + test_raw` check would compress two diagnostic signals into one. Belt-and-braces validation on data-pipeline code is a feature on the critical path to all downstream experiments.

**Source:** Codex re-review of Day 3 P1, verdict APPROVE WITH MINOR FIXES; the "minor fix" was the redundancy cleanup, which is intentionally not applied per this decision.

**Reversible?** Yes, trivially — collapse to a single check if ever desired.

---

## D-NNN | YYYY-MM-DD | <next decision template>
**Decision:**
**Why:**
**Alternatives considered:**
**Affects:**
**Reversible?**

---

*DECISIONS.md v1 | Initialized 2026-06-14 | Append-only*
