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

## D-NNN | YYYY-MM-DD | <next decision template>
**Decision:**
**Why:**
**Alternatives considered:**
**Affects:**
**Reversible?**

---

*DECISIONS.md v1 | Initialized 2026-06-14 | Append-only*
