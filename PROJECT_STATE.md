# PROJECT_STATE.md — AIT304 PCB Defect Detection
> **The single source of truth.** Updated every night before sleep. Read first when resuming.

---

## ⚡ NEW-CHAT RESUME PROTOCOL

If you are a fresh Claude instance reading this for the first time, follow these 4 steps:

1. **Read this entire file.** It is the authoritative state.
2. **Read `DECISIONS.md`** in the same repo for the history of *why*.
3. **Read `PCB_Defect_Detection_Blueprint_v2.md`** for the full technical reference.
4. **Confirm your role:** You are King's planning + decisions partner. You do NOT write code in this chat. Claude Code (in VS Code on the Windows PC) handles all code generation. Your job is to give King the prompts he passes to Claude Code, review results, debug from his screenshots, and update this file.

Then respond with: *"Resumed. Current phase: [phase from below]. Day [N] of 19. Next action: [from below]. What do you need from me?"*

---

## 📋 PROJECT AT A GLANCE

| Item | Value |
|---|---|
| Project | PCB Defect Detection — Enhanced YOLO11 + CA |
| Course | AIT304 Advanced Issues of AI (Computer Vision) |
| Lecturer | Ashwaq Qasem |
| Target company | ViTrox Corporation (Malaysia) |
| Deadline | **3 July 2026, 6:00 PM Malaysia time** |
| Today's date | Tuesday, 2026-06-24 |
| Days remaining | 9 |
| Current phase | **Phase 3 — Day 3 in progress (P1 + P2 complete, P3 next)** |
| Repo URL | https://github.com/pipiking123/pcb-defect-detection |
| OneDrive folder | C:\Users\User\OneDrive - 厦门大学(马来西亚分校)\AIT304_PCB_Final |
| Latest git commit | d09f513: Day 3 P2: sanity checker merged |
| Kaggle notebook URL | _add after creation_ |

---

## Contribution Claim (LOCKED — Day 2, 2026-06-17)

> "This work investigates the integration of vanilla Coordinate Attention (Hou et al., CVPR 2021) with the YOLOv11 backbone for PCB defect detection on the DeepPCB benchmark, evaluated across a 2×2 ablation of optimizer (Adam, SGD) and loss function (CIoU, WIoU). We differentiate from PCB-YOLO (Liu et al., PLOS ONE) which uses the FFCA multi-branch variant with YOLOv8, and from YOLOv11-PCB (Scientific Reports) which uses EMA+CARAFE attention with EIoU loss on the same backbone."

**Status:** LOCKED. Do not modify without explicit approval.
**Source of differentiation:** Verified via NotebookLM cross-paper synthesis + manual confirmation of FFCA-vs-vanilla-CA distinction in PCB-YOLO paper (pages 3, 10–11).
**Use:** This is the sentence to defend in viva and to anchor §1 (Introduction) and §2 (Related Work) of the CVPR report.

---

## 🎯 TODAY'S STATUS

> **Update this section every night before sleep. Takes 2 minutes.**

**Day 3 Phase 2 complete.** Commit: d09f513. `src/data/sanity_check.py` merged. Class-stratified sampling (seed=42), dual-view PNG rendering (image_boxes + label_only), hard coverage gate (exit 2 = sampling bug, exit 3 = converter bug). Codex re-review: APPROVE WITH MINOR FIXES — re-scan elimination intentionally not applied per D-017. Pipeline not yet executed against real data — execution gate is Phase 3.

**Next:** Day 3 Phase 3 — execute converter + sanity checker on real DeepPCB data + eyeball PNGs (visual verification gate before any training code or Colab work).

**Pending Day 3 phases:**
- P3: execute converter + sanity check on real DeepPCB data + eyeball PNGs (gate)
- P4: Colab notebook + 5-epoch smoke test
- P5: close-out commit

---

## ✅ MASTER PHASE CHECKLIST

### Phase 1 — Setup (Day 1, today)
- [ ] Kaggle account created + phone verified
- [x] GitHub repo created: `pipiking123/pcb-defect-detection` (private)
- [x] Local project folder initialized
- [x] DeepPCB cloned from `github.com/tangsanli5201/DeepPCB`
- [ ] DeepPCB uploaded to Kaggle as private dataset
- [ ] Zotero library created with 9 verified references (see Blueprint §2)
- [x] OneDrive folder created for final deliverables
- [x] PROJECT_STATE.md + DECISIONS.md + WORKFLOW.md committed to repo

### Phase 2 — Literature Review (Day 2)
- [ ] All 9 papers downloaded into Zotero
- [ ] Note fields filled for each (problem / method / dataset / mAP / limitation / how-yours-differs)
- [ ] Mental summary written for top 5 priority papers

### Phase 3 — Data + Patches (Day 3)
- [ ] `wiou_patch.py` created and committed
- [ ] `coord_attention.py` created and committed
- [ ] `register_ca.py` created and committed
- [ ] `yolo11n-CA.yaml` created and committed
- [x] `convert_dataset.py` created, committed, and run on Kaggle (merged at ee58602)
- [x] `sanity_check.py` created and committed (merged at d09f513)
- [ ] `sanity_check.py` run — bounding boxes verified to wrap defects
- [ ] CA registration test: `ca_count == 3` ✓ confirmed
- [ ] WIoU patch test: "✓ WIoU patch applied" message confirmed

### Phase 4 — Training (Days 4–8)
- [ ] EXP-1: YOLOv8n + Adam + CIoU — `best_exp1.pt` saved
- [ ] EXP-3: YOLO11n + CA + Adam + CIoU — `best_exp3.pt` saved (ca_count==3 verified)
- [ ] **Kernel restart confirmed**
- [ ] EXP-2: YOLOv8n + SGD + WIoU — `best_exp2.pt` saved (WIoU msg confirmed)
- [ ] EXP-4: YOLO11n + CA + SGD + WIoU — `best_exp4.pt` saved (both verifications)
- [ ] All 4 `results.csv` and `results.png` downloaded to local repo
- [ ] All 4 `best.pt` uploaded to OneDrive backup

### Phase 5 — Evaluation (Days 9–12)
- [ ] `evaluate.py` run on all 4 models → `comparison_full.csv`
- [ ] FPS measured for all 4 models
- [ ] Per-class mAP chart generated
- [ ] Speed-accuracy scatter generated
- [ ] Confusion matrix (best model, normalised) generated
- [ ] 3–4 failure case images captured + annotated with brief notes

### Phase 6 — Report (Days 13–18)
- [ ] CVPR .docx template opened (Dr. Ashwaq's file)
- [ ] OneDrive link inserted on first page
- [ ] Abstract drafted
- [ ] §1 Introduction drafted
- [ ] §2 Related Work drafted
- [ ] §3 Methodology drafted (with 2 figures: sample images + architecture diagram)
- [ ] §4 Results & Discussion drafted (with 5 figures: training curves, comparison chart, per-class, speed-accuracy, confusion matrix, failure cases)
- [ ] §5 Conclusion drafted
- [ ] References ≥ 9, IEEE format
- [ ] Marking rubric appended (assignment requirement)
- [ ] Page count 6–7 confirmed (excluding refs)

### Phase 7 — Submit (Days 19–20)
- [ ] Final grammar pass
- [ ] Final figure quality check (150+ dpi)
- [ ] Final OneDrive folder contents verified (code .ipynb + best.pt + dataset + report.docx)
- [ ] Report renamed to `[student_ID].docx`
- [ ] Uploaded to Moodle BEFORE 3 July 6 PM
- [ ] Screenshot of submission saved

---

## 🔑 CRITICAL FACTS (NEVER LOSE THESE)

These are the non-negotiable constraints. If a new Claude instance contradicts them, the new Claude is wrong.

- **Dataset:** DeepPCB, 1,500 image pairs, 6 classes (open, short, mousebite, spur, copper, pin-hole), 640×640 grayscale binarised
- **Split:** 800 train / 200 val / 500 test (test = official DeepPCB test split, untouched)
- **Class ID conversion:** DeepPCB uses 1..6, YOLO needs 0..5 → `cls = cls - 1` in `convert_dataset.py`
- **Experiments:** 2×2 ablation — {YOLOv8n, YOLO11n+CA} × {Adam-CIoU, SGD-WIoU}
- **Model name:** `yolo11n.pt` NOT `yolov11n.pt` (Ultralytics dropped the "v")
- **Ultralytics version:** Pin to `8.3.40` (newer versions may break the monkey-patches)
- **Reproducibility:** seed=42 everywhere, no exceptions
- **CA verification:** Before every EXP-3 / EXP-4 run, assert `ca_count == 3`. If not, the YAML didn't parse correctly.
- **WIoU verification:** Before every EXP-2 / EXP-4 run, the patch must print `✓ WIoU patch applied`
- **Kernel order:** Run CIoU experiments first (EXP-1, EXP-3), then restart Kaggle kernel ONCE, then run WIoU experiments (EXP-2, EXP-4)
- **Weights backup:** Immediately copy `best.pt` to `/kaggle/working/best_expN.pt` after each run — Kaggle session storage is volatile
- **Augmentation:** `hsv_h=0, hsv_s=0` (grayscale), `flipud=0` (orientation-sensitive)
- **Hyperparameters:** epochs=100, batch=16, imgsz=640, patience=20, Adam lr0=0.001, SGD lr0=0.01
- **No editing of YOLO source files.** All modifications via monkey-patch or YAML registration only.
- **Cost:** RM 0 (entirely free, see Blueprint §12)

---

## 📝 RECENT DECISIONS (last 5, reverse chronological)

> When making a new architectural / methodological choice, add it here AND append the full reasoning to `DECISIONS.md`.

- 2026-06-24 — Accept duplicate build_class_index() scan in sanity_check.py (clarity over non-measurable speedup) — see DECISIONS.md #017
- 2026-06-24 — Compute platform formally amended: Kaggle → Google Colab (Kaggle GPU blocked, D-011 root cause) — see DECISIONS.md #007-amendment
- 2026-06-24 — Accept defensive redundancy in validation invariants 1a/1b (belt-and-braces on pipeline critical path) — see DECISIONS.md #016
- 2026-06-24 — Honor official trainval/test split; 80/20 val carve at seed=42 — see DECISIONS.md #015
- 2026-06-24 — Single tested-image input (no template) — see DECISIONS.md #014
- 2026-06-24 — Ultralytics version pin = 8.3.40 — see DECISIONS.md #013
- 2026-06-24 — D-010 literal path fixes superseded by flat-index implementation — see DECISIONS.md #010-amendment
- 2026-06-14 — DeepPCB structure verified; 3 path fixes captured for Day 3 convert_dataset.py — see DECISIONS.md #010

---

## 📅 DAILY LOG

### Day 2 — 2026-06-17

- Kaggle phone verification failed → pivoting to Google Colab (see D-011)
- Zotero: 9/9 papers uploaded to AIT304_PCB_Project collection, all tagged with methodology/dataset/mAP-result
- NotebookLM adopted as comprehension + cross-paper synthesis tool (see D-012)
- PCB-YOLO paper 7 manually verified: FFCA confirmed as multi-branch CA variant (not vanilla CA) — differentiation sharpened
- Contribution claim locked (see Contribution Claim section above)

---

## 🚨 ACTIVE BLOCKERS / RISKS

> Things that could derail the project. Update when new ones appear.

- _none currently_

---

## 🛠️ TOOL ROLES (DO NOT VIOLATE)

| Tool | Role | Forbidden |
|---|---|---|
| **Claude.ai (this chat)** | Decisions, planning, prompts for Claude Code, debugging from screenshots, reviewing CSVs | Generating production code files |
| **Claude Code (VS Code)** | All `.py` / `.yaml` file creation, git commits, local dataset conversion, sanity checks | Long-running training (don't tie up PC) |
| **Codex (VS Code)** | Second-opinion code review on `wiou_patch.py` and `yolo11n-CA.yaml` only | New file generation (avoid conflict with Claude Code) |
| **Claude Cowork (desktop)** | Zotero, OneDrive uploads, screenshot organization, Word doc assembly from CSV | Code generation |
| **Kaggle Notebook** | Training + evaluation only. Paste finished files; output `.pt` + `.csv` | Code authoring |
| **GitHub** | Source of truth. Every code change is committed. | Storing trained weights (use OneDrive — too large for git) |

---

## 🌙 END-OF-DAY RITUAL (5 MIN, MANDATORY)

Every night before sleep, in this order:

1. **Update this file**: today's status block + checklist boxes + recent decisions if any
2. **Commit + push to GitHub:**
   ```bash
   cd /home/king/pcb_project
   git add .
   git commit -m "Day N: <one-line summary>"
   git push
   ```
3. **Speak tomorrow's first action out loud.** (Kills morning startup friction.)

If you skip days 1 and 2, you risk losing context on chat transfer. If you skip day 3, you'll spend 30 min rediscovering where you were tomorrow morning.

---

## 📤 HANDOFF CHECKLIST (BEFORE OPENING A NEW CLAUDE CHAT)

When this chat hits its context limit and you need to start fresh:

1. ✅ This file is updated with current status
2. ✅ Latest git commit pushed
3. ✅ Open new Claude.ai chat
4. ✅ Paste the contents of THIS FILE as the first message
5. ✅ Add: "Continue from here. Today is [date]. Day [N] of 19."
6. ✅ Wait for the resume confirmation message

The new Claude will have zero context drift if you do this. The blueprint + state file + decisions file give it everything.

---

*PROJECT_STATE.md v1 | Initialized [date] | Updated daily*
