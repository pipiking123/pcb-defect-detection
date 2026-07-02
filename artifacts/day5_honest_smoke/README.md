# Day 5 — Honest Smoke Test (Plate-Aware Split)

**Naming provenance:** This is the Day 5 honest re-run on plate-aware data. Run name (day4_vanilla) was inherited from the notebook's Day 4b training cell which was reused unmodified — only the underlying data pipeline changed (plate-aware split, commit a02964b). Same training code, honest data. Save_dir in args.yaml reflects the inherited name.

- **Date:** 2 July 2026
- **Git HEAD at time of run:** `0b879cf` (day4-notebook preflight fix)
- **Config:** Ultralytics 8.3.40 (from stdout banner; not recorded in args.yaml), seed=42, epochs=5, batch=16, imgsz=640, optimizer=auto (resolved to AdamW), CIoU default. Per D-018: all geometric and photometric detection augmentations disabled (hsv_h/s/v=0, degrees=0, translate=0, scale=0, shear=0, perspective=0, mosaic=0, mixup=0, copy_paste=0, bgr=0). Only fliplr=0.5 retained. Note: args.yaml shows auto_augment=randaugment and erasing=0.4 — these are Ultralytics defaults that are inert for detection tasks (they apply only to classification).
- **Train/val:** 83 plates / 801 imgs (train), 20 plates / 199 imgs (val), plate-disjoint verified
- **Final val mAP50:** 0.9267
- **Final val mAP50-95:** 0.703

## Per-class val mAP50

| Class     | mAP50 |
|-----------|-------|
| open      | 0.935 |
| short     | 0.851 |
| mousebite | 0.902 |
| spur      | 0.922 |
| copper    | 0.972 |
| pin-hole  | 0.978 |

Source: Colab stdout at end-of-training validation. Not present in results.csv (Ultralytics writes only aggregate metrics to that file).

## Checkpoint

The trained checkpoint (5.3 MB) is retained in Drive at /content/drive/MyDrive/pcb-defect-detection/runs/smoke/day4_vanilla/weights/best.pt but not committed (repo policy excludes .pt files). Regeneratable by re-running the notebook at HEAD 0b879cf.

## Contents

- `results.csv` — per-epoch training/validation metrics
- `args.yaml` — full Ultralytics run configuration
- `confusion_matrix.png` — val confusion matrix
- `labels.jpg` — label distribution plot

Pipeline-honesty smoke test on plate-aware data. NOT the baseline of record — vanilla-100 in Item 5 ablation is the baseline for comparison.
