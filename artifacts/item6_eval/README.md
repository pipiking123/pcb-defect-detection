# AIT304 PCB Ablation Item 6 Deliverable

Generated: eval run on Kaggle P100, 2026-07-03.
Eval script commit: 5c2178a (after split='test' fix)
Training runs git head: fb40c96

## Contents

- `runs_eval/` — full eval output tree
    - `<run>__standard/` (6 folders) — 500-image test set metrics + Ultralytics artifacts
    - `<run>__cleaned/` (6 folders) — 478-image D-024 cleaned test set
    - `summary_standard.csv` — 6-row rollup, standard set (PRIMARY per D-024)
    - `summary_cleaned.csv`  — 6-row rollup, cleaned set (SUPPLEMENTARY)
    - `eval_manifest.txt`    — top-level provenance
    - `demo/` — cross-model qualitative comparison, 4 images × 6 runs + grids

- `deeppcb_kaggle.yaml`         — standard test YAML used at eval
- `deeppcb_kaggle_cleaned.yaml` — cleaned test YAML used at eval
- `cleaned_test_manifest.txt`   — 22 excluded image IDs + SHA-256s
- `demo_paths.txt`              — 4 demo image paths for reproducibility

## Headline results (standard 500-image test set)

Best config: ca_sgd_wiou (CA + WIoU + SGD)
  mAP50    = 0.9450
  mAP50-95 = 0.7582

Deltas vs vanilla baselines:
  vs vanilla_adam_ciou:    +0.68 mAP50 pt   (mAP50-95: -0.55 pt, honest trade-off)
  vs vanilla_adamw_ciou:   +0.64 mAP50 pt   (mAP50-95: +0.04 pt)

## Notes

- Standard vs cleaned differ by only ~0.001 mAP50 across all runs — validates
  that D-024 methodology was principled (no leakage-driven inflation of metrics).
- Class "short" is the consistent worst class across all 6 runs, mAP50 ≈ 0.83.
- Eval used split="test" (not val) — bug fixed at commit 5c2178a.
- Kaggle env: Python 3.12.13, torch 2.10.0+cu128, ultralytics 8.3.40, Tesla T4.
