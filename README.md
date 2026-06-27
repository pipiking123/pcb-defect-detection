# PCB Defect Detection with Coordinate Attention
### AIT304 Advanced Issues of AI (Computer Vision) — Xiamen University Malaysia
**Author:** King | **Target Company:** ViTrox Corporation | **Deadline:** 3 July 2026
**Repository:** [to be added after first push]

---

## Description

This project enhances YOLO11n for automated PCB defect detection by integrating
Coordinate Attention (Hou et al., CVPR 2021) at the P3, P4, and P5 feature maps
of the detection head. Training and evaluation are performed on the DeepPCB dataset
(1,500 image pairs, 6 defect classes: open, short, mousebite, spur, copper, pin-hole),
which reflects the real-world inspection workload of AOI systems used by companies
such as ViTrox Corporation. A 2×2 ablation across optimizer (Adam vs SGD) and loss
function (CIoU vs Wise-IoU) isolates the contribution of each design choice over
four experiments, with the best configuration benchmarked against published SOTA results.

---

## Navigation

- **[PROJECT_STATE.md](PROJECT_STATE.md)** — single source of truth: current phase,
  daily status, master checklist, and active blockers. Read this first when resuming.
- **[DECISIONS.md](DECISIONS.md)** — architectural reasoning log: every major design
  choice with alternatives considered and justification.
- **[PCB_Defect_Detection_Blueprint_v2.md](PCB_Defect_Detection_Blueprint_v2.md)** —
  full technical specification: dataset prep, patches, training config, evaluation protocol.
- **[FILE_DEPENDENCIES.md](FILE_DEPENDENCIES.md)** — file-relationship map: which scripts
  depend on which, and the correct execution order.

---

## Repository Layout

```
src/          Python source files (patches, modules, training scripts)
configs/      YAML architecture configs (yolo11n-CA.yaml, pcb.yaml)
data_prep/    Dataset conversion and sanity-check scripts
notebooks/    Exported Kaggle .ipynb notebooks
results/      Metrics CSVs, charts, confusion matrices (no .pt weights — those go to OneDrive)
```

> Trained weights (.pt files) are stored on OneDrive, not in this repo.

---

## Notebook hygiene

Notebooks are stripped of outputs automatically via nbstripout pre-commit filter.
If committing from outside VS Code (e.g. directly from Colab UI), manually run
**Cell → All Output → Clear** before commit.
