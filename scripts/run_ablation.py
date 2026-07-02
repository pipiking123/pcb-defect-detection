"""Config-driven training driver for the ablation grid (Item 5).

Reads an ablation YAML config (configs/ablation/*.yaml), applies
environment-variable overrides for data/project/pretrained-weights (the
Colab session sets these), sets IOU_TYPE so src.losses patches BboxLoss
for WIoU v3 when requested, and runs model.train().

See DECISIONS.md D-026 (6-cell ablation grid), D-027 (T4 wall-time
optimization: patience/cache/batch).
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Run one ablation cell from a config YAML.")
    parser.add_argument("--config", type=Path, required=True, help="Path to an ablation config YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print the config without training.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    iou_type = str(cfg.pop("iou_type", "ciou")).lower()
    if iou_type not in {"ciou", "wiou"}:
        raise ValueError(f"iou_type must be 'ciou' or 'wiou', got {iou_type!r}")
    os.environ["IOU_TYPE"] = iou_type

    model = cfg.pop("model")

    # Env-var overrides (env wins over config; error if data still unset)
    data_yaml = os.environ.get("DATA_YAML")
    if data_yaml:
        cfg["data"] = data_yaml
    if not cfg.get("data"):
        raise ValueError("cfg['data'] is unset: set the DATA_YAML env var.")

    project_dir = os.environ.get("PROJECT_DIR")
    if project_dir:
        cfg["project"] = project_dir

    pretrained_path = os.environ.get("PRETRAINED_WEIGHTS")
    yaml_pretrained = cfg.pop("pretrained_weights", None)
    if pretrained_path is None:
        pretrained_path = yaml_pretrained

    if args.dry_run:
        print(f"[ablation] dry-run for config: {args.config}")
        print(f"iou_type: {iou_type}")
        print(f"model: {model}")
        print(f"pretrained_path: {pretrained_path}")
        print("cfg:")
        for k, v in cfg.items():
            print(f"  {k}: {v!r}")
        sys.exit(0)

    import src.losses  # noqa: F401  (patch BboxLoss BEFORE any ultralytics load)
    import src.models  # noqa: F401  (then trigger ultralytics via CoordAtt registration)

    from ultralytics import YOLO

    if pretrained_path:
        model = YOLO(model).load(pretrained_path)
    else:
        model = YOLO(model)

    results = model.train(**cfg)

    try:
        _write_manifest(cfg, iou_type, results)
    except Exception as e:
        print(f"[ablation] WARNING: manifest write failed: {e}")

    print(f"[ablation] {cfg['name']} done. save_dir={results.save_dir}")


def _write_manifest(cfg, iou_type, results):
    """Write a manifest.txt in results.save_dir recording run provenance."""
    import torch
    import ultralytics

    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_head = "unknown"

    cuda_device = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"

    lines = [
        f"Run name: {cfg['name']}",
        f"IOU_TYPE: {iou_type}",
        f"Optimizer: {cfg['optimizer']}",
        f"Batch: {cfg['batch']}",
        f"Imgsz: {cfg['imgsz']}",
        f"Epochs: {cfg['epochs']}",
        f"Seed: {cfg['seed']}",
        f"Git HEAD: {git_head}",
        f"Ultralytics version: {ultralytics.__version__}",
        f"Torch version: {torch.__version__}",
        f"Python version: {sys.version.split()[0]}",
        f"CUDA device: {cuda_device}",
        f"Timestamp UTC: {datetime.now(timezone.utc).isoformat()}",
    ]

    manifest_path = Path(results.save_dir) / "manifest.txt"
    manifest_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
