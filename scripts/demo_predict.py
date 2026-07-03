"""AIT304 viva demonstration tool for PCB defect detection.

Standalone, read-only, CPU-only inference script. Given a single input
image (local path or HTTP/HTTPS URL), this runs inference through all
six trained ablation models and produces a 2x3 comparison grid PNG plus
a text manifest describing the run. Intended for live viva demos only:
no training, no fine-tuning, and no writes under the weights directory.
"""

from __future__ import annotations

import argparse
import atexit
import platform
import subprocess
import sys
import tempfile
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure repo root is on sys.path so custom modules like
# src.models.coord_attention (referenced by CA-based .pt files)
# resolve correctly regardless of how the script is invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ultralytics import YOLO  # noqa: E402

EXPECTED_RUN_NAMES = (
    "ca_adam_ciou",
    "ca_adam_wiou",
    "ca_sgd_ciou",
    "ca_sgd_wiou",
    "vanilla_adam_ciou_100",
    "vanilla_adamw_ciou_100",
)


class ModelResult:
    """Holds the outcome of running a single ablation model."""

    def __init__(self, run_name: str) -> None:
        self.run_name = run_name
        self.image_rgb: Optional[np.ndarray] = None
        self.num_detections: int = 0
        self.class_counts: Counter = Counter()
        self.error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Local path or http(s):// URL to the input image")
    parser.add_argument("--weights-dir", default="demo_weights/ablation", help="Root directory containing per-run weight folders")
    parser.add_argument("--out", default="demo_output", help="Output directory for the grid PNG and manifest")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    return parser.parse_args()


def is_url(image: str) -> bool:
    return image.startswith("http://") or image.startswith("https://")


def validate_image_arg(image: str) -> None:
    if is_url(image):
        return
    if not Path(image).exists():
        raise FileNotFoundError(f"--image path does not exist: {image}")


def download_image(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "AIT304-demo/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise ValueError(f"URL did not return an image content-type (got: {content_type!r})")
        data = response.read()

    suffix = Path(url.split("?")[0]).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()
    atexit.register(lambda: Path(tmp.name).unlink(missing_ok=True))
    return tmp.name


def discover_weights(weights_dir: Path) -> list[Path]:
    found = sorted(weights_dir.glob("*/weights/best.pt"), key=lambda p: p.parent.parent.name)
    if len(found) < len(EXPECTED_RUN_NAMES):
        found_names = {p.parent.parent.name for p in found}
        missing = [name for name in EXPECTED_RUN_NAMES if name not in found_names]
        print(f"WARNING: expected {len(EXPECTED_RUN_NAMES)} ablation runs, found {len(found)}. Missing: {missing}", file=sys.stderr)
    if not found:
        raise FileNotFoundError(f"No weight files found under {weights_dir}/*/weights/best.pt")
    return found


def run_single_model(weight_path: Path, image_path: str, conf: float, iou: float, imgsz: int) -> ModelResult:
    run_name = weight_path.parent.parent.name
    result = ModelResult(run_name)
    try:
        model = YOLO(str(weight_path))
        predictions = model.predict(source=image_path, conf=conf, iou=iou, imgsz=imgsz, device="cpu", verbose=False)
        prediction = predictions[0]
        annotated_bgr = prediction.plot()
        result.image_rgb = annotated_bgr[:, :, ::-1]

        class_ids = prediction.boxes.cls.tolist()
        result.num_detections = len(class_ids)
        names = prediction.names
        result.class_counts = Counter(names[int(cid)] for cid in class_ids)
    except Exception as exc:  # noqa: BLE001 - per-model isolation is required
        print(f"ERROR: model '{run_name}' failed: {exc}", file=sys.stderr)
        result.error = str(exc)
    return result


def build_grid(results: list[ModelResult], image_basename: str, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    total_detections = sum(r.num_detections for r in results)

    for ax, result in zip(axes.flat, results):
        if result.error is not None:
            ax.text(0.5, 0.5, f"FAILED:\n{result.error}", ha="center", va="center", wrap=True, color="red")
        else:
            ax.imshow(result.image_rgb)
        ax.set_title(f"{result.run_name}\nn_det={result.num_detections}")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes.flat[len(results):]:
        ax.axis("off")

    fig.suptitle(f"{image_basename} — total detections: {total_detections}")
    fig.tight_layout()

    out_path = out_dir / f"{Path(image_basename).stem}_grid.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def get_git_head() -> str:
    try:
        output = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return output.stdout.strip()
    except Exception:  # noqa: BLE001 - manifest must not fail the run
        return "unknown"


def write_manifest(results: list[ModelResult], image_source: str, image_basename: str, out_dir: Path, args: argparse.Namespace) -> Path:
    import ultralytics

    lines = [
        f"Timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"Image source: {image_source}",
        f"Image basename: {image_basename}",
        f"Git HEAD: {get_git_head()}",
        f"Torch version: {torch.__version__}",
        f"Ultralytics version: {ultralytics.__version__}",
        f"Python version: {platform.python_version()}",
        f"Conf: {args.conf}  IoU: {args.iou}  imgsz: {args.imgsz}",
        "",
        "Per-model results:",
    ]
    for result in results:
        if result.error is not None:
            lines.append(f"  {result.run_name}: FAILED ({result.error})")
            continue
        class_breakdown = ", ".join(f"{name}={count}" for name, count in sorted(result.class_counts.items())) or "none"
        lines.append(f"  {result.run_name}: n_det={result.num_detections}, classes=[{class_breakdown}]")

    manifest_path = out_dir / f"{Path(image_basename).stem}_manifest.txt"
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def print_summary(results: list[ModelResult]) -> None:
    print(f"{'run_name':<28} | {'n_det':>5} | classes")
    print("-" * 70)
    for result in results:
        if result.error is not None:
            print(f"{result.run_name:<28} | {'ERR':>5} | {result.error}")
            continue
        class_breakdown = ", ".join(f"{name}={count}" for name, count in sorted(result.class_counts.items())) or "none"
        print(f"{result.run_name:<28} | {result.num_detections:>5} | {class_breakdown}")


def main() -> int:
    args = parse_args()

    try:
        validate_image_arg(args.image)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    image_path = args.image
    if is_url(args.image):
        try:
            image_path = download_image(args.image)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: failed to download image: {exc}", file=sys.stderr)
            return 1

    weights_dir = Path(args.weights_dir)
    try:
        weight_files = discover_weights(weights_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [run_single_model(wf, image_path, args.conf, args.iou, args.imgsz) for wf in weight_files]

    successful = [r for r in results if r.error is None]
    if not successful:
        print("ERROR: zero models loaded successfully.", file=sys.stderr)
        return 1

    image_basename = Path(args.image.split("?")[0]).name

    try:
        grid_path = build_grid(results, image_basename, out_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: failed to save grid image: {exc}", file=sys.stderr)
        return 1

    manifest_path = write_manifest(results, args.image, image_basename, out_dir, args)

    print_summary(results)
    print(f"\nGrid saved to: {grid_path}")
    print(f"Manifest saved to: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
