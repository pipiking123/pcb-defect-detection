"""eval_ablation.py -- evaluate the 6-cell ablation grid (Item 6).

Runs each requested run's best.pt against the standard 500-image DeepPCB
test set and/or the D-024 cleaned 478-image test set, and optionally
produces a qualitative cross-model demo comparison on a fixed set of
images. All paths are CLI args; the script does not assume Colab or
Kaggle specifically, only that it is executed from the repository root
(see the sys.path note below).

Runtime note (src.models / src.losses import order): CA runs (ca_*) were
trained with a custom CoordAtt module registered into ultralytics.nn.tasks
via `import src.models` (see src/models/__init__.py). Loading their best.pt
requires that registration to have already happened, exactly as in
scripts/train.py and scripts/run_ablation.py -- otherwise parse_model()
raises on the CoordAtt layer when Ultralytics reconstructs the architecture
from the checkpoint. `import src.losses` is included for the same
before-any-ultralytics-model-load ordering discipline, even though the
WIoU v3 patch only affects the training-time loss term, not val()/predict().
"""

import sys
from pathlib import Path

# Ensure the repo root (parent of scripts/) is importable regardless of how
# this script is invoked (`python scripts/eval_ablation.py ...` puts only
# scripts/ on sys.path[0], not the repo root) -- required for `import src.*`
# below. See "Design decisions" note in the accompanying report.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from typing import Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import src.models  # noqa: E402,F401  (registers CoordAtt into ultralytics.nn.tasks)
import src.losses  # noqa: E402,F401  (patches BboxLoss for WIoU v3 if IOU_TYPE=wiou; no-op here)

import torch  # noqa: E402
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_ABLATION_DIR = REPO_ROOT / "configs" / "ablation"
IMG_SUFFIXES = (".jpg", ".jpeg", ".png")
STANDARD_TEST_COUNT = 500
CLEANED_TEST_COUNT = 478
SUMMARY_FIELDNAMES = [
    "run_name", "optimizer", "iou_type", "mAP50", "mAP50-95", "precision", "recall",
    "best_class_mAP50", "worst_class_mAP50", "worst_class",
]


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> str:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=REPO_ROOT)
        return "yes" if out.strip() else "no"
    except Exception:
        return "unknown"


def _cuda_info(device: str) -> str:
    try:
        if torch.cuda.is_available():
            idx = 0
            if device not in (None, "cpu") and device.isdigit():
                idx = int(device)
            return torch.cuda.get_device_name(idx)
        return "cpu"
    except Exception:
        return "unknown"


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_names(names) -> Dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


def _resolve_yaml_field(dataset_path: Path, field_value: str) -> Path:
    p = Path(field_value)
    if p.is_absolute():
        return p
    return (dataset_path / field_value).resolve()


def _list_split_images(resolved: Path) -> List[Path]:
    """Resolve a data-yaml split value to a sorted image-path list.

    Mirrors ultralytics.data.base.BaseDataset.get_img_files: a directory is
    globbed; a .txt file is read as a newline-delimited image-path list
    ('./'-prefixed lines are relative to the txt file's parent).
    """
    if resolved.is_dir():
        return sorted(p for p in resolved.rglob("*") if p.is_file() and p.suffix.lower() in IMG_SUFFIXES)
    if resolved.is_file():
        lines = [ln.strip() for ln in resolved.read_text(encoding="utf-8").splitlines() if ln.strip()]
        out = []
        for ln in lines:
            out.append((resolved.parent / ln[2:]).resolve() if ln.startswith("./") else Path(ln))
        return sorted(out)
    raise FileNotFoundError(f"Resolved split path does not exist: {resolved}")


def _resolve_test_split(yaml_path: Path, data: dict) -> List[Path]:
    dataset_path = Path(data["path"])
    if not dataset_path.is_absolute():
        dataset_path = (yaml_path.resolve().parent / dataset_path).resolve()
    resolved = _resolve_yaml_field(dataset_path, data["test"])
    return _list_split_images(resolved)


def _label_path_for_image(img_path: Path) -> Optional[Path]:
    """Map an images/<split>/x.jpg path to labels/<split>/x.txt via the
    standard Ultralytics 'images'->'labels' directory-segment convention
    (matches src/data/convert_dataset.py's output layout)."""
    parts = list(img_path.parts)
    try:
        idx = len(parts) - 1 - parts[::-1].index("images")
    except ValueError:
        return None
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def _read_yolo_labels(label_path: Optional[Path]) -> List[Tuple[int, float, float, float, float]]:
    if label_path is None or not label_path.is_file():
        return []
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        toks = line.strip().split()
        if len(toks) != 5:
            continue
        c = int(toks[0])
        cx, cy, w, h = (float(t) for t in toks[1:])
        out.append((c, cx, cy, w, h))
    return out


def _count_instances(image_paths: List[Path], nc: int) -> Tuple[Dict[int, int], int]:
    """Ground-truth instance counts per class, computed directly from the
    label files on disk for the resolved test split.

    Deliberately independent of ultralytics validator internals: model.val()
    only returns validator.metrics (a DetMetrics instance), and nt_per_class
    lives on the validator object itself, which is not exposed by that
    return value (see DesignDecision: per-class instance counts)."""
    counts = {c: 0 for c in range(nc)}
    total = 0
    for img in image_paths:
        for c, *_ in _read_yolo_labels(_label_path_for_image(img)):
            counts[c] = counts.get(c, 0) + 1
            total += 1
    return counts, total


def _default_run_names() -> List[str]:
    return sorted(p.stem for p in CONFIGS_ABLATION_DIR.glob("*.yaml"))


def _compute_env_provenance(args: argparse.Namespace) -> dict:
    """Environment-level provenance shared by every pair in this invocation
    of the script. Computed once (Codex fix 3), not recomputed per pair --
    git/version/device lookups are the same for the whole sweep."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "git_dirty": _git_dirty() == "yes",
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "python_version": sys.version.split()[0],
        "device": _cuda_info(args.device),
    }


def _train_provenance(run_name: str, runs_dir: Path) -> Tuple[str, str]:
    """Pull optimizer + iou_type for a run.

    optimizer comes from <runs_dir>/<run>/args.yaml (standard Ultralytics
    train output field). iou_type is NOT part of Ultralytics' args.yaml
    schema (iou_type is a driver-only key popped before model.train() is
    called -- verified against the committed artifacts/day4_vanilla/args.yaml
    and artifacts/day5_honest_smoke/args.yaml, neither has an iou_type
    field). Fall back chain: manifest.txt (written by run_ablation.py's
    _write_manifest, if present in the run dir) -> the repo's own
    configs/ablation/<run>.yaml (source of truth for what was requested)
    -> 'unknown' with a warning.
    """
    optimizer = "unknown"
    args_yaml_path = runs_dir / run_name / "args.yaml"
    if args_yaml_path.is_file():
        d = _load_yaml(args_yaml_path)
        optimizer = str(d.get("optimizer", "unknown"))
    else:
        print(f"[eval_ablation] WARNING: no args.yaml found for run '{run_name}'; optimizer=unknown", file=sys.stderr)

    iou_type = "unknown"
    manifest_path = runs_dir / run_name / "manifest.txt"
    if manifest_path.is_file():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip().lower().startswith("iou_type:"):
                iou_type = line.split(":", 1)[1].strip()
                break
    if iou_type == "unknown":
        cfg_path = CONFIGS_ABLATION_DIR / f"{run_name}.yaml"
        if cfg_path.is_file():
            iou_type = str(_load_yaml(cfg_path).get("iou_type", "unknown"))
        else:
            print(f"[eval_ablation] WARNING: could not resolve iou_type for run '{run_name}'", file=sys.stderr)

    return optimizer, iou_type


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def preflight(args: argparse.Namespace) -> dict:
    if ultralytics.__version__ != args.ultralytics_version:
        fail(
            f"Ultralytics version mismatch: found {ultralytics.__version__}, "
            f"expected {args.ultralytics_version} (D-013 pin)."
        )

    for label, p in (("--data-standard", args.data_standard), ("--data-cleaned", args.data_cleaned)):
        if not p.is_file():
            fail(f"{label} does not exist: {p}")

    std_yaml = _load_yaml(args.data_standard)
    cln_yaml = _load_yaml(args.data_cleaned)
    for label, y, p in (("--data-standard", std_yaml, args.data_standard), ("--data-cleaned", cln_yaml, args.data_cleaned)):
        for key in ("path", "train", "val", "test", "nc", "names"):
            if key not in y:
                fail(f"{label} ({p}) missing required key '{key}'.")

    std_images = _resolve_test_split(args.data_standard, std_yaml)
    cln_images = _resolve_test_split(args.data_cleaned, cln_yaml)
    if len(std_images) != STANDARD_TEST_COUNT:
        fail(f"--data-standard test split resolved to {len(std_images)} images, expected {STANDARD_TEST_COUNT}.")
    if len(cln_images) != CLEANED_TEST_COUNT:
        fail(f"--data-cleaned test split resolved to {len(cln_images)} images, expected {CLEANED_TEST_COUNT} (D-024).")

    run_names = _default_run_names() if args.runs == ["all"] else list(dict.fromkeys(args.runs))
    if not run_names:
        fail("No runs resolved. Pass --runs all or explicit run names.")

    weights_paths: Dict[str, Path] = {}
    missing = []
    for r in run_names:
        wp = args.runs_dir / r / "weights" / "best.pt"
        if not wp.is_file():
            missing.append(str(wp))
        else:
            weights_paths[r] = wp
    if missing:
        fail("Missing weights/best.pt for requested run(s):\n  " + "\n  ".join(missing))

    std_names = _normalize_names(std_yaml["names"])
    cln_names = _normalize_names(cln_yaml["names"])
    if std_names != cln_names:
        fail(f"Class names differ between --data-standard ({std_names}) and --data-cleaned ({cln_names}).")

    first_run = run_names[0]
    first_model = YOLO(str(weights_paths[first_run]))
    ckpt_names = _normalize_names(first_model.names)
    if ckpt_names != std_names:
        fail(
            f"Class names in checkpoint for run '{first_run}' ({ckpt_names}) do not match "
            f"the data-yaml class names ({std_names}). Note: class names are read from the "
            f"checkpoint (model.names), not from args.yaml -- Ultralytics' train args.yaml "
            f"schema has no names/nc field (verified against artifacts/day4_vanilla/args.yaml "
            f"and artifacts/day5_honest_smoke/args.yaml)."
        )

    if args.mode in ("demo", "both"):
        if not args.demo_images:
            fail("--mode includes demo but --demo-images is empty.")
        demo_missing = [str(p) for p in args.demo_images if not p.is_file()]
        if demo_missing:
            fail("Missing --demo-images file(s):\n  " + "\n  ".join(demo_missing))

    return {
        "run_names": run_names,
        "weights_paths": weights_paths,
        "std_yaml": std_yaml,
        "cln_yaml": cln_yaml,
        "std_images": std_images,
        "cln_images": cln_images,
        "names": std_names,
        "first_model": first_model,
        "first_run": first_run,
    }


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------


def _per_class_metrics(results, names: Dict[int, str]) -> Dict[int, dict]:
    box = results.box
    per_class = {c: {"mAP50": 0.0, "mAP50-95": 0.0, "precision": 0.0, "recall": 0.0} for c in names}
    ap_class_index = [int(c) for c in box.ap_class_index]
    for i, c in enumerate(ap_class_index):
        per_class[c] = {
            "mAP50": float(box.ap50[i]),
            "mAP50-95": float(box.ap[i]),
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
        }
    return per_class


def run_eval_pair(
    model, run_name: str, test_set: str, data_yaml_path: Path, image_paths: List[Path],
    names: Dict[int, str], train_optimizer: str, train_iou_type: str, weights_path: Path,
    weights_sha256: str, env_provenance: dict, output_dir: Path, args: argparse.Namespace,
) -> Tuple[Optional[dict], Optional[dict]]:
    """Run one (run, test_set) pair end-to-end.

    Returns (summary_row, failure). Exactly one of the two is None. On any
    failure anywhere in the pair body -- val(), metric extraction, instance
    counting, hashing, or the metrics.json/manifest.txt writes -- the
    exception is logged with a full traceback and a failure record is
    returned instead of raising, so the caller can continue to the next pair
    (Codex fix 2). Model loading / checkpoint hashing / train-provenance
    lookup happen once per run in main(), guarded by their own try/except
    there, so a bad checkpoint doesn't get retried and re-fail per pair.
    """
    pair_name = f"{run_name}__{test_set}"
    try:
        results = model.val(
            data=str(data_yaml_path), imgsz=args.imgsz, batch=args.batch, conf=args.conf, iou=args.iou,
            project=str(output_dir), name=pair_name, exist_ok=False, plots=True, save_json=True,
            verbose=True, seed=args.seed, device=args.device,
        )

        save_dir = Path(results.save_dir)
        if save_dir.name != pair_name:
            print(
                f"[eval_ablation] WARNING: pair '{pair_name}' save_dir auto-incremented to "
                f"'{save_dir.name}' (a prior directory with the expected name already existed).",
                file=sys.stderr,
            )

        per_class = _per_class_metrics(results, names)
        instance_counts, total_instances = _count_instances(image_paths, len(names))
        for c in per_class:
            per_class[c]["instances"] = instance_counts.get(c, 0)

        overall = {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
            "fitness": float(results.fitness),
        }

        data_yaml_sha256 = _sha256(data_yaml_path)

        provenance = {
            **env_provenance,
            "train_optimizer": train_optimizer,
            "train_iou_type": train_iou_type,
        }

        metrics_payload = {
            "run_name": run_name,
            "test_set": test_set,
            "provenance": provenance,
            "test_set_image_count": len(image_paths),
            "test_set_instance_count": total_instances,
            "checkpoint": str(weights_path),
            "checkpoint_sha256": weights_sha256,
            "data_yaml": str(data_yaml_path),
            "data_yaml_sha256": data_yaml_sha256,
            "overall": overall,
            "per_class": {names[c]: per_class[c] for c in sorted(per_class)},
            "eval_config": {
                "imgsz": args.imgsz, "batch": args.batch, "conf": args.conf, "iou": args.iou,
                "seed": args.seed, "device": args.device,
            },
        }
        (save_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        manifest_lines = [
            f"Timestamp UTC: {provenance['timestamp_utc']}",
            f"Git HEAD: {provenance['git_head']}",
            f"Git dirty: {provenance['git_dirty']}",
            f"Ultralytics version: {provenance['ultralytics_version']}",
            f"Torch version: {provenance['torch_version']}",
            f"CUDA version: {provenance['cuda_version']}",
            f"Device: {provenance['device']}",
            f"Python version: {provenance['python_version']}",
            f"Run name: {run_name}",
            f"Checkpoint: {weights_path}",
            f"Checkpoint SHA-256: {weights_sha256}",
            f"Train optimizer: {train_optimizer}",
            f"Train iou_type: {train_iou_type}",
            f"Test set: {test_set}",
            f"Data YAML: {data_yaml_path}",
            f"Data YAML SHA-256: {data_yaml_sha256}",
            f"Image count: {len(image_paths)}",
            f"Instance count: {total_instances}",
            f"Eval args: imgsz={args.imgsz} batch={args.batch} conf={args.conf} iou={args.iou} "
            f"seed={args.seed} device={args.device}",
            f"Results: mAP50={overall['mAP50']:.4f} mAP50-95={overall['mAP50-95']:.4f} "
            f"precision={overall['precision']:.4f} recall={overall['recall']:.4f} fitness={overall['fitness']:.4f}",
        ]
        (save_dir / "manifest.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

        ap50_values = {names[c]: per_class[c]["mAP50"] for c in per_class}
        best_class = max(ap50_values, key=ap50_values.get)
        worst_class = min(ap50_values, key=ap50_values.get)

        row = {
            "run_name": run_name,
            "optimizer": train_optimizer,
            "iou_type": train_iou_type,
            "mAP50": overall["mAP50"],
            "mAP50-95": overall["mAP50-95"],
            "precision": overall["precision"],
            "recall": overall["recall"],
            "best_class_mAP50": ap50_values[best_class],
            "worst_class_mAP50": ap50_values[worst_class],
            "worst_class": worst_class,
        }
        return row, None
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[eval_ablation] ERROR: pair '{pair_name}' failed: {e}", file=sys.stderr)
        print(tb, file=sys.stderr)
        failure = {"pair": pair_name, "run_name": run_name, "test_set": test_set, "error": str(e), "traceback": tb}
        return None, failure


def write_summary_csv(rows: List[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _deviations(args: argparse.Namespace) -> List[str]:
    bullets = []
    if args.ultralytics_version != "8.3.40":
        bullets.append(f"- ultralytics_version={args.ultralytics_version} (D-013 pin is 8.3.40)")
    if args.seed != 42:
        bullets.append(f"- seed={args.seed} (project-wide reproducibility policy is seed=42)")
    if args.imgsz != 640:
        bullets.append(f"- imgsz={args.imgsz} (project standard is 640)")
    return bullets


def write_eval_manifest(
    path: Path, args: argparse.Namespace, ctx: dict, test_sets: List[str],
    failures: List[dict], elapsed_seconds: float, env_provenance: dict,
) -> None:
    deviations = _deviations(args)
    lines = [
        f"Timestamp UTC: {env_provenance['timestamp_utc']}",
        f"Git HEAD: {env_provenance['git_head']}",
        f"Git dirty: {env_provenance['git_dirty']}",
        f"Ultralytics version: {env_provenance['ultralytics_version']}",
        f"Torch version: {env_provenance['torch_version']}",
        f"CUDA version: {env_provenance['cuda_version']}",
        f"Device: {env_provenance['device']}",
        f"Runs: {ctx['run_names']}",
        f"Test sets: {test_sets}",
        f"Standard test image count: {len(ctx['std_images'])} (expected {STANDARD_TEST_COUNT})",
        f"Cleaned test image count: {len(ctx['cln_images'])} (expected {CLEANED_TEST_COUNT}, D-024)",
        f"Eval args: imgsz={args.imgsz} batch={args.batch} conf={args.conf} iou={args.iou} "
        f"seed={args.seed} device={args.device}",
        f"Wall-clock: {elapsed_seconds:.1f}s",
        "Failures:",
    ]
    if failures:
        for f in failures:
            lines.append(f"  {f['run_name']} / {f['test_set']}: {f['error']}")
    else:
        lines.append("  (none)")
    lines.append("Config deviations from D-013/D-018/D-024 defaults:")
    lines += deviations if deviations else ["  (none)"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------


def _yolo_to_xyxy(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    return (cx - w / 2) * img_w, (cy - h / 2) * img_h, (cx + w / 2) * img_w, (cy + h / 2) * img_h


def _draw_boxes(img_bgr: np.ndarray, boxes: List[Tuple[float, float, float, float]], labels: List[str],
                 color: Tuple[int, int, int], thickness: int = 2, font_scale: float = 0.5) -> np.ndarray:
    import cv2

    out = img_bgr.copy()
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, thickness)
        cv2.putText(out, label, (p1[0], max(p1[1] - 4, 0)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
    return out


def _compose_grid(panels: List[Tuple[str, np.ndarray]], target_h: int = 320) -> np.ndarray:
    import cv2

    strips = []
    for label, img in panels:
        scale = target_h / img.shape[0]
        resized = cv2.resize(img, (max(1, int(img.shape[1] * scale)), target_h))
        strip = np.full((24, resized.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(strip, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        strips.append(np.vstack([strip, resized]))
    return np.hstack(strips)


def run_demo(ctx: dict, args: argparse.Namespace, output_dir: Path, model_cache: Dict[str, "YOLO"]) -> None:
    import cv2

    demo_dir = output_dir / "demo"
    (demo_dir / "source").mkdir(parents=True, exist_ok=True)
    (demo_dir / "predictions").mkdir(parents=True, exist_ok=True)

    names = ctx["names"]
    run_names_sorted = sorted(ctx["run_names"])  # alphabetical, per spec
    demo_manifest_lines = [
        f"Timestamp UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Git HEAD: {_git_head()}",
        f"Runs (alphabetical): {run_names_sorted}",
        f"Demo conf: {args.demo_conf}",
        f"Demo grid: {args.demo_grid}",
        "Per-image summary:",
    ]

    for img_path in args.demo_images:
        stem = img_path.stem
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[eval_ablation] WARNING: could not read demo image {img_path}; skipping.", file=sys.stderr)
            continue
        img_h, img_w = img_bgr.shape[:2]

        cv2.imwrite(str(demo_dir / "source" / f"{stem}.jpg"), img_bgr)

        gt_records = _read_yolo_labels(_label_path_for_image(img_path))
        gt_label_path = _label_path_for_image(img_path)
        has_gt = gt_label_path is not None and gt_label_path.is_file()
        gt_img = None
        if has_gt:
            gt_boxes = [_yolo_to_xyxy(cx, cy, w, h, img_w, img_h) for _, cx, cy, w, h in gt_records]
            gt_labels = [names.get(c, str(c)) for c, *_ in gt_records]
            gt_img = _draw_boxes(img_bgr, gt_boxes, gt_labels, color=(0, 255, 0))
            cv2.imwrite(str(demo_dir / "source" / f"{stem}_gt.jpg"), gt_img)
        else:
            print(f"[eval_ablation] WARNING: no label file found for demo image {img_path}; skipping GT overlay.", file=sys.stderr)

        pred_dir = demo_dir / "predictions" / stem
        pred_dir.mkdir(parents=True, exist_ok=True)

        all_run_predictions = {}
        panel_images = {}
        for run_name in run_names_sorted:
            if run_name not in model_cache:
                model_cache[run_name] = YOLO(str(ctx["weights_paths"][run_name]))
            model = model_cache[run_name]

            results = model.predict(
                source=str(img_path), conf=args.demo_conf, imgsz=args.imgsz,
                augment=False, agnostic_nms=False, save=False, verbose=False, device=args.device,
            )
            r0 = results[0]
            n_det = len(r0.boxes)
            boxes_xyxy = r0.boxes.xyxy.cpu().numpy().tolist() if n_det else []
            confs = r0.boxes.conf.cpu().numpy().tolist() if n_det else []
            clss = r0.boxes.cls.cpu().numpy().astype(int).tolist() if n_det else []
            labels = [f"{names.get(c, str(c))} {conf:.2f}" for c, conf in zip(clss, confs)]

            pred_img = _draw_boxes(img_bgr, boxes_xyxy, labels, color=(0, 0, 255))
            cv2.imwrite(str(pred_dir / f"{run_name}.jpg"), pred_img)
            panel_images[run_name] = pred_img

            all_run_predictions[run_name] = [
                {"cls": c, "name": names.get(c, str(c)), "conf": conf, "xyxy": box}
                for c, conf, box in zip(clss, confs, boxes_xyxy)
            ]

        (pred_dir / "predictions.json").write_text(json.dumps(all_run_predictions, indent=2), encoding="utf-8")

        if args.demo_grid:
            panels = [("source", img_bgr), ("GT", gt_img if gt_img is not None else img_bgr)]
            panels += [(rn, panel_images[rn]) for rn in run_names_sorted]
            grid_img = _compose_grid(panels)
            cv2.imwrite(str(demo_dir / f"{stem}_grid.png"), grid_img)

        demo_manifest_lines.append(
            f"  {stem}: gt={'yes' if has_gt else 'no'} detections={{{', '.join(f'{r}={len(all_run_predictions[r])}' for r in run_names_sorted)}}}"
        )

    (demo_dir / "demo_manifest.txt").write_text("\n".join(demo_manifest_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the 6-cell ablation grid (Item 6).")
    parser.add_argument("--runs-dir", required=True, type=Path, help="Read-only dir of <run>/weights/best.pt trees.")
    parser.add_argument("--runs", nargs="+", default=["all"], help="Run names to evaluate, or 'all' (default).")
    parser.add_argument("--test-set", choices=["standard", "cleaned", "both"], default="both")
    parser.add_argument("--data-standard", required=True, type=Path, help="Standard 500-image test data YAML.")
    parser.add_argument("--data-cleaned", required=True, type=Path, help="D-024 cleaned 478-image test data YAML.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--mode", choices=["eval", "demo", "both"], default="eval")
    parser.add_argument("--demo-images", nargs="*", type=Path, default=[])
    parser.add_argument("--demo-conf", type=float, default=0.25)
    parser.add_argument("--demo-grid", action="store_true")
    parser.add_argument("--ultralytics-version", type=str, default="8.3.40", help="Expected version (D-013 pin).")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print the plan; call no val()/predict().")
    return parser


def _print_dry_run_plan(args: argparse.Namespace, ctx: dict) -> None:
    test_sets = []
    if args.test_set in ("standard", "both"):
        test_sets.append("standard")
    if args.test_set in ("cleaned", "both"):
        test_sets.append("cleaned")

    print("[eval_ablation] DRY RUN -- no val()/predict() will be called.")
    print(f"  ultralytics: {ultralytics.__version__}  torch: {torch.__version__}")
    print(f"  runs ({len(ctx['run_names'])}): {ctx['run_names']}")
    for r in ctx["run_names"]:
        print(f"    {r}: {ctx['weights_paths'][r]}")
    print(f"  class names: {ctx['names']}")
    print(f"  mode: {args.mode}  test_sets: {test_sets}")
    print(f"  standard test images: {len(ctx['std_images'])} ({args.data_standard})")
    print(f"  cleaned test images: {len(ctx['cln_images'])} ({args.data_cleaned})")
    print(f"  output_dir: {args.output_dir}")
    print(f"  eval config: imgsz={args.imgsz} batch={args.batch} conf={args.conf} iou={args.iou} "
          f"seed={args.seed} device={args.device}")
    if args.mode in ("demo", "both"):
        print(f"  demo images ({len(args.demo_images)}): {[str(p) for p in args.demo_images]}")
        print(f"  demo_conf: {args.demo_conf}  demo_grid: {args.demo_grid}")


def main() -> None:
    args = build_parser().parse_args()

    # Seeds must be set before any YOLO() instantiation -- including inside
    # preflight(), which loads the first run's checkpoint for class-name
    # verification -- so this comes before args.output_dir.mkdir()/preflight()
    # rather than after the dry-run check (Codex fix 1).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ctx = preflight(args)

    if args.dry_run:
        _print_dry_run_plan(args, ctx)
        sys.exit(0)

    test_sets = []
    if args.test_set in ("standard", "both"):
        test_sets.append("standard")
    if args.test_set in ("cleaned", "both"):
        test_sets.append("cleaned")

    env_provenance = _compute_env_provenance(args)  # computed once, not per pair (Codex fix 3)

    model_cache: Dict[str, "YOLO"] = {ctx["first_run"]: ctx["first_model"]}
    failures: List[dict] = []
    summary_rows: Dict[str, List[dict]] = {"standard": [], "cleaned": []}

    start = time.time()
    if args.mode in ("eval", "both"):
        for run_name in ctx["run_names"]:
            weights_path = ctx["weights_paths"][run_name]

            # Per-run setup (model load, checkpoint hashing, train-provenance
            # lookup) is guarded on its own: a bad checkpoint or unreadable
            # args.yaml/manifest.txt must not crash the whole sweep. Both
            # test-set pairs for this run are recorded as failed and the
            # loop moves to the next run (Codex fix 2).
            try:
                if run_name not in model_cache:
                    model_cache[run_name] = YOLO(str(weights_path))
                model = model_cache[run_name]
                weights_sha256 = _sha256(weights_path)
                optimizer, iou_type = _train_provenance(run_name, args.runs_dir)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[eval_ablation] ERROR: setup for run '{run_name}' failed: {e}", file=sys.stderr)
                print(tb, file=sys.stderr)
                for test_set in test_sets:
                    failures.append({
                        "pair": f"{run_name}__{test_set}", "run_name": run_name, "test_set": test_set,
                        "error": f"run setup failed: {e}", "traceback": tb,
                    })
                continue

            for test_set in test_sets:
                data_yaml_path = args.data_standard if test_set == "standard" else args.data_cleaned
                image_paths = ctx["std_images"] if test_set == "standard" else ctx["cln_images"]
                row, failure = run_eval_pair(
                    model, run_name, test_set, data_yaml_path, image_paths, ctx["names"],
                    optimizer, iou_type, weights_path, weights_sha256, env_provenance, args.output_dir, args,
                )
                if failure is not None:
                    failures.append(failure)
                else:
                    summary_rows[test_set].append(row)
    elapsed = time.time() - start

    if args.mode in ("eval", "both"):
        for test_set in test_sets:
            if summary_rows[test_set]:
                write_summary_csv(summary_rows[test_set], args.output_dir / f"summary_{test_set}.csv")
        write_eval_manifest(args.output_dir / "eval_manifest.txt", args, ctx, test_sets, failures, elapsed, env_provenance)

    if args.mode in ("demo", "both"):
        run_demo(ctx, args, args.output_dir, model_cache)

    if failures:
        pair_names = [f["pair"] for f in failures]
        print(f"[eval_ablation] completed with {len(failures)} failed pair(s): {pair_names}", file=sys.stderr)
        sys.exit(1)

    print("[eval_ablation] done.")


if __name__ == "__main__":
    main()
