"""
sanity_check.py — Visual verification gate for convert_dataset.py output.

Purpose:
    Reads the filesystem output produced by convert_dataset.py (commit ee58602)
    and writes two annotated PNG views per sampled image:
      - image_boxes/  : original image with ground-truth boxes and class labels
      - label_only/   : white canvas with the same boxes and labels

    Designed to catch converter bugs before any training code is written.
    This script intentionally does NOT import from convert_dataset.py — it
    reads filesystem output only, so it provides an independent second opinion.

Locked decisions respected:
    D-010 amendment : class IDs in converter output are 0-indexed.
    D-014           : only *_test.jpg images exist in the converted dataset.
    D-015           : official trainval/test split honored; val carved at seed=42.
    Sampling seed   : random.Random(42) local instance; never touches global RNG.

Dependencies: stdlib + Pillow + PyYAML (transitive dep of Ultralytics, always present).
"""

import argparse
import datetime
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (220, 20, 60),    # open       — crimson
    1: (30, 144, 255),   # short      — dodger blue
    2: (50, 205, 50),    # mousebite  — lime green
    3: (255, 140, 0),    # spur       — dark orange
    4: (148, 0, 211),    # copper     — dark violet
    5: (255, 215, 0),    # pin-hole   — gold
}

FALLBACK_COLOR: Tuple[int, int, int] = (0, 0, 0)

SPLITS: Tuple[str, ...] = ("train", "val", "test")

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def load_yaml(src: Path) -> Dict:
    """Load and return deeppcb.yaml as a dict. Exits non-zero if missing or malformed."""
    yaml_path = src / "deeppcb.yaml"
    if not yaml_path.is_file():
        log.error("deeppcb.yaml not found at %s", yaml_path)
        sys.exit(1)
    try:
        with yaml_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        log.error("Failed to parse %s: %s", yaml_path, exc)
        sys.exit(1)
    except OSError as exc:
        log.error("Failed to read %s: %s", yaml_path, exc)
        sys.exit(1)
    if data is None:
        log.error("deeppcb.yaml is empty: %s", yaml_path)
        sys.exit(1)
    return data


def parse_class_names(yaml_data: Dict) -> Dict[int, str]:
    """
    Extract the names mapping from YAML.
    Source of truth for class names (D-014 / D-015 / D-010-amendment).
    Returns {int_class_id: class_name_str}.
    """
    raw = yaml_data.get("names", {})
    return {int(k): str(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_dst(dst: Path, force: bool) -> None:
    """
    Guard against accidental overwrites.
    Without --force, exits if dst is non-empty.
    With --force, removes dst entirely and recreates it empty.
    Always ensures dst exists on return.
    """
    if dst.exists() and any(dst.iterdir()):
        if not force:
            log.error(
                "Output directory %s already exists and is not empty. "
                "Pass --force to overwrite.",
                dst,
            )
            sys.exit(1)
        log.warning("--force: removing existing output directory %s", dst)
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)


def validate_src(src: Path) -> None:
    """
    Confirm expected directory structure exists and each split has at least
    one image. Exits non-zero on the first hard failure.
    """
    for split in SPLITS:
        img_dir = src / "images" / split
        lbl_dir = src / "labels" / split
        if not img_dir.is_dir():
            log.error("Expected images directory not found: %s", img_dir)
            sys.exit(1)
        if not lbl_dir.is_dir():
            log.error("Expected labels directory not found: %s", lbl_dir)
            sys.exit(1)
        images = list(img_dir.glob("*.jpg"))
        if not images:
            log.error("Split '%s' has no .jpg images in %s", split, img_dir)
            sys.exit(1)
        log.info("Split '%s': %d images found", split, len(images))


# ---------------------------------------------------------------------------
# Class index and sampling
# ---------------------------------------------------------------------------


def build_class_index(
    src: Path,
    split: str,
    nc: int,
) -> Dict[int, List[str]]:
    """
    Scan <src>/labels/<split>/ and map each valid class_id → list of stems
    that contain at least one annotation of that class.

    Out-of-range class IDs (indicating a converter bug) are logged as ERROR;
    they are excluded from the index but the image is still indexed under its
    valid class IDs.
    """
    lbl_dir = src / "labels" / split
    index: Dict[int, List[str]] = {cid: [] for cid in range(nc)}

    for lbl_file in sorted(lbl_dir.glob("*.txt")):
        stem = lbl_file.stem
        seen_for_stem: Set[int] = set()
        for line in lbl_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cid = int(line.split()[0])
            if cid in seen_for_stem:
                continue
            seen_for_stem.add(cid)
            if cid in index:
                index[cid].append(stem)
            else:
                log.error(
                    "Out-of-range class id %d in label file %s (converter bug?)",
                    cid,
                    stem,
                )

    return index


def select_images(
    src: Path,
    nc: int,
    n_per_split: int,
    rng: random.Random,
) -> Dict[str, List[str]]:
    """
    Return a class-stratified sample of up to n_per_split stems per split.

    Single rng instance is consumed in a fixed order for reproducibility.
    Algorithm:
      Phase 1 — coverage pass: for each class id in 0..nc-1, pick one image
                from the split with the most images of that class.
      Phase 2 — fill-up: shuffle remaining stems per split and fill to
                n_per_split.
    """
    class_index: Dict[str, Dict[int, List[str]]] = {
        split: build_class_index(src, split, nc) for split in SPLITS
    }

    all_stems: Dict[str, List[str]] = {
        split: sorted(p.stem for p in (src / "labels" / split).glob("*.txt"))
        for split in SPLITS
    }

    selected: Dict[str, List[str]] = {split: [] for split in SPLITS}
    selected_sets: Dict[str, Set[str]] = {split: set() for split in SPLITS}

    # Phase 1 — class coverage pass
    for cid in range(nc):
        # Default-arg capture of cid: makes closure binding explicit and refactor-safe.
        def _available_count(split: str, _cid: int = cid) -> int:
            return sum(
                1
                for s in class_index[split].get(_cid, [])
                if s not in selected_sets[split] and len(selected[split]) < n_per_split
            )

        splits_by_count = sorted(SPLITS, key=_available_count, reverse=True)

        placed = False
        for split in splits_by_count:
            candidates = [
                s
                for s in class_index[split].get(cid, [])
                if s not in selected_sets[split]
            ]
            if not candidates or len(selected[split]) >= n_per_split:
                continue
            assert candidates, f"no candidates for class {cid} in split {split}"
            stem = rng.choice(candidates)
            selected[split].append(stem)
            selected_sets[split].add(stem)
            placed = True
            break

        if not placed:
            log.warning(
                "Class %d: no available slot in any split for coverage "
                "(class absent from dataset or all splits already full).",
                cid,
            )

    # Phase 2 — fill each split to n_per_split
    for split in SPLITS:
        remaining = [s for s in all_stems[split] if s not in selected_sets[split]]
        rng.shuffle(remaining)
        needed = n_per_split - len(selected[split])
        if needed > 0:
            fill = remaining[:needed]
            selected[split].extend(fill)
            selected_sets[split].update(fill)

        if len(selected[split]) < n_per_split:
            log.warning(
                "Split '%s' provided only %d images (--n-per-split=%d requested).",
                split,
                len(selected[split]),
                n_per_split,
            )

    return selected


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------


def parse_label_file(
    lbl_path: Path,
    img_w: int,
    img_h: int,
    class_names: Dict[int, str],
) -> List[Tuple[int, int, int, int, int]]:
    """
    Parse a YOLO label file and return (class_id, x1, y1, x2, y2) tuples
    in absolute pixel coordinates, clamped to image bounds.

    YOLO format: class_id cx cy w h  (normalized, class IDs are 0-indexed per D-010).
    Out-of-range class IDs are logged as ERROR; the box is still returned with
    a fallback color so rendering always completes.
    """
    boxes: List[Tuple[int, int, int, int, int]] = []
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cid = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        x1 = int(round((cx - bw / 2) * img_w))
        y1 = int(round((cy - bh / 2) * img_h))
        x2 = int(round((cx + bw / 2) * img_w))
        y2 = int(round((cy + bh / 2) * img_h))

        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        if cid not in class_names:
            log.error(
                "Out-of-range class id %d in %s (converter bug?)",
                cid,
                lbl_path.stem,
            )

        boxes.append((cid, x1, y1, x2, y2))
    return boxes


# ---------------------------------------------------------------------------
# Font helper
# ---------------------------------------------------------------------------


def _load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _draw_boxes(
    draw: ImageDraw.ImageDraw,
    boxes: List[Tuple[int, int, int, int, int]],
    class_names: Dict[int, str],
    font: ImageFont.ImageFont,
) -> None:
    """Draw all boxes and labels onto an existing ImageDraw context."""
    for cid, x1, y1, x2, y2 in boxes:
        color = CLASS_COLORS.get(cid, FALLBACK_COLOR)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = class_names.get(cid, str(cid))
        draw.text((x1, max(0, y1 - 16)), label, fill=color, font=font)


def render_image_boxes(
    img: Image.Image,
    boxes: List[Tuple[int, int, int, int, int]],
    class_names: Dict[int, str],
    font: ImageFont.ImageFont,
) -> Image.Image:
    """Return a copy of img with ground-truth boxes and class labels overlaid."""
    out = img.copy().convert("RGB")
    _draw_boxes(ImageDraw.Draw(out), boxes, class_names, font)
    return out


def render_label_only(
    img_w: int,
    img_h: int,
    boxes: List[Tuple[int, int, int, int, int]],
    class_names: Dict[int, str],
    font: ImageFont.ImageFont,
) -> Image.Image:
    """Return a white canvas of the same size as the source image with boxes drawn."""
    out = Image.new("RGB", (img_w, img_h), "white")
    _draw_boxes(ImageDraw.Draw(out), boxes, class_names, font)
    return out


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def write_summary(
    dst: Path,
    rows: List[Tuple[str, str, List[Tuple[int, int, int, int, int]]]],
    class_names: Dict[int, str],
    nc: int,
) -> None:
    """
    Write <dst>/summary.txt with a fixed-width table of sampled images,
    followed by a class-coverage count block.

    rows: list of (split, stem, boxes) in the order they were rendered.
    """
    today = datetime.date.today().isoformat()
    class_counts: Dict[int, int] = {cid: 0 for cid in range(nc)}

    lines: List[str] = []
    lines.append(f"SANITY CHECK SAMPLE — generated {today} (seed=42)")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"{'split':<6}  {'stem':<18}  {'n_defects':>9}  classes_present")

    for split, stem, boxes in rows:
        n_defects = len(boxes)
        class_labels = sorted(set(class_names.get(cid, str(cid)) for cid, *_ in boxes))
        classes_str = ", ".join(class_labels) if class_labels else "(none)"
        for cid, *_ in boxes:
            if cid in class_counts:
                class_counts[cid] += 1
        lines.append(f"{split:<6}  {stem:<18}  {n_defects:>9}  {classes_str}")

    lines.append("")
    lines.append("=" * 60)
    lines.append(f"Class coverage across {len(rows)} samples:")
    for cid in range(nc):
        name = class_names.get(cid, str(cid))
        lines.append(f"  {cid} {name:<12}: {class_counts[cid]}")

    for cid in range(nc):
        if class_counts[cid] == 0:
            name = class_names.get(cid, str(cid))
            lines.append(f"WARNING: class {cid} {name} not covered in sample")

    (dst / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(src: Path, dst: Path, force: bool, n_per_split: int) -> None:
    validate_dst(dst, force)

    yaml_data = load_yaml(src)
    class_names = parse_class_names(yaml_data)
    nc = int(yaml_data.get("nc", len(class_names)))

    validate_src(src)

    for split in SPLITS:
        (dst / "image_boxes" / split).mkdir(parents=True, exist_ok=True)
        (dst / "label_only" / split).mkdir(parents=True, exist_ok=True)

    # Single RNG instance for all sampling (D-015 / reproducibility)
    rng = random.Random(42)
    selected = select_images(src, nc, n_per_split, rng)

    # ------------------------------------------------------------------
    # Hard post-sampling coverage gate — runs before any PNG is written.
    # Distinguishes two failure modes with distinct exit codes:
    #   exit(2): sampling bug — a class present in the dataset was missed.
    #   exit(3): converter bug — a class declared in nc is absent from all labels.
    # ------------------------------------------------------------------
    dataset_class_ids: Set[int] = set()
    for _split in SPLITS:
        split_index = build_class_index(src, _split, nc)
        for _cid, _stems in split_index.items():
            if _stems:
                dataset_class_ids.add(_cid)

    covered: Set[int] = set()
    for _split in SPLITS:
        for _stem in selected[_split]:
            _lbl = src / "labels" / _split / f"{_stem}.txt"
            if _lbl.exists():
                for _line in _lbl.read_text(encoding="utf-8").splitlines():
                    _line = _line.strip()
                    if _line:
                        covered.add(int(_line.split()[0]))

    missing_existing = dataset_class_ids - covered
    missing_absent = set(range(nc)) - dataset_class_ids

    if missing_existing:
        log.error(
            "Sampling failed coverage: class IDs %s exist in the dataset "
            "but were not sampled. This is a sampling bug.",
            sorted(missing_existing),
        )
        sys.exit(2)

    if missing_absent:
        log.error(
            "Class IDs %s are declared in deeppcb.yaml (nc=%d) but appear "
            "in ZERO label files across the entire dataset. This is likely "
            "a converter bug — investigate before trusting the sanity gate.",
            sorted(missing_absent),
            nc,
        )
        sys.exit(3)

    font = _load_font()

    rows: List[Tuple[str, str, List[Tuple[int, int, int, int, int]]]] = []
    n_image_boxes = 0
    n_label_only = 0

    for split in SPLITS:
        for stem in selected[split]:
            img_path = src / "images" / split / f"{stem}.jpg"
            lbl_path = src / "labels" / split / f"{stem}.txt"

            if not img_path.exists():
                log.error("Image file not found: %s", img_path)
                continue
            if not lbl_path.exists():
                log.error("Label file not found: %s", lbl_path)
                continue

            with Image.open(img_path) as img:
                img_w, img_h = img.size  # read actual dimensions; do not assume 640×640
                img_rgb = img.convert("RGB").copy()

            boxes = parse_label_file(lbl_path, img_w, img_h, class_names)
            rows.append((split, stem, boxes))

            annotated = render_image_boxes(img_rgb, boxes, class_names, font)
            annotated.save(dst / "image_boxes" / split / f"{stem}.png")
            n_image_boxes += 1

            label_img = render_label_only(img_w, img_h, boxes, class_names, font)
            label_img.save(dst / "label_only" / split / f"{stem}.png")
            n_label_only += 1

            log.info("Rendered %s/%s (%d boxes)", split, stem, len(boxes))

    write_summary(dst, rows, class_names, nc)

    n_selected = sum(len(stems) for stems in selected.values())
    expected_pngs = n_selected * 2
    written_pngs = n_image_boxes + n_label_only
    if written_pngs != expected_pngs:
        log.error(
            "PNG count mismatch: expected %d (2 × %d selected), wrote %d",
            expected_pngs,
            n_selected,
            written_pngs,
        )
        sys.exit(1)

    print(
        f"Wrote {n_image_boxes} image_boxes and {n_label_only} label_only PNGs to {dst}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Sanity-check visual verifier for convert_dataset.py output. "
            "Samples a class-stratified set of images, renders GT boxes on "
            "them, and writes annotated PNGs for human review."
        )
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to converted dataset root (contains deeppcb.yaml, images/, labels/).",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("runs/sanity_day3"),
        help="Output directory for annotated PNGs and summary.txt (default: runs/sanity_day3).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory without prompting.",
    )
    parser.add_argument(
        "--n-per-split",
        type=int,
        default=5,
        metavar="N",
        help="Number of images to sample per split (default: 5).",
    )
    args = parser.parse_args()
    run(
        src=args.src,
        dst=args.dst,
        force=args.force,
        n_per_split=args.n_per_split,
    )


if __name__ == "__main__":
    main()
