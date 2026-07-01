"""
convert_dataset.py — DeepPCB → YOLO format conversion.

Converts the DeepPCB benchmark dataset (Tang et al., arXiv:1902.06197) into
a YOLO-compatible directory tree with normalized bounding boxes and a dataset
YAML file suitable for Ultralytics training.

Locked decisions implemented here:
  D-014: Only the tested image (_test.jpg) is used as YOLO input.
         Template images (_temp.jpg) are never loaded, referenced, or copied.
  D-015: Official trainval.txt / test.txt splits are honored exactly.
         An 80/20 validation subset is carved from trainval using a local
         random.Random(42) instance so the seed is auditable and does not
         touch Python's global RNG state. Global seed for this script: 42.
"""

import argparse
import logging
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CLASS_NAMES: Dict[int, str] = {
    0: "open",
    1: "short",
    2: "mousebite",
    3: "spur",
    4: "copper",
    5: "pin-hole",
}

SPLITS = ("train", "val", "test")
_CLAMP_EPSILON = 1e-3


# ---------------------------------------------------------------------------
# Path / index helpers
# ---------------------------------------------------------------------------


def _bare_id(stem: str) -> str:
    """Return the bare sample id by stripping a trailing '_test' or '_temp'."""
    if stem.endswith("_test") or stem.endswith("_temp"):
        return stem[: stem.rfind("_")]
    return stem


def _resolve_pcbdata_root(src: Path) -> Path:
    """Find the directory containing trainval.txt and test.txt.

    Supports both layouts:
      - <src>/PCBData/{trainval.txt, test.txt, group*/}
      - <src>/PCBData/PCBData/{trainval.txt, test.txt, group*/}
    Searches at most 2 levels deep under src.
    """
    candidates = [src, src / "PCBData", src / "PCBData" / "PCBData"]
    for candidate in candidates:
        if (candidate / "trainval.txt").is_file() and (candidate / "test.txt").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate DeepPCB root containing trainval.txt and test.txt "
        f"under {src}. Checked: {[str(c) for c in candidates]}"
    )


def build_file_index(
    pcbdata_root: Path,
) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """
    Walk pcbdata_root recursively and build two flat look-up tables keyed by
    bare sample id (e.g. '00041000').

      image_index : {bare_id -> absolute path to *_test.jpg}   (D-014: _temp ignored)
      label_index : {bare_id -> absolute path to *.txt}

    Both group<N> and group<N>_not directories are captured automatically
    because we walk the whole tree (D-010 fix 3).
    """
    image_index: Dict[str, Path] = {}
    label_index: Dict[str, Path] = {}

    for path in pcbdata_root.rglob("*"):
        if not path.is_file():
            continue
        stem = path.stem
        suffix = path.suffix.lower()

        if suffix in (".jpg", ".jpeg", ".png"):
            if stem.endswith("_test"):
                base = _bare_id(stem)
                if base in image_index:
                    log.warning(
                        "Duplicate _test image for id '%s'; keeping %s, ignoring %s",
                        base, image_index[base], path,
                    )
                else:
                    image_index[base] = path
            # _temp files are silently skipped per D-014

        elif suffix == ".txt":
            base = stem  # annotation stems carry no _test/_temp variant
            if base in label_index:
                log.warning(
                    "Duplicate annotation for id '%s'; keeping %s, ignoring %s",
                    base, label_index[base], path,
                )
            else:
                label_index[base] = path

    log.info(
        "File index built: %d test images, %d annotation files",
        len(image_index),
        len(label_index),
    )
    return image_index, label_index


# ---------------------------------------------------------------------------
# Split-file parsing
# ---------------------------------------------------------------------------


def parse_split_file(
    split_file: Path,
    image_index: Dict[str, Path],
    label_index: Dict[str, Path],
) -> List[Tuple[str, Path, Path]]:
    """
    Read an official split file (trainval.txt or test.txt) and resolve each
    entry to (bare_id, image_path, label_path) via the flat indexes.

    Handles:
      - Forward / back slashes and leading './' (D-010 fix 2).
      - Lines with 1 or 2 whitespace-separated fields; only field[0] is used
        for stem derivation (D-010 fix 3).
      - Both '*_test.jpg' references and bare id stems.
    """
    pairs: List[Tuple[str, Path, Path]] = []
    missing_img = 0
    missing_ann = 0

    with split_file.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            # Normalize slashes; pathlib handles the rest (leading ./, etc.)
            fields = line.split()
            ref_path = Path(fields[0].replace("\\", "/"))
            base = _bare_id(ref_path.stem)

            img_path = image_index.get(base)
            ann_path = label_index.get(base)

            if img_path is None:
                log.warning(
                    "No _test image found for id '%s' (line: %r)", base, line
                )
                missing_img += 1
                continue
            if ann_path is None:
                log.warning(
                    "No annotation file found for id '%s' (line: %r)", base, line
                )
                missing_ann += 1
                continue

            pairs.append((base, img_path, ann_path))

    if missing_img or missing_ann:
        log.warning(
            "Skipped entries: %d missing image, %d missing annotation",
            missing_img, missing_ann,
        )
    return pairs


# ---------------------------------------------------------------------------
# Annotation conversion
# ---------------------------------------------------------------------------


def convert_annotation(
    ann_path: Path,
    img_w: int,
    img_h: int,
) -> List[str]:
    """
    Convert a DeepPCB annotation file to YOLO-format lines.

    DeepPCB per-line: x1 y1 x2 y2 class_id  (absolute px; class_id 1-6)
    YOLO per-line:    class_id cx cy w h      (normalized [0,1]; class_id 0-5)

    Lines that do not yield exactly 5 tokens after strip().split() are
    skipped with a warning and never raise (D-010 fix 1).
    """
    yolo_lines: List[str] = []

    with ann_path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            tokens = raw_line.strip().split()  # D-010 fix 1
            if len(tokens) != 5:
                if tokens:
                    log.warning(
                        "Malformed line in %s (expected 5 tokens, got %d): %r",
                        ann_path, len(tokens), raw_line.rstrip(),
                    )
                continue

            try:
                x1, y1, x2, y2 = (float(t) for t in tokens[:4])
                cls_1indexed = int(tokens[4])
            except ValueError:
                log.warning(
                    "Unparseable annotation in %s: %r", ann_path, raw_line.rstrip()
                )
                continue

            if not (1 <= cls_1indexed <= 6):
                log.warning(
                    "Out-of-range class id %d in %s; skipping line",
                    cls_1indexed, ann_path,
                )
                continue

            cls_0indexed = cls_1indexed - 1

            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            # Clamp against annotation rounding at image borders
            original = (cx, cy, w, h)
            cx_c = max(0.0, min(1.0, cx))
            cy_c = max(0.0, min(1.0, cy))
            w_c  = max(0.0, min(1.0, w))
            h_c  = max(0.0, min(1.0, h))
            clamped = (cx_c, cy_c, w_c, h_c)
            max_delta = max(abs(a - b) for a, b in zip(original, clamped))
            if max_delta > _CLAMP_EPSILON:
                log.warning(
                    "bbox clamp exceeded epsilon for %s: original=%s clamped=%s delta=%.4f",
                    ann_path.stem, original, clamped, max_delta,
                )
            cx, cy, w, h = clamped

            yolo_lines.append(
                f"{cls_0indexed} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            )

    return yolo_lines


# ---------------------------------------------------------------------------
# Split processing
# ---------------------------------------------------------------------------


def process_split(
    pairs: List[Tuple[str, Path, Path]],
    split_name: str,
    dst: Path,
) -> Dict[int, int]:
    """
    Copy tested images and write YOLO labels for one split.

    Returns the per-class instance count for the validation report.
    Images are copied (not symlinked) for Colab+Drive mount compatibility.
    """
    images_dir = dst / "images" / split_name
    labels_dir = dst / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    class_dist: Dict[int, int] = defaultdict(int)

    for bare_id, img_path, ann_path in pairs:
        with Image.open(img_path) as im:
            img_w, img_h = im.size  # read actual dimensions; do not hardcode 640

        shutil.copy2(img_path, images_dir / f"{bare_id}.jpg")

        yolo_lines = convert_annotation(ann_path, img_w, img_h)

        lbl_content = ("\n".join(yolo_lines) + "\n") if yolo_lines else ""
        (labels_dir / f"{bare_id}.txt").write_text(lbl_content, encoding="utf-8")

        for line in yolo_lines:
            class_dist[int(line.split()[0])] += 1

    log.info("Processed %d images for split '%s'", len(pairs), split_name)
    return dict(class_dist)


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------


def write_yaml(dst: Path) -> None:
    """Write deeppcb.yaml in Ultralytics dataset format."""
    names_block = "\n".join(f"  {k}: {v}" for k, v in CLASS_NAMES.items())
    content = (
        f"path: {dst.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 6\n"
        f"names:\n"
        f"{names_block}\n"
    )
    yaml_path = dst / "deeppcb.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    log.info("Wrote %s", yaml_path)


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


def _count_split_lines(path: Path) -> int:
    """Count non-empty, non-whitespace lines in a DeepPCB split file."""
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def validate_and_report(
    train_pairs: List[Tuple[str, Path, Path]],
    val_pairs: List[Tuple[str, Path, Path]],
    test_pairs: List[Tuple[str, Path, Path]],
    trainval_raw_count: int,
    test_raw_count: int,
    dst: Path,
    train_dist: Dict[int, int],
    val_dist: Dict[int, int],
    test_dist: Dict[int, int],
) -> None:
    """
    Check mandatory invariants and print the validation table.
    Exits non-zero if any invariant is violated.
    """
    train_n = len(train_pairs)
    val_n = len(val_pairs)
    test_n = len(test_pairs)

    errors: List[str] = []

    # 1a. Parsing must not silently drop entries vs raw file line counts
    parsed_tv = train_n + val_n
    if parsed_tv != trainval_raw_count:
        errors.append(
            f"trainval parsing dropped entries: "
            f"raw_lines={trainval_raw_count} parsed_pairs={parsed_tv}"
        )
    if test_n != test_raw_count:
        errors.append(
            f"test parsing dropped entries: "
            f"raw_lines={test_raw_count} parsed_pairs={test_n}"
        )

    # 1b. Split sum must match raw line counts
    if train_n + val_n != trainval_raw_count:
        errors.append(
            f"split sum mismatch: train={train_n} val={val_n} "
            f"trainval_raw={trainval_raw_count}"
        )
    if test_n != test_raw_count:
        errors.append(
            f"test count mismatch: test={test_n} test_raw={test_raw_count}"
        )

    # 2. Every image stem must have a corresponding label file
    for split_name in SPLITS:
        img_stems = {p.stem for p in (dst / "images" / split_name).glob("*.jpg")}
        lbl_stems = {p.stem for p in (dst / "labels" / split_name).glob("*.txt")}
        orphans = img_stems - lbl_stems
        if orphans:
            errors.append(
                f"Split '{split_name}': {len(orphans)} images missing label "
                f"(e.g. {sorted(orphans)[:3]})"
            )

    # 3. No stem may appear in more than one split
    train_stems = {p[0] for p in train_pairs}
    val_stems = {p[0] for p in val_pairs}
    test_stems = {p[0] for p in test_pairs}
    for a_name, a_set, b_name, b_set in (
        ("train", train_stems, "val", val_stems),
        ("train", train_stems, "test", test_stems),
        ("val", val_stems, "test", test_stems),
    ):
        overlap = a_set & b_set
        if overlap:
            errors.append(
                f"Stem overlap between {a_name} and {b_name}: "
                f"{len(overlap)} stems (e.g. {sorted(overlap)[:3]})"
            )

    if errors:
        for msg in errors:
            log.error("INVARIANT FAIL: %s", msg)
        sys.exit(1)

    # ── Validation table ──────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}")
    print("  DATASET CONVERSION VALIDATION REPORT")
    print(sep)
    print(f"\n  Image counts per split:")
    print(f"    {'train':<8}: {train_n:>6}  (expected ~800)")
    print(f"    {'val':<8}: {val_n:>6}  (expected ~200)")
    print(f"    {'test':<8}: {test_n:>6}  (expected ~500)")
    print(f"    {'total':<8}: {train_n + val_n + test_n:>6}")
    print(f"\n  Raw split-file line counts vs parsed pairs:")
    print(f"    trainval.txt : raw={trainval_raw_count}  parsed={train_n + val_n}")
    print(f"    test.txt     : raw={test_raw_count}  parsed={test_n}")

    col_w = 9
    print(f"\n  Class distribution (defect instances per split):")
    hdr = (
        f"    {'Class':<12}"
        f"{'Train':>{col_w}}"
        f"{'Val':>{col_w}}"
        f"{'Test':>{col_w}}"
        f"{'Total':>{col_w}}"
    )
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for cls_id, cls_name in CLASS_NAMES.items():
        tr = train_dist.get(cls_id, 0)
        vl = val_dist.get(cls_id, 0)
        te = test_dist.get(cls_id, 0)
        print(
            f"    {cls_name:<12}"
            f"{tr:>{col_w}}"
            f"{vl:>{col_w}}"
            f"{te:>{col_w}}"
            f"{tr + vl + te:>{col_w}}"
        )

    print(f"\n  Sanity invariants:")
    print(
        f"    [OK] trainval parsed == raw lines  "
        f"({train_n + val_n} == {trainval_raw_count})"
    )
    print(
        f"    [OK] test parsed == raw lines      "
        f"({test_n} == {test_raw_count})"
    )
    print(f"    [OK] Every image has a corresponding label file")
    print(f"    [OK] No stem appears in more than one split")
    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DeepPCB dataset to YOLO format (D-014, D-015)."
    )
    parser.add_argument(
        "--src", required=True, type=Path,
        help="DeepPCB root directory (contains trainval.txt, test.txt, PCBData/)",
    )
    parser.add_argument(
        "--dst", required=True, type=Path,
        help="Output directory for the YOLO dataset",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete and rebuild --dst if it already exists",
    )
    args = parser.parse_args()

    src: Path = args.src.resolve()
    dst: Path = args.dst.resolve()

    # ── Locate DeepPCB root (auto-detects layout variant) ─────────────────
    try:
        pcbdata_root = _resolve_pcbdata_root(src)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)
    trainval_file = pcbdata_root / "trainval.txt"
    test_file = pcbdata_root / "test.txt"
    log.info("DeepPCB root resolved to: %s", pcbdata_root)

    # ── Idempotency guard ──────────────────────────────────────────────────
    if dst.exists() and any(dst.iterdir()):
        if not args.force:
            log.error(
                "--dst '%s' already exists and is non-empty. "
                "Pass --force to delete and rebuild.",
                dst,
            )
            sys.exit(1)
        log.info("--force: removing existing --dst '%s'", dst)
        shutil.rmtree(dst)

    dst.mkdir(parents=True, exist_ok=True)

    # ── Build flat file indexes (D-010 fix 3) ──────────────────────────────
    image_index, label_index = build_file_index(pcbdata_root)

    # ── Parse official split files ─────────────────────────────────────────
    trainval_pairs = parse_split_file(trainval_file, image_index, label_index)
    test_pairs = parse_split_file(test_file, image_index, label_index)

    trainval_raw_count = _count_split_lines(trainval_file)
    test_raw_count = _count_split_lines(test_file)
    log.info(
        "Resolved %d/%d trainval entries and %d/%d test entries (parsed/raw)",
        len(trainval_pairs), trainval_raw_count,
        len(test_pairs), test_raw_count,
    )

    # Plate-aware 80/20 split. Random per-image split (pre-Day 5)
    # leaked 99.5% of val plates into train — see audit at Day 5
    # item 2a. Grouping by 7-char plate ID enforces disjoint
    # source plates between train and val.
    # ── D-015 (revised): plate-aware 80/20 val from trainval, seed=42 ───────
    plate_to_pairs: Dict[str, List[Tuple[str, Path, Path]]] = defaultdict(list)
    for pair in trainval_pairs:
        plate_to_pairs[pair[0][:7]].append(pair)

    plate_ids = sorted(plate_to_pairs)          # deterministic input order
    rng = random.Random(42)
    rng.shuffle(plate_ids)

    if len(plate_ids) < 5:
        raise ValueError(
            f"Plate-aware split requires at least 5 plates, "
            f"got {len(plate_ids)}. Cannot form 80/20 split."
        )

    val_plate_count = len(plate_ids) // 5       # 20 % of plates

    # Image counts differ slightly from old 800/200; plates have variable
    # image counts (typically 8-10). That is expected and correct.
    train_pairs: List[Tuple[str, Path, Path]] = [
        pair for pid in plate_ids[val_plate_count:]
             for pair in plate_to_pairs[pid]
    ]
    val_pairs: List[Tuple[str, Path, Path]] = [
        pair for pid in plate_ids[:val_plate_count]
             for pair in plate_to_pairs[pid]
    ]

    # Permanent safety check — fails loudly if the split ever regresses.
    _train_plate_set = {p[0][:7] for p in train_pairs}
    _val_plate_set   = {p[0][:7] for p in val_pairs}
    _leaked = _train_plate_set & _val_plate_set
    if _leaked:
        raise AssertionError(
            f"Plate-disjoint invariant violated: {len(_leaked)} shared plate IDs "
            f"(first 5: {sorted(_leaked)[:5]})"
        )

    n_train_plates = len(plate_ids) - val_plate_count
    n_val_plates   = val_plate_count
    print(
        f"Split by plate (seed=42): {n_train_plates} plates -> {len(train_pairs)} imgs,\n"
        f"                          {n_val_plates} plates -> {len(val_pairs)} imgs"
    )
    log.info(
        "D-015 split (plate-aware): %d plates / %d imgs train, %d plates / %d imgs val",
        n_train_plates, len(train_pairs), n_val_plates, len(val_pairs),
    )

    # ── Process splits ─────────────────────────────────────────────────────
    train_dist = process_split(train_pairs, "train", dst)
    val_dist = process_split(val_pairs, "val", dst)
    test_dist = process_split(test_pairs, "test", dst)

    # ── Write dataset YAML ─────────────────────────────────────────────────
    write_yaml(dst)

    # ── Validate and print report ──────────────────────────────────────────
    validate_and_report(
        train_pairs, val_pairs, test_pairs,
        trainval_raw_count, test_raw_count,
        dst,
        train_dist, val_dist, test_dist,
    )


if __name__ == "__main__":
    main()
