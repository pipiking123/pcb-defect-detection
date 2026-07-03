"""build_cleaned_test_yaml.py -- generate the D-024 cleaned 478-image test set.

Reads the standard DeepPCB test split (500 images) plus the machine-readable
exclusion rule in configs/cleaned_test_exclusion.yaml, filters out the 22
images that belong to the 4 upstream-leaked plates, and emits:
  - a cleaned image-list .txt (Ultralytics list-file format, absolute paths)
  - a cleaned dataset YAML mirroring the standard YAML but with test:
    pointing at the cleaned list
  - a traceability manifest recording exactly which images were excluded
    and why (D-024)

This script is read-only with respect to --dataset-root and the directory
tree pointed to by --standard-yaml, aside from writing the cleaned list file
next to the standard split (see _cleaned_list_path). It never touches
trainval data.

Environment-agnostic: every path is a CLI arg. Designed to run standalone
on Kaggle (see eval_ablation.py for the consumer of its outputs).
"""

import argparse
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import yaml

STANDARD_TEST_COUNT = 500  # DeepPCB official test.txt line count (D-015/D-024)
IMG_SUFFIXES = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------------------
# Small helpers (mirrors src/data/convert_dataset.py conventions)
# ---------------------------------------------------------------------------


def _bare_id(stem: str) -> str:
    """Strip a trailing '_test' or '_temp' suffix from a sample stem."""
    if stem.endswith("_test") or stem.endswith("_temp"):
        return stem[: stem.rfind("_")]
    return stem


def _resolve_pcbdata_root(src: Path) -> Path:
    """Locate the directory containing trainval.txt / test.txt.

    Supports both <src>/PCBData/ and <src>/PCBData/PCBData/ layouts, same
    as convert_dataset.py._resolve_pcbdata_root.
    """
    candidates = [src, src / "PCBData", src / "PCBData" / "PCBData"]
    for candidate in candidates:
        if (candidate / "trainval.txt").is_file() and (candidate / "test.txt").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate DeepPCB root containing trainval.txt and test.txt "
        f"under {src}. Checked: {[str(c) for c in candidates]}"
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Data-yaml split resolution (mirrors ultralytics.data.utils.check_det_dataset
# path-joining semantics closely enough for our read-only purposes, without
# importing ultralytics -- this script has no hard ultralytics dependency)
# ---------------------------------------------------------------------------


def _resolve_yaml_field(yaml_dir: Path, dataset_path: Path, field_value: str) -> Path:
    p = Path(field_value)
    if p.is_absolute():
        return p
    return (dataset_path / field_value).resolve()


def _list_split_images(resolved: Path) -> List[Path]:
    """Return the sorted image list a split path resolves to.

    Mirrors ultralytics.data.base.BaseDataset.get_img_files: a directory is
    globbed recursively; a .txt file is read as a newline-delimited list of
    image paths (lines starting with './' are relative to the txt file's
    parent, otherwise used as-is).
    """
    if resolved.is_dir():
        return sorted(
            p for p in resolved.rglob("*") if p.is_file() and p.suffix.lower() in IMG_SUFFIXES
        )
    if resolved.is_file():
        lines = [ln.strip() for ln in resolved.read_text(encoding="utf-8").splitlines() if ln.strip()]
        out = []
        for ln in lines:
            if ln.startswith("./"):
                out.append((resolved.parent / ln[2:]).resolve())
            else:
                out.append(Path(ln))
        return sorted(out)
    raise FileNotFoundError(f"Resolved split path does not exist: {resolved}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the D-024 cleaned 478-image DeepPCB test set (YAML + manifest)."
    )
    parser.add_argument(
        "--dataset-root", required=True, type=Path,
        help="DeepPCB root directory (contains PCBData/test.txt) -- source of the canonical 500-image manifest.",
    )
    parser.add_argument(
        "--standard-yaml", required=True, type=Path,
        help="Existing standard Ultralytics dataset YAML (path/train/val/test/nc/names) to mirror.",
    )
    parser.add_argument(
        "--exclusion-config", required=True, type=Path,
        help="Path to configs/cleaned_test_exclusion.yaml (D-024 exclusion rule, machine-readable).",
    )
    parser.add_argument("--output-yaml", required=True, type=Path, help="Where to write the cleaned dataset YAML.")
    parser.add_argument(
        "--output-manifest", required=True, type=Path, help="Where to write the traceability manifest (.txt)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Recorded in the manifest for consistency with project-wide seed policy. "
             "Filtering here is fully deterministic (sorted ids, no RNG) -- this does not drive any random step.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing outputs even if their content differs from what would be generated now.",
    )
    args = parser.parse_args()

    # ── Load exclusion config (D-024 rule; never hardcoded here) ───────────
    with args.exclusion_config.open(encoding="utf-8") as f:
        excl_cfg = yaml.safe_load(f)

    required_keys = {
        "exclusion_plate_prefixes",
        "expected_excluded_count",
        "expected_remaining_count",
        "source_decision",
    }
    missing = required_keys - excl_cfg.keys()
    if missing:
        print(f"ERROR: --exclusion-config missing required keys: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    prefixes = [str(p) for p in excl_cfg["exclusion_plate_prefixes"]]
    expected_excluded = int(excl_cfg["expected_excluded_count"])
    expected_remaining = int(excl_cfg["expected_remaining_count"])
    source_decision = str(excl_cfg["source_decision"])

    # ── Canonical 500-image manifest from raw DeepPCB test.txt ─────────────
    pcbdata_root = _resolve_pcbdata_root(args.dataset_root)
    test_txt = pcbdata_root / "test.txt"

    txt_ids: List[str] = []
    with test_txt.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            ref_path = Path(line.split()[0].replace("\\", "/"))
            txt_ids.append(_bare_id(ref_path.stem))
    txt_ids_sorted = sorted(txt_ids)

    if len(txt_ids_sorted) != STANDARD_TEST_COUNT:
        print(
            f"ERROR: expected {STANDARD_TEST_COUNT} entries in {test_txt}, got {len(txt_ids_sorted)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Resolve the standard YAML's test: split to actual image files ──────
    with args.standard_yaml.open(encoding="utf-8") as f:
        std_yaml = yaml.safe_load(f)

    for key in ("path", "train", "val", "test", "nc", "names"):
        if key not in std_yaml:
            print(f"ERROR: --standard-yaml {args.standard_yaml} missing required key '{key}'.", file=sys.stderr)
            sys.exit(1)

    yaml_dir = args.standard_yaml.resolve().parent
    dataset_path = Path(std_yaml["path"])
    if not dataset_path.is_absolute():
        dataset_path = (yaml_dir / dataset_path).resolve()

    test_split_resolved = _resolve_yaml_field(yaml_dir, dataset_path, std_yaml["test"])
    test_images = _list_split_images(test_split_resolved)
    id_to_path: Dict[str, Path] = {p.stem: p.resolve() for p in test_images}
    yaml_ids_sorted = sorted(id_to_path)

    if len(yaml_ids_sorted) != STANDARD_TEST_COUNT:
        print(
            f"ERROR: expected {STANDARD_TEST_COUNT} images at resolved test split "
            f"{test_split_resolved}, found {len(yaml_ids_sorted)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    set_txt, set_yaml = set(txt_ids_sorted), set(yaml_ids_sorted)
    if set_txt != set_yaml:
        diff = sorted(set_txt ^ set_yaml)[:10]
        print(
            "ERROR: test.txt ids and standard-yaml test-split ids disagree. "
            f"First diffs (up to 10): {diff}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Filter by plate prefix (D-024) ──────────────────────────────────────
    prefix_set = set(prefixes)
    excluded_ids = sorted(i for i in txt_ids_sorted if i[:7] in prefix_set)
    remaining_ids = sorted(i for i in txt_ids_sorted if i[:7] not in prefix_set)

    if len(excluded_ids) != expected_excluded or len(remaining_ids) != expected_remaining:
        per_prefix = {p: sum(1 for i in excluded_ids if i.startswith(p)) for p in prefixes}
        print(
            "ERROR: exclusion filter produced unexpected counts. "
            f"excluded={len(excluded_ids)} (expected {expected_excluded}), "
            f"remaining={len(remaining_ids)} (expected {expected_remaining}). "
            f"Per-prefix breakdown: {per_prefix}",
            file=sys.stderr,
        )
        sys.exit(1)

    cleaned_paths = sorted(id_to_path[i] for i in remaining_ids)

    # ── Idempotency check ────────────────────────────────────────────────
    cleaned_list_path = dataset_path / "test_cleaned.txt"
    new_list_content = "\n".join(str(p) for p in cleaned_paths) + "\n"
    new_list_bytes = new_list_content.encode("utf-8")
    new_list_sha256 = _sha256_bytes(new_list_bytes)

    outputs_exist = cleaned_list_path.exists() and args.output_yaml.exists()
    partial_exist = (cleaned_list_path.exists() or args.output_yaml.exists()) and not outputs_exist

    if not args.force and outputs_exist:
        existing_sha256 = _sha256(cleaned_list_path)
        if existing_sha256 == new_list_sha256:
            print(
                f"[build_cleaned_test_yaml] Outputs already exist and match "
                f"(SHA-256={new_list_sha256[:12]}...) -- no change. Pass --force to overwrite."
            )
            sys.exit(0)
        print(
            "ERROR: existing outputs differ from freshly computed content. "
            f"existing test_cleaned.txt SHA-256={existing_sha256}, new={new_list_sha256}. "
            "Refusing to overwrite silently -- pass --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.force and partial_exist:
        print(
            f"ERROR: one of the two outputs already exists ({cleaned_list_path} / {args.output_yaml}) "
            "but not both -- ambiguous partial state. Pass --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Write cleaned list ───────────────────────────────────────────────
    cleaned_list_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_list_path.write_text(new_list_content, encoding="utf-8")

    # ── Write cleaned YAML (mirrors convert_dataset.py's write_yaml style) ─
    names_block = "\n".join(f"  {k}: {v}" for k, v in std_yaml["names"].items())
    yaml_content = (
        f"# D-024 cleaned test set -- generated by build_cleaned_test_yaml.py\n"
        f"# Mirrors {args.standard_yaml} with test: replaced by the 478-image\n"
        f"# cleaned list (22 images from 4 upstream-leaked plates removed).\n"
        f"# Source decision: {source_decision}\n"
        f"path: {dataset_path}\n"
        f"train: {std_yaml['train']}\n"
        f"val: {std_yaml['val']}\n"
        f"test: {cleaned_list_path}\n"
        f"nc: {std_yaml['nc']}\n"
        f"names:\n"
        f"{names_block}\n"
    )
    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.output_yaml.write_text(yaml_content, encoding="utf-8")

    # ── Write manifest ───────────────────────────────────────────────────
    cleaned_list_sha256 = _sha256(cleaned_list_path)
    test_txt_sha256 = _sha256(test_txt)

    manifest_lines = [
        f"Timestamp UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Git HEAD: {_git_head()}",
        f"Source decision: {source_decision}",
        f"Seed (recorded, not used -- filtering is deterministic): {args.seed}",
        f"Dataset root: {args.dataset_root.resolve()}",
        f"Standard YAML: {args.standard_yaml.resolve()}",
        f"Output YAML: {args.output_yaml.resolve()}",
        f"Exclusion plate prefixes: {prefixes}",
        f"Standard test count: {STANDARD_TEST_COUNT}",
        f"Excluded count: {len(excluded_ids)}",
        f"Remaining count: {len(remaining_ids)}",
        f"Cleaned list file: {cleaned_list_path}",
        f"Cleaned list SHA-256: {cleaned_list_sha256}",
        f"Source test.txt: {test_txt}",
        f"Source test.txt SHA-256: {test_txt_sha256}",
        "Excluded image IDs (sorted):",
    ]
    manifest_lines += [f"  {i}" for i in excluded_ids]

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    print(
        f"[build_cleaned_test_yaml] wrote {cleaned_list_path} ({len(remaining_ids)} images), "
        f"{args.output_yaml}, {args.output_manifest}"
    )


if __name__ == "__main__":
    main()
