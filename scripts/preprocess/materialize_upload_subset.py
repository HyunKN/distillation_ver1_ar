#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def _read_images_from_jsonl(path: Path) -> Set[str]:
    images: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            image = obj.get("image")
            if isinstance(image, str) and image:
                images.add(image.replace("\\", "/"))
    return images


def _format_gb(n_bytes: int) -> str:
    return f"{n_bytes / (1024**3):.2f} GB"


def _collect_sizes(data_root: Path, images: Iterable[str]) -> Tuple[int, int, Dict[str, int], Dict[str, int]]:
    total = 0
    missing = 0
    by_source_bytes: Dict[str, int] = defaultdict(int)
    by_source_count: Dict[str, int] = defaultdict(int)

    for rel in images:
        src = data_root / rel
        source = rel.split("/", 1)[0] if "/" in rel else "<unknown>"
        if not src.exists():
            missing += 1
            continue
        sz = src.stat().st_size
        total += sz
        by_source_bytes[source] += sz
        by_source_count[source] += 1
    return total, missing, dict(by_source_bytes), dict(by_source_count)


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        # Fall back to copy if hardlink is not supported.
        try:
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Materialize an upload subset folder using train/val JSONL image references."
    )
    ap.add_argument("--data_root", required=True, help="Root containing original images.")
    ap.add_argument(
        "--jsonl_dir",
        required=True,
        help="Directory containing train.jsonl and val.jsonl.",
    )
    ap.add_argument("--train_name", default="train.jsonl")
    ap.add_argument("--val_name", default="val.jsonl")
    ap.add_argument(
        "--out_root",
        required=True,
        help="Output root for upload-ready subset.",
    )
    ap.add_argument(
        "--mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="hardlink works only on same filesystem; otherwise falls back to copy.",
    )
    ap.add_argument(
        "--budget_gb",
        type=float,
        default=0.0,
        help="If >0, print warning when estimated image bytes exceed budget.",
    )
    ap.add_argument("--dry_run", action="store_true", help="Estimate only, do not copy files.")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    jsonl_dir = Path(args.jsonl_dir)
    out_root = Path(args.out_root)

    train_jsonl = jsonl_dir / args.train_name
    val_jsonl = jsonl_dir / args.val_name
    if not train_jsonl.exists() or not val_jsonl.exists():
        raise FileNotFoundError(f"JSONL files not found: {train_jsonl}, {val_jsonl}")

    train_images = _read_images_from_jsonl(train_jsonl)
    val_images = _read_images_from_jsonl(val_jsonl)
    all_images = train_images | val_images

    total_bytes, missing, by_source_bytes, by_source_count = _collect_sizes(data_root, all_images)

    print("=== SUBSET SUMMARY ===")
    print(f"data_root:       {data_root}")
    print(f"jsonl_dir:       {jsonl_dir}")
    print(f"train_images:    {len(train_images)}")
    print(f"val_images:      {len(val_images)}")
    print(f"unique_images:   {len(all_images)}")
    print(f"missing_images:  {missing}")
    print(f"image_bytes:     {total_bytes} ({_format_gb(total_bytes)})")
    print("")
    print("By source:")
    for k in sorted(by_source_count.keys()):
        print(
            f"  {k}: {by_source_count[k]} files, {by_source_bytes[k]} bytes ({_format_gb(by_source_bytes[k])})"
        )

    if args.budget_gb > 0:
        budget_bytes = int(args.budget_gb * (1024**3))
        if total_bytes > budget_bytes:
            print("")
            print(
                f"WARNING: estimated image bytes exceed budget "
                f"({_format_gb(total_bytes)} > {args.budget_gb:.2f} GB)"
            )
        else:
            print("")
            print(
                f"OK: estimated image bytes within budget "
                f"({_format_gb(total_bytes)} <= {args.budget_gb:.2f} GB)"
            )

    if args.dry_run:
        print("\nDry run complete. No files copied.")
        return

    out_jsonl_dir = out_root / "prepared_jsonl"
    out_jsonl_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_jsonl, out_jsonl_dir / args.train_name)
    shutil.copy2(val_jsonl, out_jsonl_dir / args.val_name)

    copied = 0
    for rel in sorted(all_images):
        src = data_root / rel
        if not src.exists():
            continue
        dst = out_root / rel
        _copy_or_link(src, dst, args.mode)
        copied += 1
        if copied % 5000 == 0:
            print(f"copied {copied}/{len(all_images)} ...")

    print("\nMaterialization complete.")
    print(f"out_root: {out_root}")
    print(f"copied_images: {copied}")
    print(f"jsonl: {out_jsonl_dir / args.train_name}")
    print(f"jsonl: {out_jsonl_dir / args.val_name}")
    print("\nUse config with:")
    print(f"  data.mode: jsonl")
    print(f"  data.image_root: {out_root.as_posix()}")
    print(f"  data.train_jsonl: {(out_jsonl_dir / args.train_name).as_posix()}")
    print(f"  data.val_jsonl: {(out_jsonl_dir / args.val_name).as_posix()}")


if __name__ == "__main__":
    main()
