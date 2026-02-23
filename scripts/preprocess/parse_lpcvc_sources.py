#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlsplit


def _clean_text(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if not text:
        return ""
    return " ".join(text.split())


def _parse_source_caps(spec: str) -> Dict[str, int]:
    """
    Parse source-specific caps.
    Format:
      "open_images=40000,coco=0"
    Meaning:
      - value > 0: cap to that many unique images
      - value <= 0: no cap
    """
    caps: Dict[str, int] = {}
    spec = (spec or "").strip()
    if not spec:
        return caps
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid source_caps item: '{part}' (expected key=value)")
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Invalid source_caps key in '{part}'")
        try:
            n = int(v)
        except ValueError as e:
            raise ValueError(f"Invalid source_caps value in '{part}'") from e
        caps[k] = n
    return caps


def _add_caption(
    store: Dict[str, Dict[str, object]],
    image_rel: str,
    caption: str,
    source: str,
) -> None:
    if image_rel not in store:
        store[image_rel] = {
            "image": image_rel,
            "captions": [],
            "source": source,
        }
    store[image_rel]["captions"].append(caption)


def _iter_coco(
    root: Path,
    splits: List[str],
    sample_limit: int,
    check_exists: bool,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, int]]:
    stats = defaultdict(int)
    data: Dict[str, Dict[str, object]] = {}

    for split in splits:
        ann_path = root / "coco" / "annotations" / f"captions_{split}.json"
        if not ann_path.exists():
            continue

        with ann_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        id_to_file = {int(x["id"]): x["file_name"] for x in coco.get("images", [])}

        for ann in coco.get("annotations", []):
            stats["rows_total"] += 1
            image_id = int(ann.get("image_id", -1))
            file_name = id_to_file.get(image_id)
            caption = _clean_text(ann.get("caption"))
            if not file_name:
                stats["rows_missing_meta"] += 1
                continue
            if not caption:
                stats["rows_empty_caption"] += 1
                continue

            image_rel = f"coco/{split}/{file_name}".replace("\\", "/")
            if check_exists and not (root / image_rel).exists():
                stats["rows_missing_image"] += 1
                continue

            _add_caption(data, image_rel, caption, "coco")
            stats["rows_valid"] += 1
            if sample_limit > 0 and len(data) >= sample_limit:
                break

        if sample_limit > 0 and len(data) >= sample_limit:
            break

    stats["unique_images"] = len(data)
    return data, dict(stats)


def _iter_flickr30k(
    root: Path,
    sample_limit: int,
    check_exists: bool,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, int]]:
    stats = defaultdict(int)
    data: Dict[str, Dict[str, object]] = {}

    csv_path = root / "flickr30k" / "flickr30k_images" / "results.csv"
    if not csv_path.exists():
        stats["missing_csv"] = 1
        return data, dict(stats)

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            stats["rows_total"] += 1
            image_name = _clean_text(row.get("image_name") or row.get(" image_name"))
            caption = _clean_text(row.get(" comment") or row.get("comment"))
            if not image_name:
                stats["rows_missing_meta"] += 1
                continue
            if not caption:
                stats["rows_empty_caption"] += 1
                continue

            image_rel = f"flickr30k/flickr30k_images/flickr30k_images/{image_name}".replace(
                "\\", "/"
            )
            if check_exists and not (root / image_rel).exists():
                stats["rows_missing_image"] += 1
                continue

            _add_caption(data, image_rel, caption, "flickr30k")
            stats["rows_valid"] += 1
            if sample_limit > 0 and len(data) >= sample_limit:
                break

    stats["unique_images"] = len(data)
    return data, dict(stats)


def _find_open_images_file(base_dir: Path, image_id: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = base_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def _iter_open_images(
    root: Path,
    sample_limit: int,
    check_exists: bool,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, int]]:
    stats = defaultdict(int)
    data: Dict[str, Dict[str, object]] = {}

    jsonl_path = root / "open_images" / "open_images_test_localized_narratives.jsonl"
    image_dir = root / "open_images" / "test"
    if not jsonl_path.exists():
        stats["missing_jsonl"] = 1
        return data, dict(stats)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            stats["rows_total"] += 1
            obj = json.loads(line)
            image_id = _clean_text(obj.get("image_id"))
            caption = _clean_text(obj.get("caption"))
            if not image_id:
                stats["rows_missing_meta"] += 1
                continue
            if not caption:
                stats["rows_empty_caption"] += 1
                continue

            image_path = _find_open_images_file(image_dir, image_id)
            if check_exists and image_path is None:
                stats["rows_missing_image"] += 1
                continue
            if image_path is None:
                image_rel = f"open_images/test/{image_id}.jpg"
            else:
                image_rel = str(image_path.relative_to(root)).replace("\\", "/")

            _add_caption(data, image_rel, caption, "open_images")
            stats["rows_valid"] += 1
            if sample_limit > 0 and len(data) >= sample_limit:
                break

    stats["unique_images"] = len(data)
    return data, dict(stats)


def _pick_wit_caption(row: Dict[str, str]) -> str:
    candidates = [
        row.get("caption_reference_description", ""),
        row.get("caption_attribution_description", ""),
        row.get("caption_alt_text_description", ""),
        row.get("context_section_description", ""),
        row.get("context_page_description", ""),
    ]
    for c in candidates:
        c = _clean_text(c)
        if c:
            return c
    return ""


def _wit_file_candidates(image_url: str) -> List[str]:
    path = urlsplit(image_url).path
    if not path:
        return []
    raw_name = path.rsplit("/", 1)[-1]
    if not raw_name:
        return []
    unquoted = unquote(raw_name)
    if unquoted == raw_name:
        return [raw_name]
    return [raw_name, unquoted]


def _iter_wit(
    root: Path,
    sample_limit: int,
    check_exists: bool,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, int]]:
    stats = defaultdict(int)
    data: Dict[str, Dict[str, object]] = {}

    tsv_gz_path = root / "wit" / "wit_v1.train.all-1percent_sample.tsv.gz"
    image_dir = root / "wit" / "images"
    if not tsv_gz_path.exists():
        stats["missing_tsv_gz"] = 1
        return data, dict(stats)

    # Windows is case-insensitive, Linux is case-sensitive.
    # Build a lowercase -> actual filename map to keep canonical on-disk casing.
    image_name_map: Dict[str, str] = {}
    if check_exists and image_dir.exists():
        for p in image_dir.iterdir():
            if p.is_file():
                image_name_map[p.name.lower()] = p.name

    with gzip.open(tsv_gz_path, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            stats["rows_total"] += 1
            image_url = _clean_text(row.get("image_url"))
            caption = _pick_wit_caption(row)
            if not image_url:
                stats["rows_missing_meta"] += 1
                continue
            if not caption:
                stats["rows_empty_caption"] += 1
                continue

            image_rel = ""
            if check_exists:
                found = None
                for name in _wit_file_candidates(image_url):
                    actual_name = image_name_map.get(name.lower())
                    if actual_name is not None:
                        found = image_dir / actual_name
                        break
                if found is None:
                    stats["rows_missing_image"] += 1
                    continue
                image_rel = str(found.relative_to(root)).replace("\\", "/")
            else:
                names = _wit_file_candidates(image_url)
                if not names:
                    stats["rows_missing_meta"] += 1
                    continue
                image_rel = f"wit/images/{names[0]}"

            _add_caption(data, image_rel, caption, "wit")
            stats["rows_valid"] += 1
            if sample_limit > 0 and len(data) >= sample_limit:
                break

    stats["unique_images"] = len(data)
    return data, dict(stats)


def _dedupe_records(records: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for obj in records.values():
        caps: List[str] = obj["captions"]  # type: ignore[assignment]
        dedup = list(OrderedDict.fromkeys(caps))
        if not dedup:
            continue
        out.append(
            {
                "image": obj["image"],
                "captions": dedup,
                "source": obj["source"],
            }
        )
    return out


def _split_train_val(
    records: List[Dict[str, object]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in records:
        groups[str(r.get("source", "unknown"))].append(r)

    rng = random.Random(seed)
    train: List[Dict[str, object]] = []
    val: List[Dict[str, object]] = []

    for _, items in groups.items():
        rng.shuffle(items)
        n_val = int(len(items) * val_ratio)
        if val_ratio > 0 and n_val == 0 and len(items) > 1:
            n_val = 1
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _print_stats(source: str, stats: Dict[str, int], record_count: int, caption_count: int) -> None:
    print(f"[{source}]")
    print(f"  rows_total:         {stats.get('rows_total', 0)}")
    print(f"  rows_valid:         {stats.get('rows_valid', 0)}")
    print(f"  rows_empty_caption: {stats.get('rows_empty_caption', 0)}")
    print(f"  rows_missing_meta:  {stats.get('rows_missing_meta', 0)}")
    print(f"  rows_missing_image: {stats.get('rows_missing_image', 0)}")
    print(f"  unique_images:      {record_count}")
    print(f"  total_captions:     {caption_count}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse LPCVC_Data (coco/flickr30k/open_images/wit) into project JSONL format."
    )
    ap.add_argument("--data_root", default=r"D:\LPCVC_Data", help="Root directory of datasets.")
    ap.add_argument("--out_dir", default="dataset/lpcvc", help="Output directory.")
    ap.add_argument("--train_name", default="train.jsonl", help="Train JSONL filename.")
    ap.add_argument("--val_name", default="val.jsonl", help="Val JSONL filename.")
    ap.add_argument("--val_ratio", type=float, default=0.01, help="Validation split ratio.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument(
        "--sources",
        default="coco,flickr30k,open_images,wit",
        help="Comma-separated source list.",
    )
    ap.add_argument(
        "--coco_splits",
        default="train2017",
        help="Comma-separated COCO splits (train2017,val2017).",
    )
    ap.add_argument(
        "--sample_per_source",
        type=int,
        default=0,
        help="If >0, stop after N unique images per source for dry validation.",
    )
    ap.add_argument(
        "--source_caps",
        default="",
        help=(
            "Per-source unique image caps, comma-separated. "
            "Example: open_images=40000,coco=0 (0 means no cap)."
        ),
    )
    ap.add_argument(
        "--show_examples",
        type=int,
        default=0,
        help="Print first N parsed records after merge.",
    )
    ap.add_argument(
        "--no_check_exists",
        action="store_true",
        help="Do not verify image file existence.",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Parse and report only; do not write JSONL files.",
    )
    args = ap.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root not found: {root}")

    source_names = [s.strip() for s in args.sources.split(",") if s.strip()]
    coco_splits = [s.strip() for s in args.coco_splits.split(",") if s.strip()]
    sample_limit = int(args.sample_per_source)
    source_caps = _parse_source_caps(args.source_caps)
    check_exists = not args.no_check_exists

    parser_map = {
        "coco": lambda lim: _iter_coco(root, coco_splits, lim, check_exists),
        "flickr30k": lambda lim: _iter_flickr30k(root, lim, check_exists),
        "open_images": lambda lim: _iter_open_images(root, lim, check_exists),
        "wit": lambda lim: _iter_wit(root, lim, check_exists),
    }

    merged: List[Dict[str, object]] = []
    source_totals = {}

    for src in source_names:
        if src not in parser_map:
            print(f"[warn] Unknown source skipped: {src}")
            continue
        # sample_per_source: fast dry validation cap (early-stop while parsing)
        source_dict, stats = parser_map[src](sample_limit)
        records = _dedupe_records(source_dict)

        # source_caps: production cap (random image-level sampling after full parse)
        cap = source_caps.get(src, 0)
        if cap > 0 and len(records) > cap:
            local_seed = int(args.seed) + sum(ord(ch) for ch in src)
            rng = random.Random(local_seed)
            records = rng.sample(records, cap)
            print(f"[{src}] applied source_caps={cap} (random sampled from full parsed records)")

        caption_count = sum(len(x["captions"]) for x in records)
        _print_stats(src, stats, len(records), caption_count)
        merged.extend(records)
        source_totals[src] = {"images": len(records), "captions": caption_count}

    if not merged:
        raise RuntimeError("No records parsed. Check paths and dataset files.")

    train_rows, val_rows = _split_train_val(merged, args.val_ratio, args.seed)

    print("\n[merged]")
    print(f"  total_images:   {len(merged)}")
    print(f"  total_captions: {sum(len(r['captions']) for r in merged)}")
    print(f"  train_images:   {len(train_rows)}")
    print(f"  val_images:     {len(val_rows)}")

    if args.show_examples > 0:
        print("\n[examples]")
        for i, row in enumerate(merged[: args.show_examples], start=1):
            print(f"{i}. source={row['source']} image={row['image']} caps={len(row['captions'])}")
            print(f"   caption: {row['captions'][0]}")

    if args.dry_run:
        print("\nDry run complete. No files written.")
        return

    out_dir = Path(args.out_dir)
    train_path = out_dir / args.train_name
    val_path = out_dir / args.val_name
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)

    print("\nFiles written:")
    print(f"  train: {train_path}")
    print(f"  val:   {val_path}")
    print("\nUse with config:")
    print(f"  data.mode: jsonl")
    print(f"  data.image_root: {root.as_posix()}")
    print(f"  data.train_jsonl: {train_path.as_posix()}")
    print(f"  data.val_jsonl: {val_path.as_posix()}")


if __name__ == "__main__":
    main()
