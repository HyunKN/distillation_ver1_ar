#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def _parse_source_caps(spec: str) -> Dict[str, int]:
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
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid source_caps key in '{part}'")
        caps[key] = int(value)
    return caps


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sample_rows(rows: List[dict], fraction: float, seed: int, source_caps: Dict[str, int]) -> List[dict]:
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        source = str(row.get("source", "unknown"))
        grouped[source].append(row)

    out: List[dict] = []
    for source, items in sorted(grouped.items()):
        local = list(items)
        rng = random.Random(seed + sum(ord(ch) for ch in source))
        rng.shuffle(local)

        cap = int(source_caps.get(source, 0))
        if cap > 0:
            n_keep = min(cap, len(local))
        elif fraction >= 1.0:
            n_keep = len(local)
        else:
            n_keep = int(len(local) * fraction)
            if len(local) > 0 and n_keep == 0:
                n_keep = 1

        out.extend(local[:n_keep])

    rng = random.Random(seed)
    rng.shuffle(out)
    return out


def _print_stats(name: str, rows: List[dict]) -> None:
    grouped: Dict[str, int] = defaultdict(int)
    for row in rows:
        grouped[str(row.get("source", "unknown"))] += 1

    print(f"[{name}] rows={len(rows)}")
    for source, count in sorted(grouped.items()):
        print(f"  - {source}: {count}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create deterministic JSONL subsets for autoresearch-lite experiments."
    )
    ap.add_argument("--train_jsonl", required=True, help="Input train JSONL path.")
    ap.add_argument("--val_jsonl", required=True, help="Input val JSONL path.")
    ap.add_argument("--out_dir", required=True, help="Output directory for subset JSONLs.")
    ap.add_argument("--train_fraction", type=float, default=0.1, help="Fraction of train rows to keep.")
    ap.add_argument("--val_fraction", type=float, default=1.0, help="Fraction of val rows to keep.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument(
        "--train_source_caps",
        default="",
        help="Optional per-source caps for train subset. Example: coco=12000,flickr30k=3000",
    )
    ap.add_argument(
        "--val_source_caps",
        default="",
        help="Optional per-source caps for val subset.",
    )
    args = ap.parse_args()

    train_path = Path(args.train_jsonl)
    val_path = Path(args.val_jsonl)
    out_dir = Path(args.out_dir)

    if not train_path.exists():
        raise FileNotFoundError(f"train_jsonl not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"val_jsonl not found: {val_path}")

    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)

    train_subset = _sample_rows(
        train_rows,
        fraction=float(args.train_fraction),
        seed=int(args.seed),
        source_caps=_parse_source_caps(args.train_source_caps),
    )
    val_subset = _sample_rows(
        val_rows,
        fraction=float(args.val_fraction),
        seed=int(args.seed) + 1000,
        source_caps=_parse_source_caps(args.val_source_caps),
    )

    train_out = out_dir / "train.jsonl"
    val_out = out_dir / "val.jsonl"
    _write_jsonl(train_out, train_subset)
    _write_jsonl(val_out, val_subset)

    _print_stats("train_subset", train_subset)
    _print_stats("val_subset", val_subset)
    print(f"\nFiles written:")
    print(f"  train: {train_out}")
    print(f"  val:   {val_out}")


if __name__ == "__main__":
    main()

