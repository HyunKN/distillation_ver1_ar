"""
Offline Feature Extraction Script.

Pre-extract Teacher embeddings to .pt files so that Teacher models
do not occupy VRAM during training.

Usage:
    python scripts/extract_features.py --config config.yaml --out_dir features/
    python scripts/extract_features.py --config config.yaml --out_dir features/ --fp32
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lpcvc_retrieval.config import load_config
from lpcvc_retrieval.data import build_tokenizer, make_datasets, collate_fn
from lpcvc_retrieval.distill import DistillConfig, create_teacher


def extract_features_for_split(
    teacher,
    loader: DataLoader,
    device: str,
    save_dtype: torch.dtype = torch.float16,
) -> dict:
    """Extract (img_emb, txt_emb) for each Teacher across the entire DataLoader."""
    teacher.eval()
    
    # Determine number of teachers
    from lpcvc_retrieval.distill import EnsembleTeacher
    if isinstance(teacher, EnsembleTeacher):
        num_teachers = len(teacher.teachers)
    else:
        num_teachers = 1
    
    # Accumulate per-teacher
    all_img_embs = [[] for _ in range(num_teachers)]
    all_txt_embs = [[] for _ in range(num_teachers)]
    all_caption_indices = []
    dataset_hasher = hashlib.sha1()
    
    with torch.no_grad():
        for imgs, toks, metas in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(device, non_blocking=True).float()
            raw_captions = [m["caption"] for m in metas]
            cap_idx = torch.tensor(
                [int(m.get("caption_idx", -1)) if m.get("caption_idx", None) is not None else -1 for m in metas],
                dtype=torch.int32,
            )
            all_caption_indices.append(cap_idx)
            for m in metas:
                image_id = int(m.get("image_id", -1)) if m.get("image_id", None) is not None else -1
                img_rel = str(m.get("img_rel", ""))
                dataset_hasher.update(f"{image_id}|{img_rel}\n".encode("utf-8"))
            
            # Teacher forward
            teacher_out = teacher(imgs, raw_captions)
            
            # Normalize output format
            if isinstance(teacher_out, list):
                # EnsembleTeacher: list of (img, txt) tuples
                outputs = teacher_out
            else:
                # Single teacher: (img, txt) tuple
                outputs = [teacher_out]
            
            for t_idx, (t_img, t_txt) in enumerate(outputs):
                all_img_embs[t_idx].append(t_img.cpu().to(save_dtype))
                all_txt_embs[t_idx].append(t_txt.cpu().to(save_dtype))
    
    # Concatenate
    caption_indices = torch.cat(all_caption_indices, dim=0)
    dataset_fingerprint = dataset_hasher.hexdigest()
    results = {}
    for t_idx in range(num_teachers):
        results[t_idx] = {
            "img_embs": torch.cat(all_img_embs[t_idx], dim=0),
            "txt_embs": torch.cat(all_txt_embs[t_idx], dim=0),
            "caption_indices": caption_indices,
            "sample_count": int(caption_indices.shape[0]),
            "dataset_fingerprint": dataset_fingerprint,
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract Teacher embeddings to .pt files."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out_dir", default="features")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config, overrides=args.override)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dtype = torch.float32 if args.fp32 else torch.float16
    train_augment = bool(cfg.data.get("train_augment", True))
    
    print(f"[Extract] Device: {device}")
    print(f"[Extract] Save dtype: {save_dtype}")
    print(f"[Extract] Output dir: {args.out_dir}")
    print(f"[Extract] Effective data.train_augment: {train_augment}")
    if train_augment:
        print("[Extract] Warning: train_augment=True. Teacher features will be extracted on augmented views.")

    # Load teacher
    distill_section = cfg.get("distill", {})
    
    def _to_clean_dict(obj):
        if hasattr(obj, 'as_dict'): return _to_clean_dict(obj.as_dict())
        if isinstance(obj, list): return [_to_clean_dict(x) for x in obj]
        if isinstance(obj, dict): return {k: _to_clean_dict(v) for k, v in obj.items()}
        return obj

    distill_section = _to_clean_dict(distill_section)
    distill_cfg = DistillConfig(**distill_section) if isinstance(distill_section, dict) else DistillConfig()

    if not distill_cfg.use_teacher:
        print("[Extract] Error: distill.use_teacher is False. Enable it in config.yaml.")
        return

    teacher = create_teacher(distill_cfg, device=device)
    if hasattr(teacher, "teachers"):
        print(f"[Extract] Teacher loaded: EnsembleTeacher(num_teachers={len(teacher.teachers)})")
    else:
        print(f"[Extract] Teacher loaded: {teacher.__class__.__name__}")

    # Build datasets (force offline wrapping OFF to avoid self-contamination)
    if hasattr(cfg, "as_dict"):
        cfg_dict = cfg.as_dict()
        cfg_dict.setdefault("distill", {})["offline_feature_dir"] = None
    elif isinstance(cfg, dict):
        cfg.setdefault("distill", {})["offline_feature_dir"] = None
    tokenizer = build_tokenizer()
    train_ds, val_ds = make_datasets(cfg, tokenizer)

    batch_size = args.batch_size or int(cfg.data.get("batch_size", 64))
    num_workers = int(cfg.data.get("num_workers", 4))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
        collate_fn=collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
        collate_fn=collate_fn, drop_last=False,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # Extract & Save — Train
    print("\n=== Extracting TRAIN features ===")
    train_results = extract_features_for_split(teacher, train_loader, device, save_dtype)
    for t_idx, data in train_results.items():
        path = os.path.join(args.out_dir, f"teacher_{t_idx}_train.pt")
        torch.save(data, path)
        print(f"[Saved] {path} | img: {data['img_embs'].shape}, txt: {data['txt_embs'].shape}")

    # Extract & Save — Val
    print("\n=== Extracting VAL features ===")
    val_results = extract_features_for_split(teacher, val_loader, device, save_dtype)
    for t_idx, data in val_results.items():
        path = os.path.join(args.out_dir, f"teacher_{t_idx}_val.pt")
        torch.save(data, path)
        print(f"[Saved] {path} | img: {data['img_embs'].shape}, txt: {data['txt_embs'].shape}")

    print(f"\n[Extract] Done! All features saved to: {args.out_dir}")
    print(f"[Extract] Set in config.yaml:")
    print(f"  distill:")
    print(f"    offline_feature_dir: {args.out_dir}")


if __name__ == "__main__":
    main()
