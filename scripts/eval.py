import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure local src/ takes precedence over any globally installed package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lpcvc_retrieval.config import load_config, resolve_device
from lpcvc_retrieval.data import build_tokenizer, make_datasets, collate_fn
from lpcvc_retrieval.model import create_model_from_config
from lpcvc_retrieval.metrics import format_metrics, coco_bidirectional_recall, bidirectional_recall


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    device = resolve_device(cfg.get("device", "auto"))

    tok = build_tokenizer(cfg)
    
    # Use factory function
    model = create_model_from_config(cfg, tok.vocab_size, tok.eos_token_id)
    model = model.to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    _, val_ds = make_datasets(cfg, tok)
    loader = DataLoader(
        val_ds,
        batch_size=int(cfg.data.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(cfg.data.get("num_workers", 4)),
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )

    all_img, all_txt = [], []
    all_image_ids = []
    
    for imgs, toks, metas in tqdm(loader, desc="Evaluating"):
        imgs = imgs.to(device, non_blocking=True)
        toks = toks.to(device, non_blocking=True)
        img_emb, txt_emb = model(imgs, toks)
        all_img.append(img_emb.cpu())
        all_txt.append(txt_emb.cpu())
        
        for meta in metas:
            if isinstance(meta, dict) and meta.get("image_id") is not None:
                all_image_ids.append(meta["image_id"])
            else:
                all_image_ids.append(None)

    image_emb = torch.cat(all_img, dim=0)
    text_emb = torch.cat(all_txt, dim=0)
    
    has_image_ids = all(img_id is not None for img_id in all_image_ids)
    
    if has_image_ids:
        seen_image_ids = {}
        unique_image_embs = []
        unique_image_ids = []
        
        for idx, img_id in enumerate(all_image_ids):
            if img_id not in seen_image_ids:
                seen_image_ids[img_id] = len(unique_image_ids)
                unique_image_embs.append(image_emb[idx])
                unique_image_ids.append(img_id)
        
        unique_image_emb = torch.stack(unique_image_embs, dim=0)
        text_image_ids = all_image_ids
        
        print(f"[COCO Eval] {len(unique_image_ids)} unique images, {len(text_image_ids)} captions")
        
        metrics = coco_bidirectional_recall(
            unique_image_emb, text_emb, unique_image_ids, text_image_ids
        )
    else:
        print("[Index-based Eval] No image_ids found, using legacy evaluation")
        metrics = bidirectional_recall(image_emb, text_emb)
    
    print(format_metrics(metrics))

    i2t = metrics['I2T']
    print(f"Recall@1 : {i2t['R@1']*100:.2f}%")
    print(f"Recall@5 : {i2t['R@5']*100:.2f}%")
    print(f"Recall@10: {i2t['R@10']*100:.2f}%")
    competition = metrics.get("competition", {})
    if competition:
        print(f"Competition I2T Text Recall@1 : {competition['I2T_text_R@1']*100:.2f}%")
        print(f"Competition I2T Text Recall@5 : {competition['I2T_text_R@5']*100:.2f}%")
        print(f"Competition I2T Text Recall@10: {competition['I2T_text_R@10']*100:.2f}%")


if __name__ == "__main__":
    main()
