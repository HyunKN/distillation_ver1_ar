import argparse
import torch
from torch.utils.data import DataLoader
from lpcvc_retrieval.config import load_config, resolve_device
from lpcvc_retrieval.data import build_tokenizer, make_datasets, collate_fn
from lpcvc_retrieval.model import ClipLite
from lpcvc_retrieval.metrics import bidirectional_recall, format_metrics, recall_at_1_5_10

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    device = resolve_device(cfg.get("device", "auto"))

    tok = build_tokenizer()
    vocab_size = int(tok.vocab_size)
    eos_id = int(tok.eos_token_id)

    model = ClipLite(
        vision_backbone=str(cfg.model.vision_backbone),
        vision_pretrained=bool(cfg.model.get("vision_pretrained", True)),
        normalize_input=bool(cfg.model.get("normalize_input", True)),
        clip_mean=list(cfg.model.get("clip_mean", [0.48145466, 0.4578275, 0.40821073])),
        clip_std=list(cfg.model.get("clip_std", [0.26862954, 0.26130258, 0.27577711])),
        embed_dim=int(cfg.model.get("embed_dim", 256)),
        vocab_size=vocab_size,
        context_length=77,
        text_width=int(cfg.model.get("text_width", 256)),
        text_layers=int(cfg.model.get("text_layers", 4)),
        text_heads=int(cfg.model.get("text_heads", 4)),
        text_mlp_ratio=float(cfg.model.get("text_mlp_ratio", 4.0)),
        dropout=float(cfg.model.get("dropout", 0.0)),
        temperature_init=float(cfg.model.get("temperature_init", 0.07)),
        eos_id=eos_id,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    _, val_ds = make_datasets(cfg, tok)
    loader = DataLoader(val_ds, batch_size=int(cfg.data.get("batch_size", 64)), shuffle=False,
                        num_workers=int(cfg.data.get("num_workers", 4)), pin_memory=(device=="cuda"),
                        collate_fn=collate_fn, drop_last=False)

    all_img, all_txt = [], []
    for imgs, toks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        toks = toks.to(device, non_blocking=True)
        img_emb, txt_emb = model(imgs, toks)
        all_img.append(img_emb.cpu())
        all_txt.append(txt_emb.cpu())

    image_emb = torch.cat(all_img, dim=0)
    text_emb = torch.cat(all_txt, dim=0)
    metrics = bidirectional_recall(image_emb, text_emb)
    print(format_metrics(metrics))

    # Optional: keep I2T-only lines for quick comparison
    r1, r5, r10 = recall_at_1_5_10(image_emb, text_emb)
    print(f"Recall@1 : {r1*100:.2f}%")
    print(f"Recall@5 : {r5*100:.2f}%")
    print(f"Recall@10: {r10*100:.2f}%")
if __name__ == "__main__":
    main()
