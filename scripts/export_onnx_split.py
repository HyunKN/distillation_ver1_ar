import argparse
import os
import torch
from datetime import datetime

from lpcvc_retrieval.config import load_config
from lpcvc_retrieval.data import build_tokenizer
from lpcvc_retrieval.model import ClipLite
from lpcvc_retrieval.export import export_onnx_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="exported_onnx")
    ap.add_argument("--prefix", default="", help="Optional prefix for output filenames (e.g. 'fastvit_s12')")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.override)
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
    )

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    # Generate timestamp-based filenames (prevents accidental overwrite)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    prefix = f"{args.prefix}_" if args.prefix else ""
    img_name = f"{prefix}image_encoder_{timestamp}.onnx"
    txt_name = f"{prefix}text_encoder_{timestamp}.onnx"

    opset = int(cfg.export.get("opset", 18))
    os.makedirs(args.out_dir, exist_ok=True)
    img_path, txt_path = export_onnx_split(
        model, 
        out_dir=args.out_dir, 
        opset=opset,
        img_name=img_name,
        txt_name=txt_name
    )

    print("Exported ONNX (split):")
    print(" -", img_path)
    print(" -", txt_path)


if __name__ == "__main__":
    main()
