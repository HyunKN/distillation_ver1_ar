import argparse
import os
import sys
import torch
from datetime import datetime

# Ensure local src/ takes precedence over any globally installed package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lpcvc_retrieval.config import load_config
from lpcvc_retrieval.data import build_tokenizer
from lpcvc_retrieval.model import create_model_from_config
from lpcvc_retrieval.export import export_onnx_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="exported_onnx")
    ap.add_argument("--prefix", default="", help="Optional prefix for output filenames")
    ap.add_argument("--add_timestamp", action="store_true", help="Append timestamp to output filenames")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    tok = build_tokenizer(cfg)

    model = create_model_from_config(cfg, tok.vocab_size, tok.eos_token_id)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    prefix = f"{args.prefix}_" if args.prefix else ""
    if args.add_timestamp:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        img_name = f"{prefix}image_encoder_{timestamp}.onnx"
        txt_name = f"{prefix}text_encoder_{timestamp}.onnx"
    else:
        img_name = f"{prefix}image_encoder.onnx"
        txt_name = f"{prefix}text_encoder.onnx"

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
