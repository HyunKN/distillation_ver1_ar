import argparse
import torch

from lpcvc_retrieval.config import load_config
from lpcvc_retrieval.data import build_tokenizer
from lpcvc_retrieval.model import create_model_from_config
from lpcvc_retrieval.export import export_onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="model.onnx")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    tok = build_tokenizer()

    model = create_model_from_config(cfg, tok.vocab_size, tok.eos_token_id)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    export_onnx(model, args.out, opset=int(cfg.export.get("opset", 17)))
    print("Exported ONNX:", args.out)


if __name__ == "__main__":
    main()
