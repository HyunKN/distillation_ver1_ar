import argparse
from lpcvc_retrieval.config import load_config
from lpcvc_retrieval.train import train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--override", action="append", default=[], help="override like model.embed_dim=128")
    args = ap.parse_args()
    cfg = load_config(args.config, overrides=args.override)
    ckpt_path = train(cfg)
    print("Saved checkpoint:", ckpt_path)

if __name__ == "__main__":
    main()
