#!/usr/bin/env python
"""
MobileCLIP2-Retrieval-Optimization Training Script
===================================================
CMD에서 쉽게 학습을 실행할 수 있는 엔트리포인트입니다.

Usage:
    python run_train.py --config config.yaml
    python run_train.py  # config.yaml이 기본값
"""
import argparse
import sys
import torch
from datetime import datetime
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lpcvc_retrieval.config import load_config
from lpcvc_retrieval.train import train


def main():
    parser = argparse.ArgumentParser(
        description="Train MobileCLIP2 Retrieval Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to config YAML file"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[Error] Config file not found: {config_path}")
        sys.exit(1)

    print(f"[Config] Loading: {config_path}")
    cfg = load_config(str(config_path))
    
    # GPU 정보 출력
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_info = f"GPU: {device_name}"
    else:
        device_info = "CPU (GPU not available)"
    
    print("=" * 60)
    print("  MobileCLIP2-Retrieval-Optimization Training")
    print("=" * 60)
    print(f"  Device: {device_info}")
    print(f"  MobileCLIP2 Variant: {cfg.model.mobileclip2_variant}")
    print(f"  Epochs: {cfg.train.epochs}")
    print(f"  Batch Size: {cfg.data.batch_size}")
    print(f"  Learning Rate: {cfg.train.lr}")
    print(f"  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    best_path = train(cfg)
    
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Best Model: {best_path}")
    print(f"  End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
