# src/lpcvc_retrieval/model.py
"""
Model Factory for MobileCLIP2 Student Model.

This module provides the factory function to create models from config.
Legacy ClipLite code has been removed - now uses Apple MobileCLIP2.
"""
from __future__ import annotations
from typing import Union, Optional
import torch.nn as nn

from .mobileclip2 import MobileCLIP2Student


def create_model_from_config(cfg, vocab_size: int = None, eos_id: int = None) -> MobileCLIP2Student:
    """
    Factory function to create MobileCLIP2 Student model from config.
    
    Note: vocab_size, eos_id are kept for backward compatibility with scripts
    but are not used (MobileCLIP2 has its own tokenizer).
    
    Args:
        cfg: Configuration object with model parameters
        vocab_size: (unused) Legacy parameter for compatibility
        eos_id: (unused) Legacy parameter for compatibility
    
    Returns:
        MobileCLIP2Student model instance
    
    Example:
        model = create_model_from_config(cfg)
        model = model.to(device)
    """
    return MobileCLIP2Student(
        variant=str(cfg.model.mobileclip2_variant),
        embed_dim=int(cfg.model.get("embed_dim", 256)),
        freeze_backbone=bool(cfg.model.get("freeze_backbone", False)),
        checkpoint_path=cfg.model.get("checkpoint_path", None),
    )


class OnnxWrapper(nn.Module):
    """ONNX export wrapper for MobileCLIP2Student."""
    
    def __init__(self, model: MobileCLIP2Student):
        super().__init__()
        self.model = model

    def forward(self, image, text_input):
        img, txt = self.model(image, text_input)
        return img, txt
