# src/lpcvc_retrieval/model.py
"""
Model Factory for MobileCLIP2 Student Model.

This module provides the factory function to create models from config.
Legacy ClipLite code has been removed - now uses Apple MobileCLIP2.
"""
from __future__ import annotations
from typing import Union, Optional
import torch.nn as nn

from .dual_tower import DualTowerStudent
from .mobileclip2 import MobileCLIP2Student


def create_model_from_config(cfg, vocab_size: int = None, eos_id: int = None) -> nn.Module:
    """
    Factory function to create MobileCLIP2 Student model from config.
    
    Note: vocab_size, eos_id are kept for backward compatibility with scripts
    but are not used (MobileCLIP2 has its own tokenizer).
    
    Args:
        cfg: Configuration object with model parameters
        vocab_size: (unused) Legacy parameter for compatibility
        eos_id: (unused) Legacy parameter for compatibility
    
    Returns:
        Student model instance
    
    Example:
        model = create_model_from_config(cfg)
        model = model.to(device)
    """
    student_type = str(cfg.model.get("student_type", "mobileclip2")).lower()

    if student_type == "mobileclip2":
        return MobileCLIP2Student(
            variant=str(cfg.model.mobileclip2_variant),
            embed_dim=int(cfg.model.get("embed_dim", 256)),
            freeze_backbone=bool(cfg.model.get("freeze_backbone", False)),
            checkpoint_path=cfg.model.get("checkpoint_path", None),
        )

    if student_type == "dual_tower":
        return DualTowerStudent(
            image_model_name=str(cfg.model.get("image_model_name")),
            text_model_name=str(cfg.model.get("text_model_name")),
            embed_dim=int(cfg.model.get("embed_dim", 256)),
            image_pretrained=bool(cfg.model.get("image_pretrained", True)),
            text_pretrained=cfg.model.get("text_pretrained", None),
            freeze_image_backbone=bool(cfg.model.get("freeze_image_backbone", False)),
            freeze_text_backbone=bool(cfg.model.get("freeze_text_backbone", False)),
            image_input_size=int(cfg.model.get("image_input_size", 224)),
        )

    raise ValueError(f"Unknown model.student_type: {student_type}")


class OnnxWrapper(nn.Module):
    """ONNX export wrapper for the retrieval student."""
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image, text_input):
        img, txt = self.model(image, text_input)
        return img, txt
