# src/lpcvc_retrieval/model.py
"""Model factory for the current dual-tower retrieval student."""
from __future__ import annotations
import torch.nn as nn

from .dual_tower import DualTowerStudent


def create_model_from_config(cfg, vocab_size: int = None, eos_id: int = None) -> nn.Module:
    """
    Create the current dual-tower student from config.

    Note: ``vocab_size`` and ``eos_id`` are kept only for call-site compatibility.

    Args:
        cfg: Configuration object with model parameters
        vocab_size: Unused compatibility parameter
        eos_id: Unused compatibility parameter

    Returns:
        Student model instance
    """
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


class OnnxWrapper(nn.Module):
    """ONNX export wrapper for the retrieval student."""
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image, text_input):
        img, txt = self.model(image, text_input)
        return img, txt
