# src/lpcvc_retrieval/ema.py
"""
Exponential Moving Average (EMA) for model weights.

EMA maintains a shadow copy of model weights that is updated as:
    ema_weight = decay * ema_weight + (1 - decay) * current_weight

This produces smoother, more stable weights that often generalize better.

Usage:
    ema = EMA(model, decay=0.999)
    
    for batch in dataloader:
        # Normal training step
        loss.backward()
        optimizer.step()
        
        # Update EMA weights
        ema.update()
    
    # For evaluation, use EMA weights
    ema.apply_shadow()
    evaluate(model)
    ema.restore()

References:
    - Polyak averaging (1992)
    - Used in CLIP, Stable Diffusion, GPT, etc.
"""
from __future__ import annotations

import copy
from typing import Optional
import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average for PyTorch models.
    
    Args:
        model: The model to track
        decay: EMA decay rate (0.999 = 99.9% old, 0.1% new)
        device: Device for shadow weights (None = same as model)
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[str] = None,
    ):
        self.model = model
        self.decay = decay
        self.device = device
        
        # Shadow weights (EMA copy)
        self.shadow = {}
        # Backup for restore
        self.backup = {}
        
        # Initialize shadow weights
        self._init_shadow()
    
    def _init_shadow(self):
        """Initialize shadow weights from current model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if self.device is not None:
                    self.shadow[name] = self.shadow[name].to(self.device)
    
    @torch.no_grad()
    def update(self):
        """Update shadow weights with current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # EMA update: shadow = decay * shadow + (1 - decay) * current
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )
    
    def apply_shadow(self):
        """Apply shadow weights to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original weights after evaluation."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self) -> dict:
        """Get EMA state for checkpointing."""
        return {
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
            "decay": self.decay,
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load EMA state from checkpoint."""
        self.decay = state_dict.get("decay", self.decay)
        for name, tensor in state_dict.get("shadow", {}).items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor.to(self.shadow[name].device))
