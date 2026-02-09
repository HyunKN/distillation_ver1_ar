# src/lpcvc_retrieval/mobileclip2.py
"""
MobileCLIP2 Student Model Wrapper

Apple의 MobileCLIP2-S4 모델을 우리 프레임워크에 통합하는 래퍼입니다.
Apple의 ml-mobileclip 패키지와 open_clip을 사용합니다.

사전 준비:
    pip install open_clip_torch
    pip install git+https://github.com/apple/ml-mobileclip.git

출처: https://huggingface.co/apple/MobileCLIP2-S4
라이선스: MIT License

Usage:
    from .mobileclip2 import MobileCLIP2Student
    
    model = MobileCLIP2Student(variant="S4", checkpoint_path="/path/to/mobileclip2_s4.pt")
    img_emb, txt_emb = model(images, text_tokens)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Any


class MobileCLIP2Student(nn.Module):
    """
    MobileCLIP2 Student Model Wrapper for LPCVC Competition.
    
    Apple의 MobileCLIP2 모델을 로드하고, 우리 학습 파이프라인에 맞게 래핑합니다.
    open_clip API를 사용하여 모델을 로드합니다.
    
    Supported variants:
        - S0: Smallest, fastest
        - S2: Small
        - S3: Medium
        - S4: Largest, highest accuracy (default)
        - B: Base
        - L-14: Large
    """
    
    # Variant name mapping for open_clip
    VARIANT_MAP = {
        "S0": "MobileCLIP2-S0",
        "S2": "MobileCLIP2-S2",
        "S3": "MobileCLIP2-S3",
        "S4": "MobileCLIP2-S4",
        "B": "MobileCLIP2-B",
        "L-14": "MobileCLIP2-L-14",
    }
    
    def __init__(
        self, 
        variant: str = "S4",
        embed_dim: int = 256,
        freeze_backbone: bool = False,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize MobileCLIP2 Student model.
        
        Args:
            variant: Model variant ("S0", "S2", "S3", "S4", "B", "L-14")
            embed_dim: Output embedding dimension (if different from model, adds projection)
            freeze_backbone: Whether to freeze the pretrained backbone
            checkpoint_path: Path to the downloaded MobileCLIP2 checkpoint (.pt file)
        """
        super().__init__()
        
        self.variant = variant.upper()
        self.embed_dim = embed_dim
        self.freeze_backbone = freeze_backbone
        self.checkpoint_path = checkpoint_path
        
        if self.variant not in self.VARIANT_MAP:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(self.VARIANT_MAP.keys())}")
        
        self.model_name = self.VARIANT_MAP[self.variant]
        self._load_model()
        
        # Learnable parameters for contrastive learning
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / 0.07).log())
        self.logit_bias = nn.Parameter(torch.zeros([]))
        
    def _load_model(self):
        """Load the MobileCLIP2 model using open_clip."""
        try:
            import open_clip
            
            print(f"[MobileCLIP2] Loading {self.model_name}...")
            
            # Use dfndr2b pretrained if no checkpoint path specified
            pretrained = self.checkpoint_path if self.checkpoint_path else "dfndr2b"
            
            # Load model with open_clip
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=pretrained,
            )
            
            # Get tokenizer
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            
            # For inference/export, reparameterize the model (required for MobileCLIP2)
            try:
                from mobileclip.modules.common.mobileone import reparameterize_model
                self.clip_model = reparameterize_model(self.clip_model)
                print("[MobileCLIP2] Model reparameterized for inference")
            except ImportError:
                print("[MobileCLIP2] Warning: mobileclip not installed, skipping reparameterization")
            
            # Get the actual embedding dimension from the model
            # Typical CLIP models have projection_dim or we can infer from visual
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                out = self.clip_model.encode_image(dummy)
                self.clip_embed_dim = out.shape[-1]
            
            # Add projection if needed to match target embed_dim
            if self.clip_embed_dim != self.embed_dim:
                self.image_proj = nn.Linear(self.clip_embed_dim, self.embed_dim, bias=False)
                self.text_proj = nn.Linear(self.clip_embed_dim, self.embed_dim, bias=False)
                print(f"[MobileCLIP2] Added projection: {self.clip_embed_dim} -> {self.embed_dim}")
            else:
                self.image_proj = nn.Identity()
                self.text_proj = nn.Identity()
            
            if self.freeze_backbone:
                for param in self.clip_model.parameters():
                    param.requires_grad = False
                print("[MobileCLIP2] Backbone frozen")
            
            print(f"[MobileCLIP2] Loaded successfully! Embed dim: {self.clip_embed_dim}")
            
        except ImportError as e:
            raise ImportError(
                "MobileCLIP2 requires 'open_clip_torch' and 'ml-mobileclip' packages. "
                "Install with: pip install open_clip_torch && pip install git+https://github.com/apple/ml-mobileclip.git"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MobileCLIP2 model '{self.model_name}'. "
                f"Make sure you have downloaded the checkpoint. "
                f"Error: {e}"
            ) from e
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings.
        
        Args:
            images: [B, 3, 224, 224] float tensor in [0, 1] range
        
        Returns:
            Image embeddings [B, embed_dim], L2 normalized
        """
        # open_clip's encode_image returns normalized features
        outputs = self.clip_model.encode_image(images)
        embeddings = self.image_proj(outputs)
        return F.normalize(embeddings, dim=-1)
    
    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode text tokens to embeddings.
        
        Args:
            input_ids: [B, 77] int32/int64 token IDs
        
        Returns:
            Text embeddings [B, embed_dim], L2 normalized
        """
        # open_clip's encode_text expects token IDs
        outputs = self.clip_model.encode_text(input_ids.long())
        embeddings = self.text_proj(outputs)
        return F.normalize(embeddings, dim=-1)
    
    def forward(
        self, 
        images: torch.Tensor, 
        text_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode both images and text.
        
        Args:
            images: [B, 3, 224, 224] float tensor
            text_input: [B, 77] int32 token IDs
        
        Returns:
            Tuple of (image_embeddings, text_embeddings), both L2 normalized
        """
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(text_input)
        return img_emb, txt_emb
    
    def get_tokenizer(self) -> Any:
        """Get the tokenizer for this model."""
        return self._tokenizer


def create_mobileclip2_model(
    variant: str = "S4",
    embed_dim: int = 256,
    freeze_backbone: bool = False,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> MobileCLIP2Student:
    """
    Factory function to create MobileCLIP2 Student model.
    
    Args:
        variant: Model variant ("S0", "S2", "S3", "S4", "B", "L-14")
        embed_dim: Output embedding dimension
        freeze_backbone: Whether to freeze the pretrained weights
        checkpoint_path: Path to the downloaded checkpoint
        device: Device to load model on
    
    Returns:
        MobileCLIP2Student model instance
    """
    model = MobileCLIP2Student(
        variant=variant,
        embed_dim=embed_dim,
        freeze_backbone=freeze_backbone,
        checkpoint_path=checkpoint_path,
    )
    return model.to(device)
