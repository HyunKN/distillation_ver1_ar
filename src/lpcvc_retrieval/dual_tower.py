from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualTowerStudent(nn.Module):
    """Retrieval student with a timm image tower and an open_clip text tower."""

    def __init__(
        self,
        image_model_name: str,
        text_model_name: str,
        embed_dim: int = 256,
        image_pretrained: bool = True,
        text_pretrained: Optional[str] = None,
        freeze_image_backbone: bool = False,
        freeze_text_backbone: bool = False,
        image_input_size: Optional[int] = None,
    ):
        super().__init__()

        import timm
        import open_clip
        from timm.data import resolve_model_data_config

        self.image_model_name = str(image_model_name)
        self.text_model_name = str(text_model_name)
        self.embed_dim = int(embed_dim)
        self.image_pretrained = bool(image_pretrained)
        self.text_pretrained = text_pretrained
        self.freeze_image_backbone = bool(freeze_image_backbone)
        self.freeze_text_backbone = bool(freeze_text_backbone)

        self.image_tower = timm.create_model(
            self.image_model_name,
            pretrained=self.image_pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.image_data_config = resolve_model_data_config(self.image_tower)
        inferred_input_size = int(self.image_data_config.get("input_size", (3, 224, 224))[-1])
        self.image_input_size = int(image_input_size or inferred_input_size)

        image_mean = self.image_data_config.get("mean", (0.485, 0.456, 0.406))
        image_std = self.image_data_config.get("std", (0.229, 0.224, 0.225))
        self.register_buffer("image_mean", torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1))

        self.text_tower = open_clip.create_model(
            self.text_model_name,
            pretrained=self.text_pretrained,
        )
        self._tokenizer = open_clip.get_tokenizer(self.text_model_name)

        self.image_output_dim = self._infer_image_output_dim()
        self.text_output_dim = self._infer_text_output_dim()

        self.image_proj = (
            nn.Identity()
            if self.image_output_dim == self.embed_dim
            else nn.Linear(self.image_output_dim, self.embed_dim, bias=False)
        )
        self.text_proj = (
            nn.Identity()
            if self.text_output_dim == self.embed_dim
            else nn.Linear(self.text_output_dim, self.embed_dim, bias=False)
        )

        if self.freeze_image_backbone:
            for param in self.image_tower.parameters():
                param.requires_grad = False

        if self.freeze_text_backbone:
            for param in self.text_tower.parameters():
                param.requires_grad = False

        self.logit_scale = nn.Parameter(torch.tensor(1.0 / 0.07).log())
        self.logit_bias = nn.Parameter(torch.zeros([]))

    def _infer_image_output_dim(self) -> int:
        was_training = self.image_tower.training
        self.image_tower.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_input_size, self.image_input_size, dtype=torch.float32)
            out = self.image_tower(dummy)
        self.image_tower.train(was_training)
        return int(out.shape[-1])

    def _infer_text_output_dim(self) -> int:
        was_training = self.text_tower.training
        self.text_tower.eval()
        with torch.no_grad():
            dummy = self._tokenizer(["test"])
            out = self.text_tower.encode_text(dummy.long())
        self.text_tower.train(was_training)
        return int(out.shape[-1])

    def _prepare_image_input(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-1] != self.image_input_size or images.shape[-2] != self.image_input_size:
            images = F.interpolate(
                images,
                size=(self.image_input_size, self.image_input_size),
                mode="bicubic",
                align_corners=False,
            )
        return (images - self.image_mean) / self.image_std

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        images = self._prepare_image_input(images.float())
        outputs = self.image_tower(images)
        outputs = self.image_proj(outputs)
        return F.normalize(outputs, dim=-1)

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.text_tower.encode_text(input_ids.long())
        outputs = self.text_proj(outputs)
        return F.normalize(outputs, dim=-1)

    def forward(self, images: torch.Tensor, text_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(images), self.encode_text(text_input)

    def get_tokenizer(self) -> Any:
        return self._tokenizer
