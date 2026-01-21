from __future__ import annotations
import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_clip_norm(mean: List[float], std: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    m = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return m, s

class VisionTower(nn.Module):
    def __init__(self, backbone: str, embed_dim: int, pretrained: bool, normalize_input: bool, clip_mean: List[float], clip_std: List[float]):
        super().__init__()
        self.normalize_input = bool(normalize_input)
        mean, std = _make_clip_norm(clip_mean, clip_std)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

        import timm
        self.backbone = timm.create_model(backbone, pretrained=bool(pretrained), num_classes=0, global_pool="avg")
        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:
            # try infer by running dummy
            with torch.no_grad():
                x = torch.zeros(1,3,224,224)
                y = self.backbone(x)
                feat_dim = y.shape[-1]
        self.proj = nn.Linear(int(feat_dim), int(embed_dim), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,224,224], float32 in [0,1]
        if self.normalize_input:
            x = (x - self.mean) / self.std
        feat = self.backbone(x)  # [B,C]
        emb = self.proj(feat)
        return emb

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B,T,C]
        B, T, C = x.shape
        qkv = self.qkv(x)  # [B,T,3C]
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)  # [3,B,H,T,D]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,T,D]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,H,T,T]
        if attn_mask is not None:
            # attn_mask: [B,1,1,T] where 0 for keep, -inf for pad
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = attn @ v  # [B,H,T,D]
        out = out.transpose(1,2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.drop(out)
        return out

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class TextTowerTinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, width: int, layers: int, heads: int, mlp_ratio: float, dropout: float, eos_id: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.context_length = int(context_length)
        self.width = int(width)
        self.eos_id = int(eos_id)

        self.token_embedding = nn.Embedding(self.vocab_size, self.width)
        self.pos_embedding = nn.Parameter(torch.zeros(self.context_length, self.width))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(self.width, heads, mlp_ratio, dropout) for _ in range(layers)])
        self.ln_final = nn.LayerNorm(self.width)

        nn.init.normal_(self.pos_embedding, std=0.01)

    def forward(self, input_ids_int32: torch.Tensor) -> torch.Tensor:
        # input_ids: [B,77] int32
        input_ids = input_ids_int32.to(torch.int64)
        x = self.token_embedding(input_ids)  # [B,T,C]
        x = x + self.pos_embedding.unsqueeze(0)
        x = self.drop(x)

        # build attention mask to ignore padding(0): mask adds -inf to pad positions in keys
        pad = (input_ids == 0)  # [B,T]
        # attn_mask for broadcasting: [B,1,1,T], 0 for keep, -inf for pad
        attn_mask = pad.unsqueeze(1).unsqueeze(2).to(x.dtype) * (-1e4)

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        x = self.ln_final(x)

        # pick EOS token position
        eos_mask = (input_ids == self.eos_id).to(torch.int64)  # [B,T]
        # argmax gives first eos if exists (since mask is 0/1). If none, returns 0; but CLIPTokenizer should add eos.
        eos_idx = torch.argmax(eos_mask, dim=1)  # [B]
        out = x[torch.arange(x.size(0), device=x.device), eos_idx]  # [B,C]
        return out

class ClipLite(nn.Module):
    def __init__(self,
                 vision_backbone: str,
                 vision_pretrained: bool,
                 normalize_input: bool,
                 clip_mean: List[float],
                 clip_std: List[float],
                 embed_dim: int,
                 vocab_size: int,
                 context_length: int,
                 text_width: int,
                 text_layers: int,
                 text_heads: int,
                 text_mlp_ratio: float,
                 dropout: float,
                 temperature_init: float,
                 eos_id: int):
        super().__init__()
        
        # 1. Vision/Text 타워 설정 (기존 유지)
        self.vision = VisionTower(
            backbone=vision_backbone,
            embed_dim=embed_dim,
            pretrained=vision_pretrained,
            normalize_input=normalize_input,
            clip_mean=clip_mean,
            clip_std=clip_std,
        )
        self.text = TextTowerTinyTransformer(
            vocab_size=vocab_size,
            context_length=context_length,
            width=text_width,
            layers=text_layers,
            heads=text_heads,
            mlp_ratio=text_mlp_ratio,
            dropout=dropout,
            eos_id=eos_id,
        )
        self.text_proj = nn.Linear(text_width, embed_dim, bias=False)

        # 2. 파라미터 설정 (수정)
        init = float(temperature_init)
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / init).log())
        
        # [추가] SigLIP을 위한 학습 가능한 Bias 파라미터
        self.logit_bias = nn.Parameter(torch.zeros([]))

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        z = self.vision(x)
        return F.normalize(z, dim=-1)

    def encode_text(self, toks: torch.Tensor) -> torch.Tensor:
        z = self.text(toks)
        z = self.text_proj(z)
        return F.normalize(z, dim=-1)

    def forward(self, images: torch.Tensor, text_input: torch.Tensor):
        img = self.encode_image(images)
        txt = self.encode_text(text_input)
        return img, txt

class OnnxWrapper(nn.Module):
    def __init__(self, model: ClipLite):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, text_input: torch.Tensor):
        img, txt = self.model(image, text_input)
        return img, txt
