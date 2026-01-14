# src/lpcvc_retrieval/distill.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class DistillConfig:
    use_teacher: bool = False
    teacher_model_name: str = "ViT-B-32"     # logical name
    teacher_pretrained: str = "openai"       # logical name
    distill_margin_thr: float = 0.2          # selective distill threshold
    affinity_temp: float = 0.1
    affinity_columns: bool = False


def _teacher_hf_id(model_name: str, pretrained: str) -> str:
    """
    Map our friendly name to HuggingFace CLIPModel id.
    We only support OpenAI CLIP variants here.
    """
    if pretrained.lower() != "openai":
        raise ValueError(f"Only teacher_pretrained='openai' is supported, got: {pretrained}")

    name = model_name.upper().replace("_", "-")
    if name == "VIT-B-32":
        return "openai/clip-vit-base-patch32"
    if name == "VIT-B-16":
        return "openai/clip-vit-base-patch16"
    # add more if you need
    raise ValueError(f"Unsupported teacher_model_name: {model_name}")


class ClipTeacher(torch.nn.Module):
    """
    Teacher CLIP model wrapper (Transformers).
    Returns normalized image/text embeddings (cosine space).
    """
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cuda"):
        super().__init__()
        from transformers import CLIPModel  # lazy import

        hf_id = _teacher_hf_id(model_name, pretrained)
        self.model = CLIPModel.from_pretrained(hf_id)
        self.model.eval()
        self.model.to(device)

        # teacher dims are usually 512, but we don't need to match student dims for affinity
        self.device = device

    @torch.no_grad()
    def forward(self, images_f32: torch.Tensor, input_ids_i32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        images_f32: (B,3,224,224) float32, expected to be CLIP-normalized already if you want strict CLIP space.
        input_ids_i32: (B,77) int32/long token ids (CLIP tokenizer ids).
        """
        # Transformers expects Long
        input_ids = input_ids_i32.to(dtype=torch.long, device=self.device)
        pixel_values = images_f32.to(device=self.device)

        out = self.model(input_ids=input_ids, pixel_values=pixel_values, return_dict=True)

        img = out.image_embeds
        txt = out.text_embeds

        img = F.normalize(img, dim=-1)
        txt = F.normalize(txt, dim=-1)
        return img, txt


def _margin_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (B,B) similarity matrix (image->text).
    returns: (B,) top1 - top2 margin
    """
    top2 = torch.topk(logits, k=2, dim=1).values  # (B,2)
    return top2[:, 0] - top2[:, 1]


def affinity_kl_rows(
    student_sim: torch.Tensor,
    teacher_sim: torch.Tensor,
    temp: float = 0.1,
    row_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    KL( softmax(teacher/temp) || softmax(student/temp) ) over rows.
    student_sim, teacher_sim: (B,B)
    row_mask: optional boolean mask (B,) selecting which rows to distill.
    """
    if row_mask is not None:
        student_sim = student_sim[row_mask]
        teacher_sim = teacher_sim[row_mask]
        if student_sim.numel() == 0:
            return student_sim.new_tensor(0.0)

    # teacher probs
    p = F.softmax(teacher_sim / temp, dim=1)
    log_q = F.log_softmax(student_sim / temp, dim=1)

    # KLDiv expects input=log_q, target=p
    return F.kl_div(log_q, p, reduction="batchmean")


def compute_affinity_distill_loss(
    student_img: torch.Tensor,
    student_txt: torch.Tensor,
    teacher_img: torch.Tensor,
    teacher_txt: torch.Tensor,
    affinity_temp: float = 0.1,
    affinity_columns: bool = False,
    distill_margin_thr: float = 0.2,
    selective: bool = True,
) -> torch.Tensor:
    """
    Computes Tiny-CLIP style affinity mimicking loss.
    - We compare similarity distributions within a batch, not raw embeddings dims.
    - Optional selective distillation: only rows with low student confidence are distilled.
    """
    # cosine similarity matrices (B,B)
    s_sim = student_img @ student_txt.t()
    t_sim = teacher_img @ teacher_txt.t()

    row_mask = None
    if selective and distill_margin_thr > 0:
        margin = _margin_from_logits(s_sim)
        row_mask = margin < distill_margin_thr

    loss = affinity_kl_rows(s_sim, t_sim, temp=affinity_temp, row_mask=row_mask)

    if affinity_columns:
        # optionally also match columns (text->image)
        # use mask for columns too (same mask logic but from transposed logits)
        col_mask = None
        if selective and distill_margin_thr > 0:
            margin_t = _margin_from_logits(s_sim.t())
            col_mask = margin_t < distill_margin_thr

        loss = loss + affinity_kl_rows(s_sim.t(), t_sim.t(), temp=affinity_temp, row_mask=col_mask)

    return loss
