# src/lpcvc_retrieval/distill.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

@dataclass
class TeacherConfig:
    name: str = "ViT-B-32"
    pretrained: str = "openai"
    weight: float = 1.0
    input_size: Optional[int] = None  # Force input size if needed

@dataclass
class DistillConfig:
    use_teacher: bool = False
    # List of teachers for ensemble distillation
    teachers: List[TeacherConfig] = field(default_factory=lambda: [TeacherConfig()])
    
    # Legacy fields support (for backward compatibility if config uses old format)
    teacher_model_name: Optional[str] = None
    teacher_pretrained: Optional[str] = None
    teacher_type: Optional[str] = None
    
    distill_margin_thr: float = 0.2
    affinity_temp: float = 0.1
    affinity_temp_start: Optional[float] = None
    affinity_temp_end: Optional[float] = None
    affinity_temp_schedule: str = "constant"  # constant | linear | cosine
    adaptive_teacher_weight: bool = False
    adaptive_teacher_tau: float = 0.07
    adaptive_teacher_w_min: float = 0.0
    affinity_columns: bool = False
    offline_feature_dir: Optional[str] = None

    def __post_init__(self):
        # Handle legacy config format by converting to list
        if self.teacher_model_name and not self.teachers:
            # If user used old config style, populate teachers list
            self.teachers = [TeacherConfig(
                name=self.teacher_model_name,
                pretrained=self.teacher_pretrained or "openai",
                weight=1.0
            )]
            
        # Ensure teachers are TeacherConfig objects (if passed as dicts from yaml)
        if self.teachers:
            self.teachers = [
                TeacherConfig(**t) if isinstance(t, dict) else t 
                for t in self.teachers
            ]


class OpenClipTeacher(nn.Module):
    """
    Wrapper for open_clip models (SigLIP, MetaCLIP, EVA-CLIP, etc.)
    Handles its own preprocessing (resize/normalize).
    """
    def __init__(self, cfg: TeacherConfig, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.mixed_precision = "fp16" if "cuda" in device else "fp32"
        
        # Use simple concatenation or format to avoid f-string issues if any
        name_str = str(cfg.name)
        pre_str = str(cfg.pretrained)
        print("[Teacher] Loading " + name_str + " (pretrained=" + pre_str + ")...")
        try:
            # Load model and transforms
            # We force device placement here
            model, _, preprocess = open_clip.create_model_and_transforms(
                cfg.name, 
                pretrained=cfg.pretrained,
                device=device
            )
            self.model = model.eval()
            self.preprocess = preprocess
            
            # Extract normalization parameters from transform for manual batch processing
            # open_clip transforms usually are Compose[Resize, CenterCrop, ToTensor, Normalize]
            # We need to apply these manually to batch tensors if they are not already normalized
            # But usually we receive raw images or already standard-normalized images
            # Let's assume we receive raw images in [0,1] or standard ImageNet normalized.
            
            # Actually, robust way is to rely on the model's expected mean/std
            # We will handle resizing internally if student/teacher input sizes differ
            
            # Check model input size
            if cfg.input_size:
                self.input_size = (cfg.input_size, cfg.input_size)
            else:
                # Try to infer from visual config
                if hasattr(self.model.visual, 'image_size'):
                     s = self.model.visual.image_size
                     if isinstance(s, int): s = (s, s)
                     self.input_size = s
                else:
                    self.input_size = (224, 224) 
            
            print(f"[Teacher] {cfg.name} loaded. Input size: {self.input_size}")
            
            # Extract mean/std for normalization
            # 1. Try model.visual.image_mean/std
            mean = getattr(model.visual, 'image_mean', None)
            std = getattr(model.visual, 'image_std', None)
            
            # 2. If not found, look into simple_tokenizer or defaults? No, look at preprocess
            # preprocess is usually Compose[..., Normalize(mean, std)]
            if mean is None or std is None:
                # Fallback to OpenAI CLIP defaults if not found (mostly correct for CLIP variations)
                # But SigLIP uses (0.5, 0.5, 0.5) usually.
                # Let's try to extract from preprocess if it's a Compose
                from torchvision.transforms import Normalize
                if hasattr(preprocess, 'transforms'):
                    for t in preprocess.transforms:
                        if isinstance(t, Normalize):
                            mean = t.mean
                            std = t.std
                            break
            
            if mean is None:
                print(f"[Teacher] Warning: Could not detect mean/std for {cfg.name}, using OpenAI default.")
                mean = (0.48145466, 0.4578275, 0.40821073)
                std = (0.26862954, 0.26130258, 0.27577711)
                
            # Keep normalization buffers on the same device as teacher model.
            # Without this, distillation can fail with CUDA/CPU device mismatch.
            self.register_buffer(
                "image_mean",
                torch.as_tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1),
            )
            self.register_buffer(
                "image_std",
                torch.as_tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1),
            )
            
            # Load tokenizer for this specific teacher model
            self.tokenizer = open_clip.get_tokenizer(cfg.name)
            print("[Teacher] " + str(cfg.name) + " tokenizer loaded.")
            
        except Exception as e:
            raise RuntimeError("Failed to load teacher " + str(cfg.name) + ": " + str(e))

    @torch.no_grad()
    def forward(self, images: torch.Tensor, raw_texts: Optional[List[str]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W) tensor. Raw [0, 1] range.
            raw_texts: List of raw caption strings. Teacher tokenizes itself.
        """
        # Resize images if necessary
        if images.shape[-2:] != self.input_size:
            images = F.interpolate(images, size=self.input_size, mode='bicubic', align_corners=False)
            
        # Normalize
        images = (images - self.image_mean) / self.image_std
            
        # Image
        image_features = self.model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        
        # Text - use own tokenizer
        if raw_texts is not None:
            text_tokens = self.tokenizer(raw_texts).to(images.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        else:
            text_features = None
            
        return image_features, text_features


class EnsembleTeacher(nn.Module):
    """
    Ensemble of multiple OpenClipTeacher models.
    """
    def __init__(self, teachers_cfg: List[TeacherConfig], device: str = "cuda"):
        super().__init__()
        self.teachers = nn.ModuleList()
        self.weights = []
        
        for cfg in teachers_cfg:
            teacher = OpenClipTeacher(cfg, device=device)
            self.teachers.append(teacher)
            self.weights.append(cfg.weight)
            
        # Normalize weights
        total_w = sum(self.weights)
        self.weights = [w / total_w for w in self.weights]
        
    @torch.no_grad()
    def forward(self, images: torch.Tensor, raw_texts: List[str]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for ensemble. Each teacher tokenizes text independently."""
        results = []
        for teacher in self.teachers:
            img, txt = teacher(images, raw_texts)
            results.append((img, txt))
        return results  # Returns list of tuples
    

def create_teacher(cfg: DistillConfig, device: str = "cuda") -> nn.Module:
    """Factory for teacher(s)"""
    if len(cfg.teachers) == 1:
        return OpenClipTeacher(cfg.teachers[0], device=device)
    else:
        return EnsembleTeacher(cfg.teachers, device=device)


# --- Loss Functions ---

def _margin_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # For tiny smoke batches (e.g., B=1), top-2 margin is undefined.
    # Return zeros so selective distillation can safely proceed.
    if logits.ndim != 2 or logits.size(1) < 2:
        return logits.new_zeros((logits.size(0),), dtype=logits.dtype)
    top2 = torch.topk(logits, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def _teacher_quality_margin(t_sim: torch.Tensor, image_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Batch-level teacher quality score:
      margin = mean(sim positives) - mean(sim negatives)
    """
    if image_ids is None:
        return None

    if not torch.is_tensor(image_ids):
        image_ids = torch.as_tensor(image_ids, device=t_sim.device)
    else:
        image_ids = image_ids.to(device=t_sim.device)

    if image_ids.ndim != 1 or image_ids.numel() != t_sim.size(0):
        return None

    pos_mask = image_ids.unsqueeze(0).eq(image_ids.unsqueeze(1))
    neg_mask = ~pos_mask

    pos_vals = t_sim[pos_mask]
    neg_vals = t_sim[neg_mask]
    if pos_vals.numel() == 0 or neg_vals.numel() == 0:
        return None

    return pos_vals.mean() - neg_vals.mean()


def _normalize_static_weights(weights: List[float], device: torch.device) -> torch.Tensor:
    w = torch.tensor([max(0.0, float(x)) for x in weights], device=device, dtype=torch.float32)
    s = w.sum()
    if s <= 0:
        w = torch.ones_like(w)
        s = w.sum()
    return w / s


def _adaptive_weights_from_scores(
    scores: torch.Tensor,
    tau: float = 0.07,
    w_min: float = 0.0,
) -> torch.Tensor:
    """
    Convert teacher quality scores into normalized mixing weights.
    """
    if scores.numel() == 1:
        return torch.ones_like(scores)

    tau = max(float(tau), 1e-6)
    shifted = (scores / tau) - (scores / tau).max()
    w = torch.softmax(shifted, dim=0)

    # Optional floor to avoid single-teacher collapse.
    n = int(w.numel())
    w_min = max(0.0, float(w_min))
    if w_min > 0.0:
        if n * w_min >= 1.0:
            w = torch.full_like(w, 1.0 / n)
        else:
            w = w * (1.0 - n * w_min) + w_min

    return w / w.sum().clamp(min=1e-12)


def affinity_kl_rows(student_logits, teacher_logits, temp=0.1, row_mask=None):
    if row_mask is not None:
        student_logits = student_logits[row_mask]
        teacher_logits = teacher_logits[row_mask]
        if student_logits.numel() == 0: return student_logits.new_tensor(0.0)
    
    p = F.softmax(teacher_logits / temp, dim=1)
    log_q = F.log_softmax(student_logits / temp, dim=1)
    return F.kl_div(log_q, p, reduction="batchmean")

def compute_affinity_distill_loss(
    student_img: torch.Tensor,
    student_txt: torch.Tensor,
    teacher_output: Union[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]],
    teachers_cfg: Optional[List[TeacherConfig]] = None,
    affinity_temp: float = 0.1,
    adaptive_teacher_weight: bool = False,
    adaptive_teacher_tau: float = 0.07,
    adaptive_teacher_w_min: float = 0.0,
    affinity_columns: bool = False,
    distill_margin_thr: float = 0.2,
    selective: bool = True,
    image_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes distillation loss. Handles both Single Teacher and Ensemble Teacher.
    """
    # Student similarity
    s_sim = student_img @ student_txt.t()
    
    # Handle Ensemble
    if isinstance(teacher_output, list):
        if len(teacher_output) == 0:
            return s_sim.new_tensor(0.0)

        # Precompute teacher similarity matrices once (used by loss + adaptive scoring).
        t_sims = [t_img @ t_txt.t() for (t_img, t_txt) in teacher_output]

        if teachers_cfg and len(teachers_cfg) == len(teacher_output):
            weights = [max(0.0, float(cfg.weight)) for cfg in teachers_cfg]
        else:
            weights = [1.0] * len(teacher_output)

        static_weights = _normalize_static_weights(weights, device=s_sim.device)
        adaptive_weights = None
        if adaptive_teacher_weight and len(teacher_output) > 1:
            scores = []
            for t_sim in t_sims:
                score = _teacher_quality_margin(t_sim, image_ids)
                if score is None:
                    score = t_sim.new_tensor(0.0)
                scores.append(score)
            score_tensor = torch.stack(scores, dim=0)
            if torch.isfinite(score_tensor).all():
                adaptive_weights = _adaptive_weights_from_scores(
                    score_tensor,
                    tau=adaptive_teacher_tau,
                    w_min=adaptive_teacher_w_min,
                )

        norm_weights = adaptive_weights if adaptive_weights is not None else static_weights

        row_mask = None
        col_mask = None
        if selective and distill_margin_thr > 0:
            margin = _margin_from_logits(s_sim)
            row_mask = margin < distill_margin_thr
            if affinity_columns:
                margin_t = _margin_from_logits(s_sim.t())
                col_mask = margin_t < distill_margin_thr

        total_loss = s_sim.new_tensor(0.0)
        for weight, t_sim in zip(norm_weights, t_sims):

            loss = affinity_kl_rows(s_sim, t_sim, temp=affinity_temp, row_mask=row_mask)

            if affinity_columns:
                loss += affinity_kl_rows(s_sim.t(), t_sim.t(), temp=affinity_temp, row_mask=col_mask)

            total_loss += weight * loss

        return total_loss

    else:
        # Single Teacher Case
        t_img, t_txt = teacher_output
        t_sim = t_img @ t_txt.t()
        
        row_mask = None
        if selective and distill_margin_thr > 0:
            margin = _margin_from_logits(s_sim)
            row_mask = margin < distill_margin_thr

        loss = affinity_kl_rows(s_sim, t_sim, temp=affinity_temp, row_mask=row_mask)

        if affinity_columns:
            col_mask = None
            if selective and distill_margin_thr > 0:
                margin_t = _margin_from_logits(s_sim.t())
                col_mask = margin_t < distill_margin_thr
            loss += affinity_kl_rows(s_sim.t(), t_sim.t(), temp=affinity_temp, row_mask=col_mask)

        return loss


def get_teacher_output(
    teacher,
    imgs: torch.Tensor,
    metas: list,
    offline_teacher_embs=None,
    device: str = "cuda",
):
    """Unified interface for teacher embeddings (Online forward or Offline pre-extracted)."""
    if offline_teacher_embs is not None:
        teacher_out = [
            (t_img.to(device).float(), t_txt.to(device).float())
            for t_img, t_txt in offline_teacher_embs
        ]
        return teacher_out if len(teacher_out) > 1 else teacher_out[0]

    if teacher is not None:
        imgs_teacher = imgs.float()
        raw_captions = [m['caption'] for m in metas]
        with torch.no_grad():
            return teacher(imgs_teacher, raw_captions)

    return None


