# src/lpcvc_retrieval/distill.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

# teacher 1개를 표현하는 설정 스키마
@dataclass
class TeacherConfig:
    name: str
    pretrained: str
    input_size: Optional[int] = None  # Force input size if needed

@dataclass
class DistillConfig:
    use_teacher: bool = False
    # List of teachers for ensemble distillation
    teachers: List[TeacherConfig] = field(default_factory=list)
    static_teacher_weights: Optional[List[float]] = None
    
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
    teacher_weight_mode: str = "static"  # static | adaptive | adaptive_source
    source_teacher_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    affinity_columns: bool = False
    offline_feature_dir: Optional[str] = None

    def __post_init__(self):
        # Handle legacy config format by converting to list
        if self.teacher_model_name and not self.teachers:
            # If user used old config style, populate teachers list
            self.teachers = [TeacherConfig(
                name=self.teacher_model_name,
                pretrained=self.teacher_pretrained or "openai",
            )]
            
        # Ensure teachers are TeacherConfig objects (if passed as dicts from yaml)
        if self.teachers:
            normalized_teachers: List[TeacherConfig] = []
            legacy_static_weights: List[float] = []

            for t in self.teachers:
                if isinstance(t, dict):
                    t_cfg = dict(t)
                    legacy_weight = t_cfg.pop("weight", None)
                    normalized_teachers.append(TeacherConfig(**t_cfg))
                    if legacy_weight is not None:
                        legacy_static_weights.append(float(legacy_weight))
                else:
                    normalized_teachers.append(t)

            self.teachers = normalized_teachers
            if self.static_teacher_weights is None and len(legacy_static_weights) == len(self.teachers):
                self.static_teacher_weights = legacy_static_weights

        if self.static_teacher_weights is not None:
            self.static_teacher_weights = [float(x) for x in self.static_teacher_weights]
            if self.teachers and len(self.static_teacher_weights) != len(self.teachers):
                raise ValueError(
                    "distill.static_teacher_weights length must match distill.teachers length."
                )


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
        
        for cfg in teachers_cfg:
            teacher = OpenClipTeacher(cfg, device=device)
            self.teachers.append(teacher)
        
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
    if len(cfg.teachers) == 0:
        raise ValueError("distill.use_teacher=True but no teachers are configured.")
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


def _teacher_quality_margin_per_row(
    t_sim: torch.Tensor,
    image_ids: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """
    Per-sample teacher quality score for retrieval.
    Each row compares the mean positive similarity against the hardest negative.
    """
    if t_sim.ndim != 2 or t_sim.size(0) != t_sim.size(1):
        return None

    if image_ids is None:
        pos_mask = torch.eye(t_sim.size(0), device=t_sim.device, dtype=torch.bool)
    else:
        if not torch.is_tensor(image_ids):
            image_ids = torch.as_tensor(image_ids, device=t_sim.device)
        else:
            image_ids = image_ids.to(device=t_sim.device)

        if image_ids.ndim != 1 or image_ids.numel() != t_sim.size(0):
            return None
        pos_mask = image_ids.unsqueeze(0).eq(image_ids.unsqueeze(1))
    neg_mask = ~pos_mask
    if not pos_mask.any() or not neg_mask.any():
        return None

    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    pos_mean = (t_sim * pos_mask.to(dtype=t_sim.dtype)).sum(dim=1) / pos_count.to(dtype=t_sim.dtype)

    neg_filled = t_sim.masked_fill(~neg_mask, float("-inf"))
    hardest_neg = neg_filled.max(dim=1).values
    no_neg = ~neg_mask.any(dim=1)
    hardest_neg = torch.where(no_neg, pos_mean, hardest_neg)
    return pos_mean - hardest_neg


def _normalize_static_weights(weights: List[float], device: torch.device) -> torch.Tensor:
    w = torch.tensor([max(0.0, float(x)) for x in weights], device=device, dtype=torch.float32)
    s = w.sum()
    if s <= 0:
        w = torch.ones_like(w)
        s = w.sum()
    return w / s


def _uniform_teacher_weights(num_teachers: int, device: torch.device) -> torch.Tensor:
    num_teachers = max(1, int(num_teachers))
    return torch.full((num_teachers,), 1.0 / float(num_teachers), device=device, dtype=torch.float32)


def _resolve_source_prior_weights(
    source: str,
    teacher_names: List[str],
    source_teacher_weights: Optional[Dict[str, Dict[str, float]]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not source_teacher_weights:
        return None
    source_cfg = source_teacher_weights.get(source)
    if not isinstance(source_cfg, dict) or len(source_cfg) == 0:
        return None
    weights = [float(source_cfg.get(name, 0.0)) for name in teacher_names]
    return _normalize_static_weights(weights, device=device)


def _combine_teacher_weights(
    prior_weights: Optional[torch.Tensor],
    adaptive_weights: Optional[torch.Tensor],
) -> torch.Tensor:
    if adaptive_weights is None and prior_weights is None:
        raise ValueError("At least one teacher weight source must be provided.")
    if adaptive_weights is None:
        if prior_weights.ndim == 1:
            return prior_weights / prior_weights.sum().clamp(min=1e-12)
        return _normalize_weight_matrix(prior_weights)
    if prior_weights is None:
        if adaptive_weights.ndim == 1:
            return adaptive_weights / adaptive_weights.sum().clamp(min=1e-12)
        return _normalize_weight_matrix(adaptive_weights)
    if prior_weights.ndim == 1:
        prior_weights = prior_weights.unsqueeze(1).expand_as(adaptive_weights)
    if adaptive_weights.ndim == 1:
        adaptive_weights = adaptive_weights.unsqueeze(1).expand_as(prior_weights)
    mixed = prior_weights * adaptive_weights
    return _normalize_weight_matrix(mixed)


def _normalize_weight_matrix(weights: torch.Tensor) -> torch.Tensor:
    return weights / weights.sum(dim=0, keepdim=True).clamp(min=1e-12)


def _adaptive_weights_from_scores(
    scores: torch.Tensor,
    tau: float = 0.07,
    w_min: float = 0.0,
) -> torch.Tensor:
    """
    Convert teacher quality scores into normalized mixing weights.
    scores shape: [num_teachers, num_rows]
    """
    if scores.ndim == 1:
        scores = scores.unsqueeze(1)

    if scores.size(0) == 1:
        return torch.ones_like(scores)

    tau = max(float(tau), 1e-6)
    shifted = (scores / tau) - (scores / tau).max(dim=0, keepdim=True).values
    w = torch.softmax(shifted, dim=0)

    # Optional floor to avoid single-teacher collapse.
    n = int(w.numel())
    w_min = max(0.0, float(w_min))
    if w_min > 0.0:
        num_teachers = int(w.size(0))
        if num_teachers * w_min >= 1.0:
            w = torch.full_like(w, 1.0 / num_teachers)
        else:
            w = w * (1.0 - num_teachers * w_min) + w_min

    return _normalize_weight_matrix(w)


def affinity_kl_rows(student_logits, teacher_logits, temp=0.1, row_mask=None):
    if row_mask is not None:
        student_logits = student_logits[row_mask]
        teacher_logits = teacher_logits[row_mask]
        if student_logits.numel() == 0: return student_logits.new_tensor(0.0)
    
    p = F.softmax(teacher_logits / temp, dim=1)
    log_q = F.log_softmax(student_logits / temp, dim=1)
    return F.kl_div(log_q, p, reduction="batchmean")


def affinity_kl_per_row(student_logits, teacher_logits, temp=0.1, row_mask=None):
    if row_mask is not None:
        student_logits = student_logits[row_mask]
        teacher_logits = teacher_logits[row_mask]
        if student_logits.numel() == 0:
            return student_logits.new_zeros((0,), dtype=student_logits.dtype)

    p = F.softmax(teacher_logits / temp, dim=1)
    log_q = F.log_softmax(student_logits / temp, dim=1)
    return F.kl_div(log_q, p, reduction="none").sum(dim=1)


def _build_prior_weight_matrix(
    static_weights: torch.Tensor,
    teacher_names: List[str],
    teacher_weight_mode: str,
    source_teacher_weights: Optional[Dict[str, Dict[str, float]]],
    sample_sources: Optional[List[str]],
) -> torch.Tensor:
    uniform_weights = _uniform_teacher_weights(len(teacher_names), static_weights.device)
    num_rows = len(sample_sources) if sample_sources is not None else 0
    if teacher_weight_mode == "static":
        return static_weights.unsqueeze(1)
    if teacher_weight_mode == "adaptive":
        return uniform_weights.unsqueeze(1)
    if num_rows <= 0:
        return uniform_weights.unsqueeze(1)

    priors = []
    for src in sample_sources:
        source_prior = _resolve_source_prior_weights(
            source=str(src or "unknown"),
            teacher_names=teacher_names,
            source_teacher_weights=source_teacher_weights,
            device=static_weights.device,
        )
        priors.append(source_prior if source_prior is not None else uniform_weights)
    return torch.stack(priors, dim=1)


def _weighted_row_loss(
    teacher_weights: torch.Tensor,
    teacher_losses: torch.Tensor,
    row_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if row_mask is not None:
        teacher_weights = teacher_weights[:, row_mask]
        teacher_losses = teacher_losses[:, row_mask]
    if teacher_losses.numel() == 0 or teacher_losses.size(1) == 0:
        return teacher_losses.new_tensor(0.0)
    blended = (teacher_weights * teacher_losses).sum(dim=0)
    return blended.mean()

def compute_affinity_distill_loss(
    student_img: torch.Tensor,
    student_txt: torch.Tensor,
    teacher_output: Union[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]],
    teachers_cfg: Optional[List[TeacherConfig]] = None,
    static_teacher_weights: Optional[List[float]] = None,
    affinity_temp: float = 0.1,
    adaptive_teacher_weight: bool = False,
    adaptive_teacher_tau: float = 0.07,
    adaptive_teacher_w_min: float = 0.0,
    teacher_weight_mode: str = "static",
    source_teacher_weights: Optional[Dict[str, Dict[str, float]]] = None,
    affinity_columns: bool = False,
    distill_margin_thr: float = 0.2,
    selective: bool = True,
    image_ids: Optional[torch.Tensor] = None,
    sample_sources: Optional[List[str]] = None,
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
            teacher_names = [str(cfg.name) for cfg in teachers_cfg]
        else:
            teacher_names = [f"teacher_{idx}" for idx in range(len(teacher_output))]

        if static_teacher_weights is not None and len(static_teacher_weights) == len(teacher_output):
            weights = static_teacher_weights
        else:
            weights = [1.0] * len(teacher_output)
        static_weights = _normalize_static_weights(weights, device=s_sim.device)
        teacher_weight_mode = str(teacher_weight_mode or "static").lower()
        if teacher_weight_mode not in {"static", "adaptive", "adaptive_source"}:
            teacher_weight_mode = "static"
        if adaptive_teacher_weight and teacher_weight_mode == "static":
            teacher_weight_mode = "adaptive_source" if (source_teacher_weights and sample_sources) else "adaptive"

        row_mask = None
        col_mask = None
        if selective and distill_margin_thr > 0:
            margin = _margin_from_logits(s_sim)
            row_mask = margin < distill_margin_thr
            if affinity_columns:
                margin_t = _margin_from_logits(s_sim.t())
                col_mask = margin_t < distill_margin_thr
        num_rows = s_sim.size(0)
        if sample_sources is None or len(sample_sources) != num_rows:
            sample_sources = ["unknown"] * num_rows

        prior_weight_matrix = _build_prior_weight_matrix(
            static_weights=static_weights,
            teacher_names=teacher_names,
            teacher_weight_mode=teacher_weight_mode,
            source_teacher_weights=source_teacher_weights,
            sample_sources=sample_sources,
        )
        if prior_weight_matrix.size(1) == 1 and num_rows > 1:
            prior_weight_matrix = prior_weight_matrix.expand(-1, num_rows)

        row_teacher_weights = prior_weight_matrix
        col_teacher_weights = prior_weight_matrix
        if adaptive_teacher_weight and len(teacher_output) > 1:
            row_scores = []
            col_scores = []
            for t_sim in t_sims:
                row_score = _teacher_quality_margin_per_row(t_sim, image_ids)
                col_score = _teacher_quality_margin_per_row(t_sim.t(), image_ids) if affinity_columns else None
                if row_score is None:
                    row_score = t_sim.new_zeros((num_rows,), dtype=t_sim.dtype)
                if col_score is None and affinity_columns:
                    col_score = t_sim.new_zeros((num_rows,), dtype=t_sim.dtype)
                row_scores.append(torch.nan_to_num(row_score, nan=0.0, posinf=0.0, neginf=0.0))
                if affinity_columns:
                    col_scores.append(torch.nan_to_num(col_score, nan=0.0, posinf=0.0, neginf=0.0))

            row_adaptive_weights = _adaptive_weights_from_scores(
                torch.stack(row_scores, dim=0),
                tau=adaptive_teacher_tau,
                w_min=adaptive_teacher_w_min,
            )
            row_teacher_weights = _combine_teacher_weights(prior_weight_matrix, row_adaptive_weights)

            if affinity_columns:
                col_adaptive_weights = _adaptive_weights_from_scores(
                    torch.stack(col_scores, dim=0),
                    tau=adaptive_teacher_tau,
                    w_min=adaptive_teacher_w_min,
                )
                col_teacher_weights = _combine_teacher_weights(prior_weight_matrix, col_adaptive_weights)

        row_losses = torch.stack(
            [affinity_kl_per_row(s_sim, t_sim, temp=affinity_temp) for t_sim in t_sims],
            dim=0,
        )
        total_loss = _weighted_row_loss(row_teacher_weights, row_losses, row_mask=row_mask)

        if affinity_columns:
            col_losses = torch.stack(
                [affinity_kl_per_row(s_sim.t(), t_sim.t(), temp=affinity_temp) for t_sim in t_sims],
                dim=0,
            )
            total_loss = total_loss + _weighted_row_loss(col_teacher_weights, col_losses, row_mask=col_mask)

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

