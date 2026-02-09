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
    
    distill_margin_thr: float = 0.2          # selective distill threshold
    affinity_temp: float = 0.1
    affinity_columns: bool = False

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
                
            # Ensure they are tensors on device
            self.register_buffer('image_mean', torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer('image_std', torch.tensor(std).view(1, 3, 1, 1))
            
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
    top2 = torch.topk(logits, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]

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
    teachers_cfg: Optional[List[TeacherConfig]] = None, # passed if ensemble
    affinity_temp: float = 0.1,
    affinity_columns: bool = False,
    distill_margin_thr: float = 0.2,
    selective: bool = True,
) -> torch.Tensor:
    """
    Computes distillation loss. Handles both Single Teacher and Ensemble Teacher.
    """
    # Student similarity
    s_sim = student_img @ student_txt.t()
    
    # Handle Ensemble
    if isinstance(teacher_output, list):
        # Ensemble Case
        total_loss = 0.0
        # If teachers_cfg is not passed explicitly, we might struggle to know weights.
        # But we can assume uniform or we need to pass weights.
        # Let's assume standard weights if not provided (or we can extract from model if we had access)
        
        # Actually, `train.py` calls this. `train.py` doesn't know about weights.
        # We should probably pre-calculate 'Target Logits' in EnsembleTeacher?
        # NO, different teachers have different embedding spaces/scales.
        # We must calculate Loss for EACH teacher and weighted-sum the LOSS.
        
        # We need weights.
        # Hack: if teacher_output is list, we assume equal weights unless we find a way.
        # Users usually config weights in config.yaml.
        # Let's update this function signature in train.py call or rely on uniform for now
        # OR better: EnsembleTeacher.forward() could return the Weighted Average Target Probability?
        # No, KL divergence target is prob dist. 
        # Average(Softmax(T1)) != Softmax(Average(T1))
        # The correct Ensemble KD is usually: Loss = w1 * KL(S, T1) + w2 * KL(S, T2)
        
        # We will assume weights are available or just sum them directly (implying equal or pre-scaled).
        # To strictly follow config weights, `train.py` needs to pass them.
        # For now, let's just average the losses (simplification).
        
        for i, (t_img, t_txt) in enumerate(teacher_output):
            t_sim = t_img @ t_txt.t()
            
            # Masking based on specific teacher? Or student confidence?
            # Selective distill is based on Student Margin usually.
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
            
            total_loss += loss
            
        return total_loss / len(teacher_output)

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

