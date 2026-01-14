from __future__ import annotations
import os, random, math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from .config import resolve_device
from .data import build_tokenizer, make_datasets, collate_fn
from .distill import DistillConfig, ClipTeacher, compute_affinity_distill_loss
from .model import ClipLite
from .losses import clip_contrastive_loss, pairwise_ranking_loss
from .metrics import recall_at_1_5_10, bidirectional_recall, format_metrics

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float = 0.0):
    """
    Cosine Annealing LR Scheduler with Linear Warmup.
    
    Why this helps:
    - Warmup: Prevents unstable gradients at the start when model weights are random.
              Learning rate starts from 0 and linearly increases to base_lr.
    - Cosine Decay: Gradually reduces learning rate following a cosine curve,
                    allowing fine-grained optimization near the end of training.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps (linear increase from 0 to base_lr)
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum LR as ratio of base LR at the end (default: 0.0)
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step + 1) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)

@torch.no_grad()
def evaluate(model: ClipLite, loader: DataLoader, device: str, use_bidirectional: bool = True):
    """
    Evaluate model on retrieval metrics.
    
    Args:
        model: ClipLite model
        loader: Validation data loader
        device: Device to run on
        use_bidirectional: If True, compute both I2T and T2I metrics
    """
    model.eval()
    all_img, all_txt = [], []
    for imgs, toks, _ in tqdm(loader, desc="eval", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        toks = toks.to(device, non_blocking=True)
        img_emb, txt_emb = model(imgs, toks)
        all_img.append(img_emb.cpu())
        all_txt.append(txt_emb.cpu())
    image_emb = torch.cat(all_img, dim=0)
    text_emb = torch.cat(all_txt, dim=0)
    
    if use_bidirectional:
        return bidirectional_recall(image_emb, text_emb)
    else:
        r1, r5, r10 = recall_at_1_5_10(image_emb, text_emb)
        return {'I2T': {'R@1': r1, 'R@5': r5, 'R@10': r10}}

def _normalize_clip(img: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    """Normalize image tensor to CLIP space (for teacher model)."""
    mean_t = img.new_tensor(mean).view(1, 3, 1, 1)
    std_t = img.new_tensor(std).view(1, 3, 1, 1)
    return (img - mean_t) / std_t

def train(cfg) -> str:
    device = resolve_device(cfg.get("device", "auto"))
    set_seed(int(cfg.get("seed", 42)))

    out_dir = str(cfg.output.get("out_dir", "runs/lpcvc_clip_lite"))
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = build_tokenizer()
    vocab_size = int(tokenizer.vocab_size)
    eos_id = int(tokenizer.eos_token_id)

    # ---- model ----
    clip_mean = list(cfg.model.get("clip_mean", [0.48145466, 0.4578275, 0.40821073]))
    clip_std  = list(cfg.model.get("clip_std",  [0.26862954, 0.26130258, 0.27577711]))

    model = ClipLite(
        vision_backbone=str(cfg.model.vision_backbone),
        vision_pretrained=bool(cfg.model.get("vision_pretrained", True)),
        normalize_input=bool(cfg.model.get("normalize_input", True)),
        clip_mean=clip_mean,
        clip_std=clip_std,
        embed_dim=int(cfg.model.get("embed_dim", 256)),
        vocab_size=vocab_size,
        context_length=77,
        text_width=int(cfg.model.get("text_width", 256)),
        text_layers=int(cfg.model.get("text_layers", 4)),
        text_heads=int(cfg.model.get("text_heads", 4)),
        text_mlp_ratio=float(cfg.model.get("text_mlp_ratio", 4.0)),
        dropout=float(cfg.model.get("dropout", 0.0)),
        temperature_init=float(cfg.model.get("temperature_init", 0.07)),
        eos_id=eos_id,
    ).to(device)

    # ---- data ----
    train_ds, val_ds = make_datasets(cfg, tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.data.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(cfg.data.get("num_workers", 4)),
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.data.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(cfg.data.get("num_workers", 4)),
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )

    # ---- training hparams ----
    lr = float(cfg.train.get("lr", 2e-4))
    wd = float(cfg.train.get("weight_decay", 0.02))
    epochs = int(cfg.train.get("epochs", 1))
    warmup_epochs = float(cfg.train.get("warmup_epochs", 1.0))
    grad_clip = float(cfg.train.get("grad_clip", 1.0))

    # ---- loss hparams ----
    label_smoothing = float(cfg.loss.get("label_smoothing", 0.0))
    w_contrastive = float(cfg.loss.get("w_contrastive", 1.0))
    w_rank = float(cfg.loss.get("w_rank", 0.0))
    rank_k = int(cfg.loss.get("rank_k", 3))
    rank_margin = float(cfg.loss.get("rank_margin", 0.1))

    # distill loss weight (NEW: actually used)
    w_distill_affinity = float(cfg.loss.get("w_distill_affinity", 0.0))

    # ---- logit scale clamp ----
    logit_scale_min = float(cfg.model.get("logit_scale_min", -4.6))
    logit_scale_max = float(cfg.model.get("logit_scale_max", 4.6))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(len(train_loader) * warmup_epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # NOTE: Do NOT manually set lr=0 here. The scheduler's lr_lambda handles warmup
    # by returning small multipliers at the start. Setting lr=0 manually causes
    # PyTorch to skip the first LR value when scheduler.step() is called.

    # AMP (updated API: torch.amp.* to avoid deprecation warnings)
    use_amp = bool(cfg.train.get("amp", True)) and device == "cuda"
    autocast_device = "cuda" if device == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(autocast_device, enabled=use_amp)

    # ---- (2) teacher 준비: epoch loop 들어가기 전에 딱 1번만! ----
    distill_section = cfg.get("distill", {})  # 없으면 {}
    if hasattr(distill_section, "as_dict"):
        distill_section = distill_section.as_dict()
    distill_cfg = DistillConfig(**distill_section) if isinstance(distill_section, dict) else DistillConfig()

    teacher = None
    if distill_cfg.use_teacher and w_distill_affinity > 0:
        if device != "cuda":
            print("[Warn] distill.use_teacher=True but device is not cuda. Teacher may be slow on CPU.")
        teacher = ClipTeacher(
            model_name=distill_cfg.teacher_model_name,
            pretrained=distill_cfg.teacher_pretrained,
            device=str(device),
        )
        print(f"[Distill] Teacher ON: {distill_cfg.teacher_model_name} ({distill_cfg.teacher_pretrained}), "
              f"w_affinity={w_distill_affinity}, thr={distill_cfg.distill_margin_thr}, temp={distill_cfg.affinity_temp}, "
              f"columns={distill_cfg.affinity_columns}")
    else:
        print("[Distill] Teacher OFF")

    best_r10 = -1.0
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")

    log_every = int(cfg.train.get("log_every", 20))
    eval_every = int(cfg.train.get("eval_every_epochs", 1))

    print(f"[Config] epochs={epochs}, warmup={warmup_epochs}, lr={lr}, grad_clip={grad_clip}")
    print(f"[Config] w_contrastive={w_contrastive}, w_rank={w_rank}, w_distill_affinity={w_distill_affinity}, "
          f"label_smoothing={label_smoothing}")
    print(f"[Config] logit_scale_range=[{logit_scale_min}, {logit_scale_max}]")

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}")

        for step, (imgs, toks, _) in enumerate(pbar, start=1):
            imgs = imgs.to(device, non_blocking=True)
            toks = toks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            distill_loss_val = imgs.new_tensor(0.0)

            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                img_emb, txt_emb = model(imgs, toks)

                # base contrastive
                loss = w_contrastive * clip_contrastive_loss(
                    img_emb, txt_emb, model.logit_scale,
                    label_smoothing=label_smoothing
                )

                # ranking (optional)
                if w_rank > 0:
                    loss = loss + w_rank * pairwise_ranking_loss(
                        img_emb, txt_emb, model.logit_scale,
                        k=rank_k, margin=rank_margin,
                    )

            # ---- distill (outside autocast for teacher stability) ----
            if teacher is not None:
                # teacher expects CLIP-normalized pixel_values ideally
                imgs_fp32 = imgs.float()
                if bool(cfg.model.get("normalize_input", True)):
                    imgs_teacher = _normalize_clip(imgs_fp32, clip_mean, clip_std)
                else:
                    imgs_teacher = imgs_fp32

                with torch.no_grad():
                    t_img, t_txt = teacher(imgs_teacher, toks)

                # student normalize for cosine space (safe even if model already normalizes)
                s_img = F.normalize(img_emb.float(), dim=-1)
                s_txt = F.normalize(txt_emb.float(), dim=-1)

                distill_loss_val = compute_affinity_distill_loss(
                    student_img=s_img,
                    student_txt=s_txt,
                    teacher_img=t_img,
                    teacher_txt=t_txt,
                    affinity_temp=distill_cfg.affinity_temp,
                    affinity_columns=distill_cfg.affinity_columns,
                    distill_margin_thr=distill_cfg.distill_margin_thr,
                    selective=True,
                )

                loss = loss + (w_distill_affinity * distill_loss_val)

            # backward
            scaler.scale(loss).backward()

            # gradient clipping
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            # optimizer step
            scaler.step(optimizer)
            scaler.update()

            # scheduler step AFTER optimizer step
            scheduler.step()

            # logit scale clamp
            with torch.no_grad():
                model.logit_scale.data.clamp_(logit_scale_min, logit_scale_max)

            if step % log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                postfix = {
                    "loss": f"{loss.item():.4f}",
                    "scale": f"{model.logit_scale.exp().item():.2f}",
                    "lr": f"{current_lr:.2e}",
                }
                if teacher is not None:
                    postfix["distill"] = f"{distill_loss_val.item():.4f}"
                pbar.set_postfix(postfix)

        # save checkpoints
        ckpt = {"model": model.state_dict(), "config": cfg.as_dict(), "epoch": epoch + 1}
        torch.save(ckpt, last_path)
        
        # save epoch-specific checkpoint
        epoch_path = os.path.join(out_dir, f"epoch_{epoch + 1}.pt")
        torch.save(ckpt, epoch_path)

        if (epoch + 1) % eval_every == 0:
            metrics = evaluate(model, val_loader, device, use_bidirectional=True)
            r10 = metrics["I2T"]["R@10"]
            print(f"[epoch {epoch+1}] {format_metrics(metrics)}")

            if r10 > best_r10:
                best_r10 = r10
                torch.save(ckpt, best_path)
                print(f"  -> New best model saved! R@10={r10*100:.2f}%")

    return best_path if os.path.exists(best_path) else last_path
