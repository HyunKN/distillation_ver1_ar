from __future__ import annotations
import os, random, math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from .config import resolve_device
from .data import build_tokenizer, make_datasets, collate_fn
from .distill import DistillConfig, compute_affinity_distill_loss
from .mobileclip2 import MobileCLIP2Student  # Apple MobileCLIP2 Student Model
# Loss 함수 import
from .losses import (
    pairwise_ranking_loss, 
    siglip_loss, 
    hard_negative_contrastive_loss,
    text_text_contrastive_loss,
)
from .metrics import recall_at_1_5_10, bidirectional_recall, format_metrics, coco_bidirectional_recall
from .logger import TrainLogger  # [OPTIONAL] WandB - 제거 시 이 줄 삭제
from .ema import EMA  # [NEW] EMA for training stabilization

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
def evaluate(model: MobileCLIP2Student, loader: DataLoader, device: str, use_bidirectional: bool = True, use_coco_eval: bool = True):
    """
    Evaluate model on retrieval metrics.
    
    Args:
        model: MobileCLIP2 student model
        loader: Validation data loader
        device: Device to run on
        use_bidirectional: If True, compute both I2T and T2I metrics
        use_coco_eval: If True, use COCO-style image_id-based evaluation (handles one-to-many)
    """
    model.eval()
    all_img, all_txt = [], []
    all_image_ids = []
    
    for imgs, toks, metas in tqdm(loader, desc="eval", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        toks = toks.to(device, non_blocking=True)
        img_emb, txt_emb = model(imgs, toks)
        all_img.append(img_emb.cpu())
        all_txt.append(txt_emb.cpu())
        
        # Collect image_ids from meta
        for meta in metas:
            if isinstance(meta, dict) and meta.get("image_id") is not None:
                all_image_ids.append(meta["image_id"])
            else:
                all_image_ids.append(None)
    
    image_emb = torch.cat(all_img, dim=0)
    text_emb = torch.cat(all_txt, dim=0)
    
    # Check if we have valid image_ids for COCO-style evaluation
    has_image_ids = all(img_id is not None for img_id in all_image_ids)
    
    if use_coco_eval and has_image_ids and use_bidirectional:
        # COCO-style evaluation with image deduplication
        # Deduplicate images by image_id (same image appears multiple times for different captions)
        seen_image_ids = {}
        unique_image_embs = []
        unique_image_ids = []
        
        for idx, img_id in enumerate(all_image_ids):
            if img_id not in seen_image_ids:
                seen_image_ids[img_id] = len(unique_image_ids)
                unique_image_embs.append(image_emb[idx])
                unique_image_ids.append(img_id)
        
        unique_image_emb = torch.stack(unique_image_embs, dim=0)
        text_image_ids = all_image_ids  # Each text's corresponding image_id
        
        print(f"[COCO Eval] {len(unique_image_ids)} unique images, {len(text_image_ids)} captions")
        
        return coco_bidirectional_recall(
            unique_image_emb, text_emb, unique_image_ids, text_image_ids
        )
    else:
        # Fallback to index-based evaluation
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


def _scheduled_distill_temp(
    epoch_idx: int,
    total_epochs: int,
    start: float,
    end: float,
    schedule: str,
) -> float:
    """Return epoch-wise distillation temperature."""
    schedule = str(schedule).lower()
    if total_epochs <= 1 or abs(start - end) < 1e-12 or schedule == "constant":
        return float(start)

    progress = float(epoch_idx) / float(max(1, total_epochs - 1))  # 0.0 -> 1.0
    if schedule == "linear":
        alpha = progress
    elif schedule == "cosine":
        alpha = 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        # Fallback to constant if an unknown schedule is provided.
        alpha = 0.0

    return float(start + (end - start) * alpha)

def train(cfg) -> str:
    device = resolve_device(cfg.get("device", "auto"))
    set_seed(int(cfg.get("seed", 42)))

    out_dir = str(cfg.output.get("out_dir", "runs/lpcvc_clip_lite"))
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = build_tokenizer()
    vocab_size = int(tokenizer.vocab_size)
    eos_id = int(tokenizer.eos_token_id)

    # ---- model: MobileCLIP2 Student ----
    embed_dim = int(cfg.model.get("embed_dim", 256))
    variant = str(cfg.model.get("mobileclip2_variant", "S4"))
    freeze_backbone = bool(cfg.model.get("freeze_backbone", False))
    checkpoint_path = cfg.model.get("checkpoint_path", None)
    
    model = MobileCLIP2Student(
        variant=variant,
        embed_dim=embed_dim,
        freeze_backbone=freeze_backbone,
        checkpoint_path=checkpoint_path,
    ).to(device)
    print(f"[Model] MobileCLIP2-{variant} (embed_dim={embed_dim}, freeze={freeze_backbone})")

    # ---- [OPTIONAL] torch.compile (PyTorch 2.x 학습 가속) ----
    use_compile = bool(cfg.train.get("use_compile", False))
    if use_compile and hasattr(torch, "compile"):
        print("[torch.compile] Compiling model...")
        model = torch.compile(model)
        print("[torch.compile] Done!")

    # ---- [OPTIONAL] WandB Logger ----
    use_wandb = bool(cfg.train.get("use_wandb", False))
    logger = TrainLogger(
        use_wandb=use_wandb,
        project=str(cfg.train.get("wandb_project", "lpcvc-clip-lite")),
        run_name=cfg.train.get("wandb_run_name"),
        config=cfg.as_dict() if hasattr(cfg, "as_dict") else dict(cfg),
    )

    # ---- data ----
    train_ds, val_ds = make_datasets(cfg, tokenizer)
    num_workers = int(cfg.data.get("num_workers", 4))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.data.get("batch_size", 64)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.data.get("batch_size", 64)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
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

    # [NEW] Paper-based loss weights (BLIP, FG-CLIP, TULIP)
    w_hard_negative = float(cfg.loss.get("w_hard_negative", 0.0))
    w_text_text = float(cfg.loss.get("w_text_text", 0.0))
    hard_negative_k = int(cfg.loss.get("hard_negative_k", 5))

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
    
    # Helper to clean up OmegaConf/yacs objects
    def _to_clean_dict(obj):
        if hasattr(obj, 'as_dict'): return _to_clean_dict(obj.as_dict())
        if isinstance(obj, list): return [_to_clean_dict(x) for x in obj]
        if isinstance(obj, dict): return {k: _to_clean_dict(v) for k,v in obj.items()}
        return obj

    distill_section = _to_clean_dict(distill_section)
    from .distill import DistillConfig, create_teacher, get_teacher_output
    distill_cfg = DistillConfig(**distill_section) if isinstance(distill_section, dict) else DistillConfig()
    distill_temp_schedule = str(getattr(distill_cfg, "affinity_temp_schedule", "constant") or "constant").lower()
    if distill_temp_schedule not in {"constant", "linear", "cosine"}:
        print(f"[Warn] Unknown affinity_temp_schedule='{distill_temp_schedule}'. Fallback to 'constant'.")
        distill_temp_schedule = "constant"
    distill_temp_start = float(
        distill_cfg.affinity_temp
        if getattr(distill_cfg, "affinity_temp_start", None) is None
        else distill_cfg.affinity_temp_start
    )
    distill_temp_end = float(
        distill_cfg.affinity_temp
        if getattr(distill_cfg, "affinity_temp_end", None) is None
        else distill_cfg.affinity_temp_end
    )
    adaptive_teacher_weight = bool(getattr(distill_cfg, "adaptive_teacher_weight", False))
    adaptive_teacher_tau = float(getattr(distill_cfg, "adaptive_teacher_tau", 0.07))
    adaptive_teacher_w_min = float(getattr(distill_cfg, "adaptive_teacher_w_min", 0.0))
    static_teacher_weights = getattr(distill_cfg, "static_teacher_weights", None)
    teacher_weight_mode = str(getattr(distill_cfg, "teacher_weight_mode", "static") or "static").lower()
    if teacher_weight_mode not in {"static", "adaptive", "adaptive_source"}:
        print(f"[Warn] Unknown teacher_weight_mode='{teacher_weight_mode}'. Fallback to 'static'.")
        teacher_weight_mode = "static"
    source_teacher_weights = getattr(distill_cfg, "source_teacher_weights", {}) or {}
    if adaptive_teacher_weight and teacher_weight_mode == "static":
        teacher_weight_mode = "adaptive_source" if source_teacher_weights else "adaptive"
        print(
            "[Config] adaptive_teacher_weight=True and teacher_weight_mode=static. "
            f"Auto-switching to '{teacher_weight_mode}'."
        )

    teacher = None
    offline_dir = distill_section.get('offline_feature_dir', None) if isinstance(distill_section, dict) else None
    use_offline = (
        distill_cfg.use_teacher
        and offline_dir is not None
        and os.path.isdir(str(offline_dir))
    )

    if distill_cfg.use_teacher and not use_offline:
        if device != "cuda":
            print("[Warn] distill.use_teacher=True but device is not cuda. Teacher may be slow on CPU.")
        try:
            teacher = create_teacher(distill_cfg, device=str(device))
            print(f"[Train] Teacher initialized (Online mode)")
        except Exception as e:
            print(f"[Train] Failed to load teacher: {e}")
            raise e
    elif use_offline:
        print(f"[Train] Offline mode — Teacher NOT loaded. Embeddings from: {offline_dir}")

    best_r10 = -1.0
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")

    log_every = int(cfg.train.get("log_every", 20))
    eval_every = int(cfg.train.get("eval_every_epochs", 1))

    print(f"[Config] epochs={epochs}, warmup={warmup_epochs}, lr={lr}, grad_clip={grad_clip}")
    print(f"[Config] w_contrastive={w_contrastive}, w_rank={w_rank}, w_distill_affinity={w_distill_affinity}, "
          f"label_smoothing={label_smoothing}")
    print(f"[Config] w_hard_negative={w_hard_negative}, w_text_text={w_text_text}, hard_negative_k={hard_negative_k}")
    print(f"[Config] logit_scale_range=[{logit_scale_min}, {logit_scale_max}]")
    print(
        f"[Config] distill_temp_schedule={distill_temp_schedule}, "
        f"start={distill_temp_start:.4f}, end={distill_temp_end:.4f}"
    )
    print(
        f"[Config] adaptive_teacher_weight={adaptive_teacher_weight}, "
        f"tau={adaptive_teacher_tau:.4f}, w_min={adaptive_teacher_w_min:.4f}, "
        f"teacher_weight_mode={teacher_weight_mode}"
    )
    if teacher_weight_mode == "static":
        print(f"[Config] static_teacher_weights={static_teacher_weights}")
    if teacher_weight_mode == "adaptive_source":
        print(f"[Config] source_teacher_weights keys={sorted(source_teacher_weights.keys())}")

    # ---- [NEW] EMA 초기화 ----
    use_ema = bool(cfg.train.get("use_ema", True))
    ema_decay = float(cfg.train.get("ema_decay", 0.999))
    ema = None
    if use_ema:
        ema = EMA(model, decay=ema_decay)
        print(f"[EMA] Enabled with decay={ema_decay}")
    else:
        print("[EMA] Disabled")

    for epoch in range(epochs):
        model.train()
        current_affinity_temp = _scheduled_distill_temp(
            epoch_idx=epoch,
            total_epochs=epochs,
            start=distill_temp_start,
            end=distill_temp_end,
            schedule=distill_temp_schedule,
        )
        if distill_cfg.use_teacher:
            print(f"[Distill] epoch {epoch+1}/{epochs} affinity_temp={current_affinity_temp:.4f}")
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}")

        for step, batch_data in enumerate(pbar, start=1):
            imgs = batch_data[0].to(device, non_blocking=True)
            toks = batch_data[1].to(device, non_blocking=True)
            metas = batch_data[2]
            offline_teacher_embs = batch_data[3] if len(batch_data) > 3 else None

            image_ids = torch.tensor([m['image_id'] for m in metas], device=device, dtype=torch.long)
            sample_sources = [str(m.get("source", "unknown")) for m in metas]

            optimizer.zero_grad(set_to_none=True)

            distill_loss_val = imgs.new_tensor(0.0)

            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                img_emb, txt_emb = model(imgs, toks)

                # [수정] Multi-GT학습 방법 적용 위해 SigLIP Loss 적용
                # logit_bias를 함께 전달하며, image_ids를 통해 1:5 매칭을 학습
                loss = w_contrastive * siglip_loss(
                    img_emb, txt_emb, 
                    model.logit_scale, 
                    model.logit_bias, 
                    image_ids
                )
                # ranking (optional)
                if w_rank > 0:
                    loss = loss + w_rank * pairwise_ranking_loss(
                        img_emb, txt_emb, model.logit_scale,
                        k=rank_k, margin=rank_margin,
                    )
                
                # [NEW] Hard Negative Mining Loss (BLIP/FG-CLIP)
                if w_hard_negative > 0:
                    loss = loss + w_hard_negative * hard_negative_contrastive_loss(
                        img_emb, txt_emb, model.logit_scale,
                        num_hard_negatives=hard_negative_k,
                    )
                
                # [NEW] Text-Text Contrastive Loss (TULIP)
                if w_text_text > 0:
                    loss = loss + w_text_text * text_text_contrastive_loss(
                        txt_emb, image_ids, model.logit_scale,
                    )

            teacher_out = get_teacher_output(
                teacher=teacher,
                imgs=imgs,
                metas=metas,
                offline_teacher_embs=offline_teacher_embs,
                device=str(device),
            )

            if teacher_out is not None:
                s_img = F.normalize(img_emb.float(), dim=-1)
                s_txt = F.normalize(txt_emb.float(), dim=-1)

                distill_loss_val = compute_affinity_distill_loss(
                    student_img=s_img,
                    student_txt=s_txt,
                    teacher_output=teacher_out,
                    teachers_cfg=distill_cfg.teachers,
                    static_teacher_weights=static_teacher_weights,
                    affinity_temp=current_affinity_temp,
                    adaptive_teacher_weight=adaptive_teacher_weight,
                    adaptive_teacher_tau=adaptive_teacher_tau,
                    adaptive_teacher_w_min=adaptive_teacher_w_min,
                    teacher_weight_mode=teacher_weight_mode,
                    source_teacher_weights=source_teacher_weights,
                    affinity_columns=distill_cfg.affinity_columns,
                    distill_margin_thr=distill_cfg.distill_margin_thr,
                    selective=True,
                    image_ids=image_ids,
                    sample_sources=sample_sources,
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

            # [NEW] EMA update
            if ema is not None:
                ema.update()

            if step % log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                postfix = {
                    "loss": f"{loss.item():.4f}",
                    "scale": f"{model.logit_scale.exp().item():.2f}",
                    "lr": f"{current_lr:.2e}",
                }
                if teacher is not None or use_offline:
                    postfix["distill"] = f"{distill_loss_val.item():.4f}"
                    postfix["dtemp"] = f"{current_affinity_temp:.3f}"
                pbar.set_postfix(postfix)
                # [OPTIONAL] WandB step logging
                logger.log({"train/loss": loss.item(), "train/lr": current_lr})

        # save checkpoints
        ckpt = {"model": model.state_dict(), "config": cfg.as_dict(), "epoch": epoch + 1}
        # [NEW] EMA state 저장
        if ema is not None:
            ckpt["ema"] = ema.state_dict()
        torch.save(ckpt, last_path)
        
        # save epoch-specific checkpoint
        epoch_path = os.path.join(out_dir, f"epoch_{epoch + 1}.pt")
        torch.save(ckpt, epoch_path)

        if (epoch + 1) % eval_every == 0:
            # [NEW] EMA 가중치로 평가 (EMA 사용 시)
            if ema is not None:
                ema.apply_shadow()
            
            metrics = evaluate(model, val_loader, device, use_bidirectional=True)
            r10 = metrics["I2T"]["R@10"]
            print(f"[epoch {epoch+1}] {format_metrics(metrics)}")

            if r10 > best_r10:
                best_r10 = r10
                torch.save(ckpt, best_path)
                print(f"  -> New best model saved! R@10={r10*100:.2f}%")
            
            # [NEW] EMA 복원
            if ema is not None:
                ema.restore()
            
            # [OPTIONAL] WandB epoch logging
            logger.log_epoch(epoch + 1, {"val/I2T_R@10": r10, "val/T2I_R@10": metrics["T2I"]["R@10"]})

    # [OPTIONAL] WandB finish
    logger.finish()

    return best_path if os.path.exists(best_path) else last_path
