from __future__ import annotations

import torch
import torch.nn.functional as F


def clip_contrastive_loss(
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    logit_scale: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    기존 CLIP 스타일의 1:1 대칭 손실 함수.
    
    Args:
        img_emb: Image embeddings [B, D]
        txt_emb: Text embeddings [B, D]
        logit_scale: Learnable temperature parameter
        label_smoothing: Label smoothing factor
    
    Returns:
        Contrastive loss value
    """
    logits = logit_scale.exp() * img_emb @ txt_emb.t()
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    loss_t = F.cross_entropy(logits.t(), labels, label_smoothing=label_smoothing)
    return 0.5 * (loss_i + loss_t)


def multi_gt_masked_contrastive_loss(
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    image_ids: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    [LPCVC 필수] 동일한 image_id를 가진 모든 쌍을 정답(Positive)으로 처리하는 손실 함수.
    COCO 데이터셋의 1:5 구조에서 False Negative 패널티를 제거합니다.
    
    Args:
        img_emb: Image embeddings [B, D]
        txt_emb: Text embeddings [B, D]
        image_ids: Image ID tensor [B]
        logit_scale: Learnable temperature parameter
    
    Returns:
        Multi-GT contrastive loss value
    """
    logits = logit_scale.exp() * img_emb @ txt_emb.t()  # [B, B]
    
    # 배치 내에서 동일한 image_id를 가진 위치는 1, 아니면 0인 마스크 생성
    # image_ids: [B] 형태의 텐서
    mask = (image_ids.unsqueeze(0) == image_ids.unsqueeze(1)).float().to(logits.device)
    
    # 행(Row) 단위로 정답 확률의 합이 1이 되도록 정규화 (Soft Target)
    targets = mask / mask.sum(dim=1, keepdim=True)
    
    # Soft Target에 대해 KL-Divergence를 사용하여 확률 분포 정렬
    loss_i = -torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1).mean()
    loss_t = -torch.sum(F.log_softmax(logits.t(), dim=1) * targets.t(), dim=1).mean()
    
    return 0.5 * (loss_i + loss_t)


def siglip_loss(
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
    image_ids: torch.Tensor,
) -> torch.Tensor:
    """
    [현재 기본 전략] Sigmoid 기반 손실 함수.
    배치 사이즈가 작을 때 Softmax보다 훨씬 안정적이며 Recall 향상에 유리합니다.
    
    Args:
        img_emb: Image embeddings [B, D]
        txt_emb: Text embeddings [B, D]
        logit_scale: Learnable temperature parameter
        logit_bias: Learnable bias parameter
        image_ids: Image ID tensor [B]
    
    Returns:
        SigLIP loss value
    """
    n = img_emb.size(0)
    logits = logit_scale.exp() * (img_emb @ txt_emb.t()) + logit_bias
    
    # 정답 마스크: 같은 image_id면 1, 다르면 -1
    mask = (image_ids.unsqueeze(0) == image_ids.unsqueeze(1)).float().to(logits.device)
    labels = 2 * mask - 1
    
    # Pairwise Sigmoid Loss
    loss = -F.logsigmoid(labels * logits).sum() / n
    return loss


def pairwise_ranking_loss(
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    logit_scale: torch.Tensor,
    k: int = 3,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    기존 Hard Negative Ranking Loss 유지.
    
    Args:
        img_emb: Image embeddings [B, D]
        txt_emb: Text embeddings [B, D]
        logit_scale: Learnable temperature parameter
        k: Number of hard negatives
        margin: Margin for ranking loss
    
    Returns:
        Pairwise ranking loss value
    """
    sim = logit_scale.exp() * (img_emb @ txt_emb.t())
    diag = sim.diag()
    B = sim.size(0)
    # [최적화] clone() 대신 masked_fill 사용으로 메모리 효율 개선
    mask = torch.eye(B, device=sim.device).bool()
    sim_neg = sim.masked_fill(mask, float('-inf'))
    topk, _ = torch.topk(sim_neg, k=min(k, B - 1), dim=1)
    loss = F.relu(margin - (diag.unsqueeze(1) - topk)).mean()
    return loss


def hard_negative_contrastive_loss(
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    logit_scale: torch.Tensor,
    num_hard_negatives: int = 5,
) -> torch.Tensor:
    """
    [BLIP/FG-CLIP 기반] Hard Negative Mining Contrastive Loss.
    
    배치 내에서 가장 혼동되기 쉬운(similarity가 높지만 실제로는 negative인) 
    샘플만 선택하여 집중 학습합니다.
    
    기존 contrastive loss는 모든 negative를 동일하게 취급하지만,
    이 loss는 어려운 negative에 더 집중하여 fine-grained 구분 능력을 향상시킵니다.
    
    Args:
        img_emb: Image embeddings [B, D], L2 normalized
        txt_emb: Text embeddings [B, D], L2 normalized
        logit_scale: Learnable temperature parameter
        num_hard_negatives: Top-K hard negatives to select (default: 5)
    
    Returns:
        Hard negative contrastive loss value
    """
    B = img_emb.size(0)
    temperature = logit_scale.exp()
    
    # Compute full similarity matrix
    sim = temperature * (img_emb @ txt_emb.t())  # [B, B]
    
    # Get positive scores (diagonal)
    pos_scores = sim.diag().unsqueeze(1)  # [B, 1]
    
    # Mask out positives and select top-K hard negatives
    mask = torch.eye(B, device=sim.device).bool()
    sim_neg = sim.masked_fill(mask, float('-inf'))
    
    # Select top-K hard negatives (highest similarity among negatives)
    k = min(num_hard_negatives, B - 1)
    hard_neg_scores, _ = sim_neg.topk(k, dim=1)  # [B, K]
    
    # Concatenate positive and hard negatives: [B, 1+K]
    logits = torch.cat([pos_scores, hard_neg_scores], dim=1)
    
    # Labels: positive is always at index 0
    labels = torch.zeros(B, dtype=torch.long, device=sim.device)
    
    # Cross entropy loss
    loss_i2t = F.cross_entropy(logits, labels)
    
    # Same for text-to-image direction
    sim_t2i = sim.t()
    pos_scores_t2i = sim_t2i.diag().unsqueeze(1)
    sim_neg_t2i = sim_t2i.masked_fill(mask, float('-inf'))
    hard_neg_scores_t2i, _ = sim_neg_t2i.topk(k, dim=1)
    logits_t2i = torch.cat([pos_scores_t2i, hard_neg_scores_t2i], dim=1)
    loss_t2i = F.cross_entropy(logits_t2i, labels)
    
    return 0.5 * (loss_i2t + loss_t2i)


def text_text_contrastive_loss(
    txt_emb: torch.Tensor,
    image_ids: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    [TULIP 기반] Text-Text Contrastive Loss.
    
    같은 이미지를 설명하는 캡션들(COCO의 5개 캡션)을 positive pair로,
    다른 이미지의 캡션들을 negative pair로 학습합니다.
    
    이를 통해 텍스트 임베딩이 같은 의미를 가진 다양한 표현을 
    더 가까이 매핑하도록 학습됩니다.
    
    Args:
        txt_emb: Text embeddings [B, D], L2 normalized
        image_ids: Image ID tensor [B] - 같은 image_id를 가진 텍스트는 positive
        logit_scale: Learnable temperature parameter
    
    Returns:
        Text-text contrastive loss value
    """
    B = txt_emb.size(0)
    temperature = logit_scale.exp()
    
    # Compute text-text similarity
    sim = temperature * (txt_emb @ txt_emb.t())  # [B, B]
    
    # Create positive mask: same image_id means positive pair
    # image_ids: [B] -> mask[i,j] = 1 if image_ids[i] == image_ids[j]
    pos_mask = (image_ids.unsqueeze(0) == image_ids.unsqueeze(1)).float()
    
    # Remove self-similarity (diagonal)
    pos_mask.fill_diagonal_(0)
    
    # If no positive pairs exist (all unique image_ids), return 0
    if pos_mask.sum() == 0:
        return txt_emb.new_tensor(0.0)
    
    # Normalize to get soft targets (sum to 1 per row)
    # Only consider rows that have at least one positive
    row_has_positive = pos_mask.sum(dim=1) > 0
    
    if not row_has_positive.any():
        return txt_emb.new_tensor(0.0)
    
    # Filter rows with positives
    sim_filtered = sim[row_has_positive]
    pos_mask_filtered = pos_mask[row_has_positive]
    
    # Create soft targets
    targets = pos_mask_filtered / pos_mask_filtered.sum(dim=1, keepdim=True).clamp(min=1e-8)
    
    # KL divergence style loss
    log_probs = F.log_softmax(sim_filtered, dim=1)
    loss = -torch.sum(log_probs * targets, dim=1).mean()
    
    return loss
