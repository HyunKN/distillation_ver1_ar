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
    [MobileCLIP2 전략] Sigmoid 기반 손실 함수.
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
    sim_neg = sim.clone()
    sim_neg.fill_diagonal_(-65000.0)
    topk, _ = torch.topk(sim_neg, k=min(k, B - 1), dim=1)
    loss = F.relu(margin - (diag.unsqueeze(1) - topk)).mean()
    return loss