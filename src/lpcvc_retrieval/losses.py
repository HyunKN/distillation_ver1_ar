from __future__ import annotations
import torch
import torch.nn.functional as F

def clip_contrastive_loss(img_emb: torch.Tensor, txt_emb: torch.Tensor, logit_scale: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """
    CLIP-style symmetric contrastive loss.
    
    Args:
        img_emb: [B, D] normalized image embeddings
        txt_emb: [B, D] normalized text embeddings
        logit_scale: learnable temperature parameter (log scale)
        label_smoothing: label smoothing factor (0.0 = no smoothing)
    
    Returns:
        Symmetric contrastive loss
    """
    logits = logit_scale.exp() * img_emb @ txt_emb.t()  # [B,B]
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    loss_t = F.cross_entropy(logits.t(), labels, label_smoothing=label_smoothing)
    return 0.5 * (loss_i + loss_t)

def pairwise_ranking_loss(img_emb: torch.Tensor, txt_emb: torch.Tensor, logit_scale: torch.Tensor, k: int = 3, margin: float = 0.1) -> torch.Tensor:
    """
    Hard negative pairwise ranking loss for additional supervision.
    
    Args:
        img_emb: [B, D] normalized image embeddings
        txt_emb: [B, D] normalized text embeddings
        logit_scale: learnable temperature parameter
        k: number of hard negatives to consider
        margin: margin for hinge loss
    
    Returns:
        Ranking loss
    """
    sim = logit_scale.exp() * (img_emb @ txt_emb.t())  # [B,B]
    diag = sim.diag()  # [B]
    # mask diagonal for negatives
    B = sim.size(0)
    sim_neg = sim.clone()
    sim_neg.fill_diagonal_(-65000.0)  # float16-safe (max ~65504)
    topk, _ = torch.topk(sim_neg, k=min(k, B-1), dim=1)
    # hinge: margin - (pos - neg)
    loss = F.relu(margin - (diag.unsqueeze(1) - topk)).mean()
    return loss
