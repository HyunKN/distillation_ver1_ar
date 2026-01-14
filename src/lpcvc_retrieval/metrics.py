from __future__ import annotations
import torch
from typing import Tuple

@torch.no_grad()
def recall_at_k(sim: torch.Tensor, ks: list = [1, 5, 10]) -> dict:
    """
    Compute Recall@K for given similarity matrix.
    
    Args:
        sim: [N, M] similarity matrix (queries x gallery)
        ks: list of K values to compute recall for
    
    Returns:
        Dictionary with recall values for each K
    """
    ranks = torch.argsort(sim, dim=1, descending=True)
    N = sim.size(0)
    targets = torch.arange(N, device=sim.device).unsqueeze(1)
    hits = (ranks == targets).nonzero(as_tuple=False)
    
    pos = torch.full((N,), fill_value=sim.size(1), device=sim.device, dtype=torch.long)
    pos[hits[:,0]] = hits[:,1]
    
    results = {}
    for k in ks:
        results[f'R@{k}'] = (pos < k).float().mean().item()
    return results

@torch.no_grad()
def recall_at_1_5_10(image_emb: torch.Tensor, text_emb: torch.Tensor) -> Tuple[float, float, float]:
    """
    Compute Image-to-Text Recall@1, 5, 10.
    Legacy function for backward compatibility.
    """
    sim = image_emb @ text_emb.t()
    results = recall_at_k(sim)
    return results['R@1'], results['R@5'], results['R@10']

@torch.no_grad()
def bidirectional_recall(image_emb: torch.Tensor, text_emb: torch.Tensor, ks: list = [1, 5, 10]) -> dict:
    """
    Compute bidirectional retrieval metrics:
    - I2T: Image-to-Text retrieval (given image, find matching text)
    - T2I: Text-to-Image retrieval (given text, find matching image)
    
    Args:
        image_emb: [N, D] normalized image embeddings
        text_emb: [N, D] normalized text embeddings  
        ks: list of K values for Recall@K
    
    Returns:
        Dictionary with I2T, T2I, and mean metrics
    """
    # Image-to-Text
    sim_i2t = image_emb @ text_emb.t()
    i2t = recall_at_k(sim_i2t, ks)
    
    # Text-to-Image
    sim_t2i = text_emb @ image_emb.t()
    t2i = recall_at_k(sim_t2i, ks)
    
    results = {
        'I2T': i2t,
        'T2I': t2i,
        'mean': {}
    }
    
    for k in ks:
        key = f'R@{k}'
        results['mean'][key] = (i2t[key] + t2i[key]) / 2
    
    return results

def format_metrics(metrics: dict) -> str:
    """Format bidirectional metrics for logging."""
    i2t = metrics['I2T']
    t2i = metrics['T2I']
    mean = metrics['mean']
    
    lines = [
        f"I2T: R@1={i2t['R@1']*100:.2f}% R@5={i2t['R@5']*100:.2f}% R@10={i2t['R@10']*100:.2f}%",
        f"T2I: R@1={t2i['R@1']*100:.2f}% R@5={t2i['R@5']*100:.2f}% R@10={t2i['R@10']*100:.2f}%",
        f"Mean: R@1={mean['R@1']*100:.2f}% R@5={mean['R@5']*100:.2f}% R@10={mean['R@10']*100:.2f}%"
    ]
    return " | ".join(lines)
