from __future__ import annotations
import torch
from typing import Tuple, List, Dict, Optional
from collections import defaultdict

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
    competition = metrics.get("competition")
    if isinstance(competition, dict) and competition:
        comp_parts = []
        for key in ("I2T_text_R@1", "I2T_text_R@5", "I2T_text_R@10"):
            if key in competition:
                label = key.replace("I2T_text_", "")
                comp_parts.append(f"{label}={competition[key]*100:.2f}%")
        if comp_parts:
            lines.append("CompI2TText: " + " ".join(comp_parts))
    return " | ".join(lines)


# ============================================================================
# COCO-style evaluation with image_id-based matching
# ============================================================================

@torch.no_grad()
def coco_i2t_recall(
    unique_image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    unique_image_ids: List[int],
    text_image_ids: List[int],
    ks: List[int] = [1, 5, 10],
    chunk_size: int = 256,
) -> Dict[str, float]:
    """
    COCO-style Image-to-Text Recall.
    
    For each unique image, find if ANY of its matching captions are in top-K.
    
    Args:
        unique_image_emb: [N_img, D] unique image embeddings (deduplicated)
        text_emb: [N_txt, D] all text embeddings
        unique_image_ids: [N_img] image_id for each unique image
        text_image_ids: [N_txt] image_id for each text (caption)
        ks: list of K values
        chunk_size: chunk size for similarity computation (OOM prevention)
    
    Returns:
        Dictionary with recall values for each K
    """
    device = unique_image_emb.device
    N_img = unique_image_emb.size(0)
    N_txt = text_emb.size(0)
    
    # Build mapping: image_id -> list of text indices
    imgid_to_txt_indices = defaultdict(list)
    for txt_idx, img_id in enumerate(text_image_ids):
        imgid_to_txt_indices[img_id].append(txt_idx)
    
    # For each unique image, find the best rank among all its matching captions
    best_ranks = []
    
    for start in range(0, N_img, chunk_size):
        end = min(start + chunk_size, N_img)
        chunk_img_emb = unique_image_emb[start:end]  # [chunk, D]
        
        # Compute similarity: [chunk, N_txt]
        sim = chunk_img_emb @ text_emb.t()
        
        # Sort by similarity (descending)
        sorted_indices = torch.argsort(sim, dim=1, descending=True)  # [chunk, N_txt]
        
        for i, global_idx in enumerate(range(start, end)):
            img_id = unique_image_ids[global_idx]
            target_txt_indices = set(imgid_to_txt_indices[img_id])
            
            # Find best (minimum) rank among all matching captions
            best_rank = N_txt  # worst case
            for rank, txt_idx in enumerate(sorted_indices[i].tolist()):
                if txt_idx in target_txt_indices:
                    best_rank = rank
                    break
            best_ranks.append(best_rank)
    
    best_ranks = torch.tensor(best_ranks, device=device, dtype=torch.long)
    
    results = {}
    for k in ks:
        results[f'R@{k}'] = (best_ranks < k).float().mean().item()
    return results


@torch.no_grad()
def coco_i2t_text_recall(
    unique_image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    unique_image_ids: List[int],
    text_image_ids: List[int],
    ks: List[int] = [1, 5, 10],
    chunk_size: int = 256,
) -> Dict[str, float]:
    """
    Competition-style Image-to-Text Recall.

    For each image query, count how many of its ground-truth texts appear in top-K.
    Final recall is:
      (# ground-truth texts retrieved in top-K across all image queries)
      / (total # ground-truth texts)

    This follows the competition-side interpretation more closely than the
    COCO-style "any matching caption hits top-K" metric.
    """
    N_img = unique_image_emb.size(0)
    max_k = max(ks)

    # Build mapping: image_id -> list of text indices
    imgid_to_txt_indices = defaultdict(list)
    for txt_idx, img_id in enumerate(text_image_ids):
        imgid_to_txt_indices[img_id].append(txt_idx)

    total_gt_texts = sum(len(indices) for indices in imgid_to_txt_indices.values())
    retrieved_counts = {f"R@{k}": 0 for k in ks}

    for start in range(0, N_img, chunk_size):
        end = min(start + chunk_size, N_img)
        chunk_img_emb = unique_image_emb[start:end]

        sim = chunk_img_emb @ text_emb.t()
        topk_indices = torch.topk(sim, k=min(max_k, sim.size(1)), dim=1, largest=True).indices

        for i, global_idx in enumerate(range(start, end)):
            img_id = unique_image_ids[global_idx]
            target_txt_indices = set(imgid_to_txt_indices[img_id])
            ranked_txt_indices = topk_indices[i].tolist()

            for k in ks:
                hits = sum(1 for txt_idx in ranked_txt_indices[:k] if txt_idx in target_txt_indices)
                retrieved_counts[f"R@{k}"] += hits

    if total_gt_texts <= 0:
        return {f"R@{k}": 0.0 for k in ks}

    return {
        key: float(value) / float(total_gt_texts)
        for key, value in retrieved_counts.items()
    }


@torch.no_grad()
def coco_t2i_recall(
    unique_image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    unique_image_ids: List[int],
    text_image_ids: List[int],
    ks: List[int] = [1, 5, 10],
    chunk_size: int = 256,
) -> Dict[str, float]:
    """
    COCO-style Text-to-Image Recall.
    
    For each caption, find if its corresponding image is in top-K.
    
    Args:
        unique_image_emb: [N_img, D] unique image embeddings (deduplicated)
        text_emb: [N_txt, D] all text embeddings
        unique_image_ids: [N_img] image_id for each unique image
        text_image_ids: [N_txt] image_id for each text (caption)
        ks: list of K values
        chunk_size: chunk size for similarity computation (OOM prevention)
    
    Returns:
        Dictionary with recall values for each K
    """
    device = text_emb.device
    N_img = unique_image_emb.size(0)
    N_txt = text_emb.size(0)
    
    # Build mapping: image_id -> unique image index
    imgid_to_img_idx = {img_id: idx for idx, img_id in enumerate(unique_image_ids)}
    
    ranks = []
    
    for start in range(0, N_txt, chunk_size):
        end = min(start + chunk_size, N_txt)
        chunk_txt_emb = text_emb[start:end]  # [chunk, D]
        
        # Compute similarity: [chunk, N_img]
        sim = chunk_txt_emb @ unique_image_emb.t()
        
        # Sort by similarity (descending)
        sorted_indices = torch.argsort(sim, dim=1, descending=True)  # [chunk, N_img]
        
        for i, global_idx in enumerate(range(start, end)):
            target_img_id = text_image_ids[global_idx]
            target_img_idx = imgid_to_img_idx[target_img_id]
            
            # Find rank of target image
            rank_pos = (sorted_indices[i] == target_img_idx).nonzero(as_tuple=False)
            if len(rank_pos) > 0:
                ranks.append(rank_pos[0, 0].item())
            else:
                ranks.append(N_img)
    
    ranks = torch.tensor(ranks, device=device, dtype=torch.long)
    
    results = {}
    for k in ks:
        results[f'R@{k}'] = (ranks < k).float().mean().item()
    return results


@torch.no_grad()
def coco_bidirectional_recall(
    unique_image_emb: torch.Tensor,
    text_emb: torch.Tensor,
    unique_image_ids: List[int],
    text_image_ids: List[int],
    ks: List[int] = [1, 5, 10],
    chunk_size: int = 256,
) -> Dict:
    """
    COCO-style bidirectional recall with proper image_id matching.
    
    Args:
        unique_image_emb: [N_img, D] deduplicated image embeddings
        text_emb: [N_txt, D] all text embeddings (one per caption)
        unique_image_ids: [N_img] image_id for each unique image
        text_image_ids: [N_txt] image_id for each caption
        ks: list of K values
        chunk_size: chunk size for OOM prevention
    
    Returns:
        Dictionary with I2T, T2I, and mean metrics
    """
    i2t = coco_i2t_recall(
        unique_image_emb, text_emb, unique_image_ids, text_image_ids, ks, chunk_size
    )
    competition_i2t = coco_i2t_text_recall(
        unique_image_emb, text_emb, unique_image_ids, text_image_ids, ks, chunk_size
    )
    t2i = coco_t2i_recall(
        unique_image_emb, text_emb, unique_image_ids, text_image_ids, ks, chunk_size
    )
    
    results = {
        'I2T': i2t,
        'T2I': t2i,
        'mean': {},
        'competition': {},
    }
    
    for k in ks:
        key = f'R@{k}'
        results['mean'][key] = (i2t[key] + t2i[key]) / 2
        results['competition'][f'I2T_text_{key}'] = competition_i2t[key]
    
    return results
