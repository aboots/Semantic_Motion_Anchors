from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def ground_truth_ranks(query: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
    """0-based rank of gallery item i for query i, over the full gallery (cosine similarity)."""
    similarity = F.normalize(query, dim=-1, eps=1e-8) @ F.normalize(gallery, dim=-1, eps=1e-8).t()
    ranked = torch.argsort(similarity, dim=1, descending=True)
    targets = torch.arange(ranked.size(0)).unsqueeze(1)
    return (ranked.cpu() == targets).nonzero()[:, 1]


def recall_and_mrr(ranks: torch.Tensor, ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    metrics = {f"R@{k}": float((ranks < k).double().mean().item()) for k in ks}
    metrics["MRR"] = float((1.0 / (ranks.double() + 1.0)).mean().item())
    return metrics


def retrieval_metrics(query: torch.Tensor, gallery: torch.Tensor) -> dict[str, float]:
    return recall_and_mrr(ground_truth_ranks(query, gallery))


def text_motion_metrics(text: torch.Tensor, motion: torch.Tensor) -> dict[str, Any]:
    text_to_motion = ground_truth_ranks(text, motion)
    motion_to_text = ground_truth_ranks(motion, text)
    t2m = recall_and_mrr(text_to_motion)
    m2t = recall_and_mrr(motion_to_text)
    return {
        "num_samples": int(text.size(0)),
        "text_to_motion": t2m,
        "motion_to_text": m2t,
        "mean_mrr": 0.5 * (t2m["MRR"] + m2t["MRR"]),
        "ranks": {
            "text_to_motion": text_to_motion.tolist(),
            "motion_to_text": motion_to_text.tolist(),
        },
    }
