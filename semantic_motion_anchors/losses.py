from __future__ import annotations

import hashlib
from typing import Sequence

import torch
import torch.nn.functional as F


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE between paired rows of `a` and `b`."""
    a = F.normalize(a, dim=-1, eps=1e-8)
    b = F.normalize(b, dim=-1, eps=1e-8)
    temperature = temperature.to(device=a.device, dtype=a.dtype)
    logits = (a @ b.t()) / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def masked_info_nce(
    a: torch.Tensor,
    b: torch.Tensor,
    temperature: torch.Tensor,
    positive_targets: torch.Tensor | None = None,
    invalid_negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Symmetric InfoNCE with soft positive targets and/or masked false negatives.

    `positive_targets` is a row-normalized target distribution (JEGAL-style soft positives); the
    reverse direction uses its transpose, as in the reported baseline, so its rows are not
    renormalized. `invalid_negative_mask` removes off-diagonal pairs from the softmax
    (TMR-style filtering).
    """
    a = F.normalize(a, dim=-1, eps=1e-8)
    b = F.normalize(b, dim=-1, eps=1e-8)
    temperature = temperature.to(device=a.device, dtype=a.dtype)
    logits = (a @ b.t()) / temperature
    batch_size = logits.size(0)
    eye = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
    if invalid_negative_mask is not None:
        invalid_negative_mask = invalid_negative_mask.to(device=logits.device, dtype=torch.bool) & ~eye
        logits = logits.masked_fill(invalid_negative_mask, torch.finfo(logits.dtype).min)

    if positive_targets is None:
        labels = torch.arange(batch_size, device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    positive_targets = positive_targets.to(device=logits.device, dtype=logits.dtype)
    if invalid_negative_mask is not None:
        positive_targets = positive_targets * (~invalid_negative_mask).to(dtype=logits.dtype)
        positive_targets = positive_targets / positive_targets.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    loss_ab = -(positive_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    loss_ba = -(positive_targets.t() * F.log_softmax(logits.t(), dim=-1)).sum(dim=-1).mean()
    return 0.5 * (loss_ab + loss_ba)


def cosine_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    embeddings = F.normalize(embeddings, dim=-1, eps=1e-8)
    return embeddings @ embeddings.t()


def soft_positive_targets(similarity: torch.Tensor, threshold: float, weight: float) -> torch.Tensor:
    """Identity targets plus `weight` on every other pair whose similarity exceeds `threshold`."""
    eye = torch.eye(similarity.size(0), device=similarity.device, dtype=similarity.dtype)
    above = (similarity >= threshold).to(dtype=similarity.dtype) * (1.0 - eye)
    targets = eye + weight * above
    return targets / targets.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def false_negative_mask(similarity: torch.Tensor, threshold: float) -> torch.Tensor:
    eye = torch.eye(similarity.size(0), device=similarity.device, dtype=torch.bool)
    return (similarity >= threshold) & ~eye


def fixed_random_anchors(uids: Sequence[str], source: str, dim: int, device: torch.device) -> torch.Tensor:
    """One fixed Gaussian unit vector per (source, sample), seeded by SHA-256 of `source::uid`."""
    rows = []
    for uid in uids:
        digest = hashlib.sha256(f"{source}::{uid}".encode("utf-8")).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest[:8], byteorder="little", signed=False))
        vector = torch.randn(dim, generator=generator, dtype=torch.float32)
        rows.append(F.normalize(vector.unsqueeze(0), dim=-1, eps=1e-8).squeeze(0))
    return torch.stack(rows).to(device)
