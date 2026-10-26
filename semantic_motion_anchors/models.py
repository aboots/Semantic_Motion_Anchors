from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_length: int) -> None:
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ProjectionHead(nn.Module):
    """LayerNorm -> Linear -> GELU -> Dropout -> Linear, followed by L2 normalization."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1, eps=1e-8)


class MotionEncoder(nn.Module):
    """Transformer over per-frame pose vectors with masked mean pooling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_length: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.positional_encoding = SinusoidalPositionalEncoding(hidden_dim, max_length=max_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, motions: torch.Tensor, motion_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.positional_encoding(self.input_projection(motions))
        encoded = self.encoder(hidden, src_key_padding_mask=~motion_mask)
        lengths = motion_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        pooled = (encoded * motion_mask.unsqueeze(-1)).sum(dim=1) / lengths
        return self.output_norm(pooled)


class GestureRetrievalModel(nn.Module):
    """Frozen-text-encoder embeddings and motion projected into one retrieval space.

    `anchor_projection` (pi_an) is shared by the physical-form and intent anchors,
    `transcript_projection` (pi_tr) embeds transcripts and `motion_projection` (pi_mot)
    sits on top of the motion transformer. Only the transcript and motion branches are used
    for retrieval at test time.
    """

    def __init__(
        self,
        text_dim: int,
        motion_dim: int,
        proj_dim: int,
        text_hidden_dim: int,
        dropout: float,
        temperature: float,
        learnable_temperature: bool,
        motion_hidden_dim: int,
        motion_layers: int,
        motion_heads: int,
        motion_dropout: float,
        motion_max_length: int,
    ) -> None:
        super().__init__()
        self.anchor_projection = ProjectionHead(text_dim, text_hidden_dim, proj_dim, dropout)
        self.transcript_projection = ProjectionHead(text_dim, text_hidden_dim, proj_dim, dropout)
        init_scale = torch.tensor(1.0 / max(1e-6, float(temperature))).log()
        if learnable_temperature:
            self.logit_scale = nn.Parameter(init_scale)
        else:
            self.register_buffer("logit_scale", init_scale)
        self.max_logit_scale = math.log(100.0)
        self.motion_encoder = MotionEncoder(
            input_dim=motion_dim,
            hidden_dim=motion_hidden_dim,
            num_layers=motion_layers,
            num_heads=motion_heads,
            dropout=motion_dropout,
            max_length=motion_max_length,
        )
        self.motion_projection = ProjectionHead(
            motion_hidden_dim, motion_hidden_dim, proj_dim, motion_dropout
        )

    @property
    def temperature(self) -> torch.Tensor:
        return torch.clamp(self.logit_scale, max=self.max_logit_scale).exp().reciprocal()

    def encode_anchor(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.anchor_projection(embeddings)

    def encode_transcript(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.transcript_projection(embeddings)

    def encode_motion(self, motions: torch.Tensor, motion_mask: torch.Tensor) -> torch.Tensor:
        return self.motion_projection(self.motion_encoder(motions, motion_mask))


def build_model(cfg: dict[str, Any], text_dim: int, motion_dim: int) -> GestureRetrievalModel:
    model_cfg = cfg["model"]
    motion_cfg = model_cfg["motion_encoder"]
    return GestureRetrievalModel(
        text_dim=text_dim,
        motion_dim=motion_dim,
        proj_dim=int(model_cfg["proj_dim"]),
        text_hidden_dim=int(model_cfg["text_hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        temperature=float(model_cfg["temperature"]),
        learnable_temperature=bool(model_cfg["learnable_temperature"]),
        motion_hidden_dim=int(motion_cfg["hidden_dim"]),
        motion_layers=int(motion_cfg["num_layers"]),
        motion_heads=int(motion_cfg["num_heads"]),
        motion_dropout=float(motion_cfg["dropout"]),
        motion_max_length=int(motion_cfg["max_length"]),
    )


def load_trained_model(
    checkpoint_path: str,
    cfg: dict[str, Any],
    device: torch.device,
) -> GestureRetrievalModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg, int(checkpoint["text_dim"]), int(checkpoint["motion_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model
