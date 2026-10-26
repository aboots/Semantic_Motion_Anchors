"""Train a text-gesture retrieval model.

    python -m semantic_motion_anchors.train --config configs/text_contrastive.yaml --seed 42
    python -m semantic_motion_anchors.train --config configs/semantic_anchors.yaml --seed 42
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import apply_seed, load_config, manifest_path, run_dir, save_json, text_cache_dir
from .data import GestureDataset, TextEmbeddingStore, collate_batch, read_manifest
from .losses import (
    cosine_similarity_matrix,
    false_negative_mask,
    fixed_random_anchors,
    info_nce,
    masked_info_nce,
    soft_positive_targets,
)
from .metrics import text_motion_metrics
from .models import build_model

BASE_OPERANDS = ("anchor", "transcript", "motion")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def auxiliary_sources(cfg: dict[str, Any]) -> list[str]:
    """Anchor sources used by the loss terms, in order of first use."""
    sources: list[str] = []
    for term in cfg["training"]["losses"].values():
        for operand in (term["left"], term["right"]):
            if operand not in BASE_OPERANDS and not operand.startswith("random_") and operand not in sources:
                sources.append(operand)
    return sources


def load_stores(cfg: dict[str, Any], sources: list[str]) -> dict[str, TextEmbeddingStore]:
    cache_dir = text_cache_dir(cfg)
    return {source: TextEmbeddingStore(cache_dir / f"{source}.npz") for source in ["transcript", "anchor", *sources]}


def make_dataset(
    cfg: dict[str, Any],
    rows: list[dict[str, str]],
    stores: dict[str, TextEmbeddingStore],
    split: str | list[str],
    extra_sources: list[str],
) -> GestureDataset:
    return GestureDataset(
        rows,
        transcript_store=stores["transcript"],
        anchor_store=stores["anchor"],
        splits=split,
        include_motion=True,
        extra_stores={source: stores[source] for source in extra_sources},
        ted_motion_path=cfg["paths"].get("ted_motion") or "",
        mismatched_source_groups=cfg["training"].get("mismatched_sources", []),
        mismatch_seed=int(cfg["training"]["seed"]),
    )


def make_loader(dataset: GestureDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_batch)


def compute_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    training_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    anchor_raw = batch["anchor_embeddings"].to(device)
    transcript_raw = batch["transcript_embeddings"].to(device)
    # The unified anchor is encoded even when no loss term uses it, which keeps the dropout
    # random stream identical to the runs reported in the paper.
    encoded = {
        "anchor": model.encode_anchor(anchor_raw),
        "transcript": model.encode_transcript(transcript_raw),
        "motion": model.encode_motion(batch["motions"].to(device), batch["motion_mask"].to(device)),
    }

    def operand(name: str) -> torch.Tensor:
        if name not in encoded:
            if name.startswith("random_"):
                raw = fixed_random_anchors(batch["uids"], name, anchor_raw.shape[-1], device)
            else:
                raw = batch["extra_anchor_embeddings"][name].to(device)
            encoded[name] = model.encode_anchor(raw)
        return encoded[name]

    # Baseline-specific handling of near-duplicate pairs in the transcript-motion term.
    positive_targets = invalid_negatives = None
    soft_cfg = training_cfg.get("soft_positives")
    filter_cfg = training_cfg.get("false_negative_filtering")
    if soft_cfg or filter_cfg:
        similarity_source = (soft_cfg or filter_cfg)["similarity"]
        raw = anchor_raw if similarity_source == "anchor" else transcript_raw
        similarity = cosine_similarity_matrix(raw)
        if soft_cfg:
            positive_targets = soft_positive_targets(similarity, float(soft_cfg["threshold"]), float(soft_cfg["weight"]))
        if filter_cfg:
            invalid_negatives = false_negative_mask(similarity, float(filter_cfg["threshold"]))

    total = torch.zeros((), device=device)
    logs: dict[str, float] = {}
    for name, term in training_cfg["losses"].items():
        left, right = operand(term["left"]), operand(term["right"])
        if (term["left"], term["right"]) == ("transcript", "motion") and (
            positive_targets is not None or invalid_negatives is not None
        ):
            loss = masked_info_nce(left, right, model.temperature, positive_targets, invalid_negatives)
        else:
            loss = info_nce(left, right, model.temperature)
        total = total + float(term["weight"]) * loss
        logs[f"loss_{name}"] = float(loss.item())
    logs["loss"] = float(total.item())
    return total, logs


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    training_cfg: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    totals: dict[str, float] = {}
    num_batches = 0
    for batch in loader:
        loss, logs = compute_loss(model, batch, device, training_cfg)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg["grad_clip"]))
            optimizer.step()
        for key, value in logs.items():
            totals[key] = totals.get(key, 0.0) + value
        num_batches += 1
    return {key: value / max(1, num_batches) for key, value in totals.items()}


@torch.no_grad()
def embed_text_and_motion(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    text, motion = [], []
    for batch in loader:
        text.append(model.encode_transcript(batch["transcript_embeddings"].to(device)).cpu())
        motion.append(model.encode_motion(batch["motions"].to(device), batch["motion_mask"].to(device)).cpu())
    return torch.cat(text), torch.cat(motion)


def reset_parameters(module: torch.nn.Module) -> None:
    for child in module.modules():
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()


def rng_state() -> dict[str, Any]:
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (name, keys.tolist(), position, has_gauss, cached),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    # The checkpoint may have been loaded onto the GPU; RNG states must be CPU byte tensors.
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in state["cuda"]])


def save_checkpoint(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="Training seed; appends _seed{N} to the run name.")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Config overrides.")
    parser.add_argument("--resume", action="store_true", help="Resume from the run's last checkpoint.")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg = apply_seed(cfg, args.seed)
    training_cfg = cfg["training"]
    out_dir = run_dir(cfg)
    save_json(out_dir / "config.json", cfg)

    seed_everything(int(training_cfg["seed"]))
    device = get_device(training_cfg.get("device", "auto"))
    rows = read_manifest(manifest_path(cfg))
    extra_sources = auxiliary_sources(cfg)
    stores = load_stores(cfg, extra_sources)
    train_set = make_dataset(cfg, rows, stores, "beat_train", extra_sources)
    val_set = make_dataset(cfg, rows, stores, "beat_val", extra_sources)
    motion_dim = int(train_set[0]["motion"].shape[-1])
    text_dim = stores["transcript"].dim

    model = build_model(cfg, text_dim, motion_dim).to(device)
    train_loader = make_loader(train_set, int(training_cfg["batch_size"]), shuffle=True)
    val_loader = make_loader(val_set, int(training_cfg["eval_batch_size"]), shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    print(f"{cfg['name']}: train={len(train_set)} val={len(val_set)} device={device}")
    print("Loss terms: " + ", ".join(
        f"{name}({term['left']}, {term['right']}) x {term['weight']}" for name, term in training_cfg["losses"].items()
    ))

    best_score, patience, start_epoch = -math.inf, 0, 1
    history: list[dict[str, Any]] = []
    last_path, best_path = out_dir / "checkpoints" / "last.pt", out_dir / "checkpoints" / "best.pt"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        best_score, patience, history = state["best_score"], state["patience"], state["history"]
        restore_rng_state(state["rng_state"])
        start_epoch = int(state["epoch"]) + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        if training_cfg.get("init_from"):
            init_path = run_dir(cfg, training_cfg["init_from"]) / "checkpoints" / "best.pt"
            init_state = torch.load(init_path, map_location=device)["model_state"]
            model.load_state_dict(init_state)
            print(f"Initialized from {init_path}")
        for name in training_cfg.get("reset_modules", []):
            reset_parameters(model.get_submodule(name))
            print(f"Re-initialized {name}")

    for epoch in range(start_epoch, int(training_cfg["epochs"]) + 1):
        train_logs = run_epoch(model, train_loader, device, training_cfg, optimizer)
        with torch.no_grad():
            val_logs = run_epoch(model, val_loader, device, training_cfg)
        text, motion = embed_text_and_motion(model, val_loader, device)
        val_metrics = text_motion_metrics(text, motion)
        val_metrics.pop("ranks")
        score = val_metrics["mean_mrr"]
        history.append({"epoch": epoch, "train": train_logs, "val_loss": val_logs, "val_retrieval": val_metrics})
        print(
            f"epoch={epoch:03d} train_loss={train_logs['loss']:.4f} "
            f"val_loss={val_logs['loss']:.4f} val_mean_mrr={score:.4f}"
        )

        improved = score > best_score
        if improved:
            best_score, patience = score, 0
        else:
            patience += 1
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "best_score": best_score,
            "patience": patience,
            "history": history,
            "text_dim": text_dim,
            "motion_dim": motion_dim,
            "rng_state": rng_state(),
        }
        save_checkpoint(last_path, **checkpoint)
        save_json(out_dir / "history.json", history)
        if improved:
            save_checkpoint(best_path, **checkpoint)
        elif patience >= int(training_cfg["early_stopping_patience"]):
            print(f"Early stopping after epoch {epoch}.")
            break
    print(f"Best validation mean MRR: {best_score:.4f} ({best_path})")


if __name__ == "__main__":
    main()
