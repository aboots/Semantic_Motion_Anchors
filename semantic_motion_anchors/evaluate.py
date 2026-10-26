"""Evaluate text<->gesture retrieval on the BEAT2 validation and test splits.

Retrieval uses only the transcript and motion branches, over the full split gallery.

    python -m semantic_motion_anchors.evaluate --config configs/semantic_anchors.yaml --seed 42
"""
from __future__ import annotations

import argparse

from .config import apply_seed, load_config, manifest_path, run_dir, save_json
from .data import read_manifest
from .metrics import text_motion_metrics
from .models import load_trained_model
from .train import embed_text_and_motion, get_device, load_stores, make_dataset, make_loader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--splits", nargs="+", default=["beat_val", "beat_test"])
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg = apply_seed(cfg, args.seed)
    device = get_device(cfg["training"].get("device", "auto"))
    out_dir = run_dir(cfg)
    model = load_trained_model(str(out_dir / "checkpoints" / "best.pt"), cfg, device)
    rows = read_manifest(manifest_path(cfg))
    stores = load_stores(cfg, [])

    for split in args.splits:
        dataset = make_dataset(cfg, rows, stores, split, [])
        loader = make_loader(dataset, int(cfg["training"]["eval_batch_size"]), shuffle=False)
        text, motion = embed_text_and_motion(model, loader, device)
        metrics = text_motion_metrics(text, motion)
        save_json(out_dir / "metrics" / f"{split}.json", metrics)
        t2m, m2t = metrics["text_to_motion"], metrics["motion_to_text"]
        print(
            f"{split} (N={metrics['num_samples']}): "
            f"T->G R@1 {100 * t2m['R@1']:.1f} MRR {100 * t2m['MRR']:.1f} | "
            f"G->T R@1 {100 * m2t['R@1']:.1f} MRR {100 * m2t['MRR']:.1f}"
        )


if __name__ == "__main__":
    main()
