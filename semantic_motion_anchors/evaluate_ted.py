"""TED-to-TED exact-pair retrieval with a BEAT2-trained model (Tables 5 and 12).

Queries are TED transcript snippets. Gallery gestures are represented either by the motion
encoder on TED motion (needs `paths.ted_motion`) or by the text description of the gesture's
lexicon entry, projected with the anchor projector and, for comparison, with the transcript
projector. Several TED gestures share a lexicon description or an interval, so exact-pair ranks
among such tied gallery items follow the (unstable) sort order, as in the paper's evaluation.

    python -m semantic_motion_anchors.evaluate_ted --config configs/semantic_anchors.yaml --seed 42
"""
from __future__ import annotations

import argparse
from typing import Any

import torch

from .config import apply_seed, load_config, manifest_path, run_dir, save_json
from .data import read_manifest
from .metrics import retrieval_metrics
from .models import load_trained_model
from .train import embed_text_and_motion, get_device, load_stores, make_dataset, make_loader


@torch.no_grad()
def project(encode, raw: torch.Tensor, device: torch.device, batch_size: int = 256) -> torch.Tensor:
    return torch.cat([encode(raw[i : i + batch_size].to(device)).cpu() for i in range(0, raw.size(0), batch_size)])


def both_directions(text: torch.Tensor, gallery: torch.Tensor) -> dict[str, Any]:
    return {"text_to_gesture": retrieval_metrics(text, gallery), "gesture_to_text": retrieval_metrics(gallery, text)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg = apply_seed(cfg, args.seed)
    device = get_device(cfg["training"].get("device", "auto"))
    out_dir = run_dir(cfg)
    model = load_trained_model(str(out_dir / "checkpoints" / "best.pt"), cfg, device)
    rows = read_manifest(manifest_path(cfg))
    stores = load_stores(cfg, [])
    ted_rows = [row for row in rows if row["split"] == "ted_eval"]
    uids = [row["uid"] for row in ted_rows]
    if not uids:
        raise RuntimeError("No TED rows in the manifest; set paths.ted_eval and rerun preprocess.")

    text = project(model.encode_transcript, stores["transcript"].stack(uids), device)
    description = stores["anchor"].stack(uids)  # TED anchor text is the lexicon description
    results: dict[str, Any] = {
        "num_samples": len(uids),
        "description_anchor_projection": both_directions(text, project(model.encode_anchor, description, device)),
        "description_transcript_projection": both_directions(
            text, project(model.encode_transcript, description, device)
        ),
    }
    if all(row["motion_archive_id"] for row in ted_rows):
        dataset = make_dataset(cfg, rows, stores, "ted_eval", [])
        loader = make_loader(dataset, int(cfg["training"]["eval_batch_size"]), shuffle=False)
        motion_text, motion = embed_text_and_motion(model, loader, device)
        results["motion_embedding"] = both_directions(motion_text, motion)

    save_json(out_dir / "metrics" / "ted_to_ted.json", results)
    for name, block in results.items():
        if isinstance(block, dict):
            t2g, g2t = block["text_to_gesture"], block["gesture_to_text"]
            print(
                f"{name}: T->G R@5 {100 * t2g['R@5']:.1f} MRR {100 * t2g['MRR']:.2f} | "
                f"G->T R@5 {100 * g2t['R@5']:.1f} MRR {100 * g2t['MRR']:.2f}"
            )


if __name__ == "__main__":
    main()
