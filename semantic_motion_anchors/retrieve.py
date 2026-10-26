"""Retrieve BEAT2 gestures for free-text queries with a trained model.

Used for the retrieval-augmented generation study: each query (e.g. a few words of context
around a RAG-Gesture query word) is embedded with the frozen text encoder, projected by the
transcript branch and matched against BEAT2 motion embeddings.

    python -m semantic_motion_anchors.retrieve --config configs/semantic_anchors.yaml --seed 42 \
        --queries queries.csv --query-column context --output outputs/retrievals.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from .config import apply_seed, load_config, manifest_path, run_dir
from .data import read_csv_rows, read_manifest, sanitize_text
from .models import load_trained_model
from .preprocess import encode_texts, load_text_encoder
from .train import embed_text_and_motion, get_device, load_stores, make_dataset, make_loader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--queries", required=True, type=Path, help="CSV with one query per row.")
    parser.add_argument("--query-column", default="context")
    parser.add_argument("--fallback-column", default=None, help="Used when the query column is empty.")
    parser.add_argument("--gallery-splits", nargs="+", default=["beat_train", "beat_val", "beat_test"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg = apply_seed(cfg, args.seed)
    device = get_device(cfg["training"].get("device", "auto"))
    model = load_trained_model(str(run_dir(cfg) / "checkpoints" / "best.pt"), cfg, device)

    queries = []
    for row in read_csv_rows(args.queries):
        text = sanitize_text(row.get(args.query_column)) or sanitize_text(row.get(args.fallback_column or ""))
        if text:
            queries.append((row, text))
    encoder_cfg = cfg["text_encoder"]
    raw = encode_texts(
        load_text_encoder(encoder_cfg["model_name"]),
        [text for _, text in queries],
        int(encoder_cfg["batch_size"]),
        encoder_cfg.get("transcript_prompt_name"),
    )
    with torch.no_grad():
        query_embeddings = model.encode_transcript(torch.from_numpy(raw).to(device)).cpu()

    rows = read_manifest(manifest_path(cfg))
    gallery = make_dataset(cfg, rows, load_stores(cfg, []), args.gallery_splits, [])
    loader = make_loader(gallery, int(cfg["training"]["eval_batch_size"]), shuffle=False)
    _, motion = embed_text_and_motion(model, loader, device)
    scores, indices = (query_embeddings @ motion.t()).topk(min(args.top_k, motion.size(0)), dim=1)

    anchors = {f"beat:{row['anchor_id']}": row for row in read_csv_rows(cfg["paths"]["beat2_anchors"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = None
        for (query_row, text), row_scores, row_indices in zip(queries, scores.tolist(), indices.tolist()):
            for rank, (score, index) in enumerate(zip(row_scores, row_indices), start=1):
                hit = gallery.rows[index]
                record = {
                    **query_row,
                    "query_text": text,
                    "rank": rank,
                    "score": f"{score:.6f}",
                    "retrieved_clip": hit["clip"],
                    "retrieved_split": hit["split"],
                    "retrieved_transcript": hit["transcript_text"],
                    "retrieved_anchor": anchors[hit["uid"]]["anchor"],
                }
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(record))
                    writer.writeheader()
                writer.writerow(record)
    print(f"Wrote top-{args.top_k} retrievals for {len(queries)} queries to {args.output}")


if __name__ == "__main__":
    main()
