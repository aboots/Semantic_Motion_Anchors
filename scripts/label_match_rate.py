"""Semantic label match rate of top-1 text->gesture retrieval on BEAT2 test (Table 3).

A query counts as matched when the top-1 retrieved gesture has the same intent category as
the ground-truth gesture. The per-query CSV lists the retrieved gestures' anchors, which is
how the qualitative examples (Tables 4 and 13) were selected.

    python scripts/label_match_rate.py --config configs/semantic_anchors.yaml \
        --runs Ours=semantic_anchors_seed42 Text=text_contrastive_seed42 Random=random_anchor_seed42
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from semantic_motion_anchors.config import load_config, manifest_path, run_dir
from semantic_motion_anchors.data import read_csv_rows, read_manifest
from semantic_motion_anchors.models import load_trained_model
from semantic_motion_anchors.train import embed_text_and_motion, get_device, load_stores, make_dataset, make_loader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="Any experiment config (for paths and data).")
    parser.add_argument("--runs", nargs="+", required=True, metavar="LABEL=RUN", help="Trained runs to compare.")
    parser.add_argument("--output", type=Path, default=Path("outputs/label_match_rate.csv"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["training"].get("device", "auto"))
    stores = load_stores(cfg, [])
    dataset = make_dataset(cfg, read_manifest(manifest_path(cfg)), stores, "beat_test", [])
    loader = make_loader(dataset, int(cfg["training"]["eval_batch_size"]), shuffle=False)
    anchors = {f"beat:{row['anchor_id']}": row for row in read_csv_rows(cfg["paths"]["beat2_anchors"])}
    uids = [row["uid"] for row in dataset.rows]
    labels = [anchors[uid]["intent_category"] for uid in uids]

    retrieved: dict[str, list[int]] = {}
    for spec in args.runs:
        name, _, run = spec.partition("=")
        run_cfg = load_config(run_dir(cfg, run) / "config.json")
        model = load_trained_model(str(run_dir(cfg, run) / "checkpoints" / "best.pt"), run_cfg, device)
        text, motion = embed_text_and_motion(model, loader, device)
        similarity = F.normalize(text, dim=-1, eps=1e-8) @ F.normalize(motion, dim=-1, eps=1e-8).t()
        retrieved[name] = torch.argsort(similarity, dim=1, descending=True)[:, 0].tolist()

    matches: dict[str, dict[str, list[bool]]] = {name: defaultdict(list) for name in retrieved}
    for name, top1 in retrieved.items():
        for query, gallery in enumerate(top1):
            matches[name][labels[query]].append(labels[gallery] == labels[query])

    categories = sorted(matches[next(iter(matches))], key=lambda c: -len(matches[next(iter(matches))][c]))
    names = list(retrieved)
    print("| Category | n | " + " | ".join(names) + " |")
    print("|---|---:|" + "---:|" * len(names))
    for category in categories:
        n = len(matches[names[0]][category])
        rates = [100.0 * sum(matches[name][category]) / n for name in names]
        print(f"| {category} | {n} | " + " | ".join(f"{rate:.1f}" for rate in rates) + " |")
    for title, aggregate in (
        ("Overall", lambda m: sum(sum(v) for v in m.values()) / sum(len(v) for v in m.values())),
        ("Macro average", lambda m: sum(sum(v) / len(v) for v in m.values()) / len(m)),
    ):
        print(f"| {title} | {len(uids)} | " + " | ".join(f"{100.0 * aggregate(matches[name]):.1f}" for name in names) + " |")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["clip", "intent_category", "transcript", "anchor"]
        for name in names:
            header += [f"{name}_retrieved_clip", f"{name}_retrieved_category", f"{name}_retrieved_anchor"]
        writer.writerow(header)
        for query, uid in enumerate(uids):
            row = [anchors[uid]["clip"], labels[query], dataset.rows[query]["transcript_text"], anchors[uid]["anchor"]]
            for name in names:
                hit = uids[retrieved[name][query]]
                row += [anchors[hit]["clip"], anchors[hit]["intent_category"], anchors[hit]["anchor"]]
            writer.writerow(row)
    print(f"Per-query retrievals: {args.output}")


if __name__ == "__main__":
    main()
