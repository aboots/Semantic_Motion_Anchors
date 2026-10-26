"""Recompute semantic-context metrics of saved cross-dataset rankings with another text encoder.

Rankings and intent-label metrics stay fixed; only the relevance r(q, g) between the TED query's
communicative intent and the retrieved BEAT2 gesture's intent is re-embedded (e.g. with BGE-M3,
an embedding family independent of the Qwen3 encoder used by the retrieval models).

    python scripts/rescore_semantic_context.py --config configs/semantic_anchors.yaml --seed 42 \
        --methods "Text Contrastive=text_contrastive" "Random Anchor=random_anchor_matched" \
                  "Proposed=semantic_anchors" --encoder BAAI/bge-m3 --pooling cls
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from semantic_motion_anchors.config import load_config, manifest_path
from semantic_motion_anchors.data import read_csv_rows, read_manifest, sanitize_text

BLOCKS = ("ted_transcript->beat_motion", "ted_transcript->beat_anchor")


@torch.no_grad()
def encode(model, tokenizer, texts: list[str], batch_size: int, pooling: str, device: torch.device) -> torch.Tensor:
    outputs = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[start : start + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        hidden = model(**tokens).last_hidden_state
        if pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        outputs.append(F.normalize(pooled.float(), dim=-1).cpu())
    return torch.cat(outputs)


def score(retrieved: torch.Tensor, relevance: torch.Tensor) -> dict[str, torch.Tensor]:
    top = relevance.gather(1, retrieved)
    discounts = 1.0 / torch.log2(torch.arange(10, dtype=torch.float32) + 2.0)
    dcg = (top * discounts).sum(dim=1)
    idcg = (torch.topk(relevance, k=10, dim=1).values * discounts).sum(dim=1)
    return {
        "best_cosine@5": top[:, :5].max(dim=1).values,
        "mean_cosine@10": top.mean(dim=1),
        "ndcg@10": torch.where(idcg > 0, dcg / idcg.clamp_min(1e-8), torch.zeros_like(dcg)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", required=True, metavar="LABEL=RUN")
    parser.add_argument("--reference", default="Proposed")
    parser.add_argument("--encoder", default="BAAI/bge-m3")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/runs"))
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    cfg = load_config(args.config)
    rows = read_manifest(manifest_path(cfg))
    beat_uids = [row["uid"] for row in rows if row["split"] == "beat_test"]
    ted_uids = [row["uid"] for row in rows if row["split"] == "ted_eval"]
    beat_intent = {f"beat:{r['anchor_id']}": sanitize_text(r["communicative_intent"]) for r in read_csv_rows(cfg["paths"]["beat2_anchors"])}
    ted_intent = {f"ted:{r['sample_id']}": sanitize_text(r["communicative_intent"]) for r in read_csv_rows(cfg["paths"]["ted_eval"])}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    model = AutoModel.from_pretrained(args.encoder).to(device).eval()
    query = encode(model, tokenizer, [ted_intent[uid] for uid in ted_uids], args.batch_size, args.pooling, device)
    gallery = encode(model, tokenizer, [beat_intent[uid] for uid in beat_uids], args.batch_size, args.pooling, device)
    relevance = (query @ gallery.t()).clamp_min(0.0)
    gallery_index = {uid: index for index, uid in enumerate(beat_uids)}

    methods = dict(spec.split("=", 1) for spec in args.methods)
    for block in BLOCKS:
        per_method = {}
        for label, run in methods.items():
            path = args.runs_dir / f"{run}_seed{args.seed}" / "cross_dataset" / f"{block.replace('->', '_to_')}__top10.json"
            ranking = {item["query"]: item["retrieved"] for item in json.loads(path.read_text())}
            retrieved = torch.tensor([[gallery_index[uid] for uid in ranking[q][:10]] for q in ted_uids])
            per_method[label] = score(retrieved, relevance)
        print(f"## {block} (semantic context with {args.encoder})")
        print("| Method | BestCos@5 | MeanCos@10 | Semantic nDCG@10 |\n|---|---:|---:|---:|")
        for label, values in per_method.items():
            print(f"| {label} | " + " | ".join(f"{100 * float(values[k].mean()):.1f}" for k in ("best_cosine@5", "mean_cosine@10", "ndcg@10")) + " |")
        for label, values in per_method.items():
            if label != args.reference:
                wins = {k: float(((per_method[args.reference][k] - values[k]) > 1e-8).float().mean()) for k in ("ndcg@10", "mean_cosine@10")}
                print(f"- {args.reference} win rate over {label}: nDCG@10 {100 * wins['ndcg@10']:.1f}, MeanCos@10 {100 * wins['mean_cosine@10']:.1f}")
        print()


if __name__ == "__main__":
    main()
