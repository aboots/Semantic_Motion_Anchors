"""Cross-dataset TED <-> BEAT2 retrieval with a BEAT2-trained model (Table 6, App. A.9).

There are no exact cross-dataset pairs, so rankings are scored against
  * the intent category (Acc@1, Hit@K, MRR, nDCG@K over the categories both datasets share), and
  * semantic-context similarity: frozen-encoder cosine between the query's and the retrieved
    gesture's communicative intent (BestCos@K, MeanCos@K, graded nDCG@K).
Blocks with a physical-form proxy additionally report physical-context similarity between the
physical-form descriptions of query and retrieved gesture.

Gesture representations: raw motion embeddings, the generated semantic anchor (BEAT2) and the
physical-form description (BEAT2 a_phys; TED gesture-lexicon description). Descriptions go
through the anchor projector of anchor-trained models and through the transcript projector of
models without anchor supervision.

    python -m semantic_motion_anchors.evaluate_cross_dataset --config configs/semantic_anchors.yaml --seed 42
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import apply_seed, load_config, manifest_path, run_dir, save_json, text_cache_dir
from .data import read_csv_rows, read_manifest, sanitize_text
from .evaluate_ted import project
from .models import load_trained_model
from .preprocess import cache_is_current, encode_texts, load_text_encoder, text_fingerprint
from .train import auxiliary_sources, embed_text_and_motion, get_device, load_stores, make_dataset, make_loader

KS = (1, 3, 5, 10)
LABEL_METRICS = ("acc@1", "hit@3", "hit@5", "hit@10", "mrr", "ndcg@3", "ndcg@5", "ndcg@10")
# (query dataset, query representation, gallery dataset, gallery representation)
BLOCKS = {
    "ted_transcript->beat_motion": ("ted", "transcript", "beat", "motion"),
    "ted_transcript->beat_anchor": ("ted", "transcript", "beat", "anchor"),
    "ted_transcript->beat_physical": ("ted", "transcript", "beat", "physical"),
    "beat_transcript->ted_motion": ("beat", "transcript", "ted", "motion"),
    "beat_transcript->ted_physical": ("beat", "transcript", "ted", "physical"),
    "beat_motion->ted_transcript": ("beat", "motion", "ted", "transcript"),
    "beat_physical->ted_transcript": ("beat", "physical", "ted", "transcript"),
    "ted_motion->beat_transcript": ("ted", "motion", "beat", "transcript"),
    "ted_physical->beat_transcript": ("ted", "physical", "beat", "transcript"),
    "ted_physical->beat_motion": ("ted", "physical", "beat", "motion"),
    "beat_physical->ted_motion": ("beat", "physical", "ted", "motion"),
}


def normalize_label(text: str) -> str:
    value = re.sub(r"\s+", " ", (text or "").strip().lower().replace("_", " ").replace("-", " "))
    return "symbolise" if value.startswith("symbolise (") else value


def default_projection(cfg: dict[str, Any]) -> str:
    """Descriptions use the anchor projector only if the model was trained with anchor terms."""
    return "anchor" if auxiliary_sources(cfg) or any(
        term["left"].startswith("random_") for term in cfg["training"]["losses"].values()
    ) else "transcript"


def cached_encoding(cfg: dict[str, Any], name: str, uids: list[str], texts: list[str], prompt: str | None) -> torch.Tensor:
    """Embed extra evaluation texts with the frozen text encoder, cached by their content."""
    encoder_cfg = cfg["text_encoder"]
    path = text_cache_dir(cfg) / f"{name}.npz"
    fingerprint = text_fingerprint(encoder_cfg["model_name"], prompt, uids, texts)
    if cache_is_current(path, fingerprint):
        return torch.from_numpy(np.load(path, allow_pickle=False)["embedding"].astype(np.float32))
    model = load_text_encoder(encoder_cfg["model_name"])
    embeddings = encode_texts(model, texts, int(encoder_cfg["batch_size"]), prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, uid=np.asarray(uids), embedding=embeddings, fingerprint=np.asarray(fingerprint))
    return torch.from_numpy(embeddings)


def similarity_metrics(
    query_refs: torch.Tensor, gallery_refs: torch.Tensor, ranked: torch.Tensor
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Graded relevance r(q, g) = max(0, cos) between reference texts (Eqs. 11-17)."""
    relevance = (F.normalize(query_refs, dim=-1, eps=1e-8) @ F.normalize(gallery_refs, dim=-1, eps=1e-8).t()).clamp(min=0.0)
    summary: dict[str, float] = {}
    per_query: dict[str, np.ndarray] = {}
    top1 = relevance[torch.arange(ranked.shape[0]), ranked[:, 0]]
    summary["cosine@1"], per_query["cosine@1"] = float(top1.mean()), top1.numpy()
    for k in KS:
        values = relevance.gather(1, ranked[:, :k])
        best, mean = values.max(dim=1).values, values.mean(dim=1)
        discounts = 1.0 / torch.log2(torch.arange(k, dtype=torch.float32) + 2.0)
        dcg = (values * discounts.unsqueeze(0)).sum(dim=1)
        ideal = torch.topk(relevance, k=min(k, relevance.shape[1]), dim=1).values
        idcg = (ideal * discounts[: ideal.shape[1]].unsqueeze(0)).sum(dim=1)
        ndcg = torch.where(idcg > 0, dcg / idcg.clamp_min(1e-8), torch.zeros_like(dcg))
        for name, array in ((f"best_cosine@{k}", best), (f"mean_cosine@{k}", mean), (f"ndcg@{k}", ndcg)):
            summary[name], per_query[name] = float(array.mean()), array.numpy()
    return summary, per_query


def binary_metrics(relevant: list[bool]) -> dict[str, float]:
    """Exact intent-category relevance (Eqs. 18-24)."""
    first = next((rank for rank, hit in enumerate(relevant, start=1) if hit), None)
    result = {
        "acc@1": float(bool(relevant) and relevant[0]),
        "hit@3": float(any(relevant[:3])),
        "hit@5": float(any(relevant[:5])),
        "hit@10": float(any(relevant[:10])),
        "mrr": 0.0 if first is None else 1.0 / first,
    }
    total = sum(relevant)
    for k in (3, 5, 10):
        dcg = sum(1.0 / math.log2(rank + 1) for rank, hit in enumerate(relevant[:k], start=1) if hit)
        ideal = min(total, k)
        result[f"ndcg@{k}"] = dcg / sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal + 1)) if ideal else 0.0
    return result


def label_metrics(
    query_labels: list[str], gallery_labels: list[str], ranked: torch.Tensor, shared: set[str]
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[int]]:
    rows, kept = [], []
    for index, label in enumerate(query_labels):
        if label in shared:
            rows.append({"label": label, **binary_metrics([gallery_labels[i] == label for i in ranked[index].tolist()])})
            kept.append(index)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    micro = {key: float(np.mean([row[key] for row in rows])) for key in LABEL_METRICS}
    macro = {
        key: float(np.mean([np.mean([row[key] for row in group]) for group in by_label.values()]))
        for key in LABEL_METRICS
    }
    per_query = {key: np.asarray([row[key] for row in rows], dtype=np.float32) for key in LABEL_METRICS}
    return {"included_queries": len(rows), "micro": micro, "macro": macro}, per_query, kept


def evaluate_block(
    name: str,
    query: dict[str, Any],
    gallery: dict[str, Any],
    shared_labels: set[str],
    out_dir: Path,
    top_k: int,
) -> dict[str, Any]:
    scores = F.normalize(query["embedding"], dim=-1, eps=1e-8) @ F.normalize(gallery["embedding"], dim=-1, eps=1e-8).t()
    # TED gestures can share a lexicon description or an interval; ties keep gallery order.
    ranked = torch.argsort(scores, dim=1, descending=True, stable=True)
    result: dict[str, Any] = {"num_queries": len(query["uids"]), "num_gallery": len(gallery["uids"]), "semantic": {}}
    per_query: dict[str, np.ndarray] = {"query_uid": np.asarray(query["uids"])}
    for ref_name in ("semantic_context", "physical_context"):
        if ref_name in query["refs"] and ref_name in gallery["refs"]:
            summary, values = similarity_metrics(query["refs"][ref_name], gallery["refs"][ref_name], ranked)
            result["semantic"][ref_name] = summary
            per_query.update({f"{ref_name}/{key}": value for key, value in values.items()})
    result["label"], label_values, kept = label_metrics(query["labels"], gallery["labels"], ranked, shared_labels)
    per_query["label_query_uid"] = np.asarray([query["uids"][i] for i in kept])
    per_query.update({f"label/{key}": value for key, value in label_values.items()})

    np.savez_compressed(out_dir / f"{name.replace('->', '_to_')}__per_query.npz", **per_query)
    save_json(
        out_dir / f"{name.replace('->', '_to_')}__top{top_k}.json",
        [
            {"query": query["uids"][i], "retrieved": [gallery["uids"][j] for j in ranked[i, :top_k].tolist()]}
            for i in range(ranked.shape[0])
        ],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--projection", choices=["anchor", "transcript"], default=None,
                        help="Projector for description proxies (default: by training objective).")
    parser.add_argument("--blocks", nargs="+", default=list(BLOCKS), choices=list(BLOCKS))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg = apply_seed(cfg, args.seed)
    device = get_device(cfg["training"].get("device", "auto"))
    out_dir = run_dir(cfg) / "cross_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_trained_model(str(run_dir(cfg) / "checkpoints" / "best.pt"), cfg, device)
    projection = args.projection or default_projection(cfg)
    encode_description = model.encode_anchor if projection == "anchor" else model.encode_transcript
    batch_size = int(cfg["training"]["eval_batch_size"])

    rows = read_manifest(manifest_path(cfg))
    stores = load_stores(cfg, ["physical_motion_llm", "semantic_intent_llm"])
    beat_labels = {f"beat:{row['anchor_id']}": row["intent_category"] for row in read_csv_rows(cfg["paths"]["beat2_anchors"])}
    ted_info = {f"ted:{row['sample_id']}": row for row in read_csv_rows(cfg["paths"]["ted_eval"])}

    domains: dict[str, dict[str, Any]] = {}
    for domain, split in (("beat", "beat_test"), ("ted", "ted_eval")):
        split_rows = [row for row in rows if row["split"] == split]
        uids = [row["uid"] for row in split_rows]
        info: dict[str, Any] = {"uids": uids, "reps": {}}
        info["reps"]["transcript"] = project(model.encode_transcript, stores["transcript"].stack(uids), device, batch_size)
        info["reps"]["physical"] = project(encode_description, stores["physical_motion_llm"].stack(uids), device, batch_size)
        if all(row["motion_path"] or row["motion_archive_id"] for row in split_rows):
            loader = make_loader(make_dataset(cfg, rows, stores, split, []), batch_size, shuffle=False)
            info["reps"]["motion"] = embed_text_and_motion(model, loader, device)[1]
        if domain == "beat":
            info["labels"] = [normalize_label(beat_labels.get(uid, "")) for uid in uids]
            info["refs"] = {"semantic_context": stores["semantic_intent_llm"].stack(uids)}
            anchors = [row["anchor_text"] for row in split_rows]
            raw_anchor = (
                stores["anchor"].stack(uids)
                if projection == "anchor"
                else cached_encoding(cfg, "beat_test_anchor_as_query", uids, anchors, cfg["text_encoder"]["transcript_prompt_name"])
            )
            info["reps"]["anchor"] = project(encode_description, raw_anchor, device, batch_size)
        else:
            intents = [sanitize_text(ted_info[uid]["communicative_intent"]) for uid in uids]
            info["labels"] = [normalize_label(ted_info[uid]["intent_category"]) for uid in uids]
            info["refs"] = {"semantic_context": cached_encoding(cfg, "ted_communicative_intent", uids, intents, None)}
        info["refs"]["physical_context"] = stores["physical_motion_llm"].stack(uids)
        domains[domain] = info

    shared_labels = {label for label in domains["beat"]["labels"] if label} & {label for label in domains["ted"]["labels"] if label}
    summary: dict[str, Any] = {"projection": projection, "shared_labels": sorted(shared_labels), "blocks": {}}
    for name in args.blocks:
        query_domain, query_rep, gallery_domain, gallery_rep = BLOCKS[name]
        q, g = domains[query_domain], domains[gallery_domain]
        if query_rep not in q["reps"] or gallery_rep not in g["reps"]:
            print(f"Skipping {name}: no {query_domain if query_rep not in q['reps'] else gallery_domain} motion.")
            continue
        summary["blocks"][name] = evaluate_block(
            name,
            {"uids": q["uids"], "embedding": q["reps"][query_rep], "labels": q["labels"], "refs": q["refs"]},
            {"uids": g["uids"], "embedding": g["reps"][gallery_rep], "labels": g["labels"], "refs": g["refs"]},
            shared_labels,
            out_dir,
            args.top_k,
        )
        block = summary["blocks"][name]
        label, context = block["label"]["micro"], block["semantic"]["semantic_context"]
        print(
            f"{name}: Acc@1 {100 * label['acc@1']:.1f} MRR {100 * label['mrr']:.1f} "
            f"nDCG@10 {100 * label['ndcg@10']:.1f} | BestCos@5 {100 * context['best_cosine@5']:.1f} "
            f"MeanCos@10 {100 * context['mean_cosine@10']:.1f} sem-nDCG@10 {100 * context['ndcg@10']:.1f}"
        )
    save_json(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
