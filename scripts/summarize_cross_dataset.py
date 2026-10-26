"""Summarize cross-dataset retrieval over methods and seeds (Table 6 and its multi-seed variants).

Reports the mean (with a 95% Student-t interval when several seeds are given) of each metric
and the per-query win rate of `--reference` over every other method.

    python scripts/summarize_cross_dataset.py --seeds 42 \
        --methods "Text Contrastive=text_contrastive" "Random Anchor=random_anchor_matched" \
                  "Proposed=semantic_anchors" \
        --blocks ted_transcript->beat_motion ted_transcript->beat_anchor
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

METRICS = (
    ("Acc@1", "label", "acc@1"),
    ("Hit@5", "label", "hit@5"),
    ("Hit@10", "label", "hit@10"),
    ("Label MRR", "label", "mrr"),
    ("Label nDCG@10", "label", "ndcg@10"),
    ("BestCos@5", "semantic_context", "best_cosine@5"),
    ("MeanCos@10", "semantic_context", "mean_cosine@10"),
    ("Semantic nDCG@10", "semantic_context", "ndcg@10"),
    ("Physical MeanCos@10", "physical_context", "mean_cosine@10"),
    ("Physical nDCG@10", "physical_context", "ndcg@10"),
)
WIN_METRICS = (("semantic_context", "ndcg@10"), ("semantic_context", "mean_cosine@10"))


def metric_value(block: dict, family: str, key: str) -> float:
    return block["label"]["micro"][key] if family == "label" else block["semantic"][family][key]


def fmt(values: list[float]) -> str:
    array = 100.0 * np.asarray(values)
    if len(array) == 1:
        return f"{array[0]:.1f}"
    margin = stats.t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / np.sqrt(len(array))
    return f"{array.mean():.2f} [{array.mean() - margin:.2f}, {array.mean() + margin:.2f}]"


def per_query(run: Path, block: str, family: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(run / "cross_dataset" / f"{block.replace('->', '_to_')}__per_query.npz")
    return payload["query_uid"], payload[f"{family}/{key}"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--methods", nargs="+", required=True, metavar="LABEL=RUN")
    parser.add_argument("--reference", default="Proposed", help="Method whose win rate is reported.")
    parser.add_argument("--blocks", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    methods = dict(spec.split("=", 1) for spec in args.methods)
    runs = {label: [args.runs_dir / f"{run}_seed{seed}" for seed in args.seeds] for label, run in methods.items()}
    summaries = {
        label: [json.loads((path / "cross_dataset" / "summary.json").read_text()) for path in paths]
        for label, paths in runs.items()
    }
    lines = [f"Seeds: {', '.join(map(str, args.seeds))}.", ""]
    for block in args.blocks:
        if not all(block in s["blocks"] for items in summaries.values() for s in items):
            print(f"Skipping {block}: not evaluated for every run (TED motion missing?).")
            continue
        available = [m for m in METRICS if m[1] in summaries[args.reference][0]["blocks"][block].get("semantic", {}) or m[1] == "label"]
        lines += [f"## {block}", "", "| Method | " + " | ".join(m[0] for m in available) + " |", "|---" + "|---:" * len(available) + "|"]
        for label, items in summaries.items():
            cells = [fmt([metric_value(s["blocks"][block], family, key) for s in items]) for _, family, key in available]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
        for label in methods:
            if label == args.reference:
                continue
            cells = []
            for family, key in WIN_METRICS:
                rates = []
                for reference_run, baseline_run in zip(runs[args.reference], runs[label]):
                    uid_a, a = per_query(reference_run, block, family, key)
                    uid_b, b = per_query(baseline_run, block, family, key)
                    if not np.array_equal(uid_a, uid_b):
                        raise ValueError(f"Query order differs between {reference_run} and {baseline_run}")
                    rates.append(float(((a - b) > 1e-8).mean()))
                cells.append(f"{family} {key}: {fmt(rates)}")
            lines.append(f"- {args.reference} win rate over {label}: " + "; ".join(cells))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
