"""Aggregate BEAT2 test retrieval over training seeds (Tables 1, 2 and 11, anchor controls).

    python scripts/summarize_beat2.py --runs-dir outputs/runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

SEEDS = (42, 43, 44, 45, 46, 47, 48)
METRICS = ("R@1", "R@5", "R@10", "MRR")
DIRECTIONS = (("motion_to_text", "Gesture->Text"), ("text_to_motion", "Text->Gesture"))
MAIN_TABLE = (
    ("GestureDiffuCLIP", "gesturediffuclip"),
    ("TMR", "tmr"),
    ("JEGAL", "jegal"),
    ("Text Contrastive", "text_contrastive"),
    ("Text Contrastive w Semantic Anchors (Ours)", "semantic_anchors"),
)
ANCHOR_TABLE = (
    ("No Anchor", "text_contrastive"),
    ("Random Anchor", "random_anchor"),
    ("Semantic Anchors (Ours)", "semantic_anchors"),
)
CONTROL_TABLE = (
    ("No Anchor", "text_contrastive"),
    ("Random Anchors", "random_anchor"),
    ("Mismatched Anchors", "mismatched_anchor"),
    ("Physical Only", "physical_only"),
    ("Intent Only", "intent_only"),
    ("Proposed", "semantic_anchors"),
)
CONTROL_COMPARISONS = (
    ("Random - No Anchor", "random_anchor", "text_contrastive"),
    ("Mismatched - Random", "mismatched_anchor", "random_anchor"),
    ("Proposed - Mismatched", "semantic_anchors", "mismatched_anchor"),
    ("Proposed - Random", "semantic_anchors", "random_anchor"),
    ("Proposed - Physical Only", "semantic_anchors", "physical_only"),
    ("Intent Only - Random", "intent_only", "random_anchor"),
)


def load_runs(runs_dir: Path, run: str, seeds: tuple[int, ...]) -> list[dict]:
    results = []
    for seed in seeds:
        path = runs_dir / f"{run}_seed{seed}" / "metrics" / "beat_test.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run train.py and evaluate.py for seed {seed}.")
        results.append(json.loads(path.read_text()))
    return results


def values(results: list[dict], direction: str, metric: str) -> np.ndarray:
    return np.array([100.0 * result[direction][metric] for result in results])


def average_mrr(results: list[dict]) -> np.ndarray:
    return np.array([50.0 * (r["text_to_motion"]["MRR"] + r["motion_to_text"]["MRR"]) for r in results])


def mean_ci(array: np.ndarray) -> tuple[float, float, float]:
    margin = stats.t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / np.sqrt(len(array))
    return float(array.mean()), float(array.mean() - margin), float(array.mean() + margin)


def marker(p: float) -> str:
    return "†" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def retrieval_table(title: str, rows: tuple, reference: str, compared: str, runs: dict[str, list[dict]]) -> list[str]:
    header = " | ".join(f"{name} {metric}" for _, name in DIRECTIONS for metric in METRICS)
    lines = [f"## {title}", "", f"| Method | {header} |", "|---" + "|---:" * 8 + "|"]
    for label, run in rows:
        cells = []
        for direction, _ in DIRECTIONS:
            for metric in METRICS:
                mean = values(runs[run], direction, metric).mean()
                tag = ""
                if run == compared:
                    _, p = stats.ttest_rel(values(runs[compared], direction, metric), values(runs[reference], direction, metric))
                    tag = marker(p)
                cells.append(f"{mean:.1f}{tag}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", f"Markers: paired t-test of `{compared}` against `{reference}` over seeds; * p<0.05, ** p<0.01, † p<0.001.", ""]
    return lines


def control_table(runs: dict[str, list[dict]]) -> list[str]:
    lines = ["## Anchor controls (average MRR over both directions)", "", "| Method | Avg. MRR [95% CI] |", "|---|---:|"]
    for label, run in CONTROL_TABLE:
        mean, low, high = mean_ci(average_mrr(runs[run]))
        lines.append(f"| {label} | {mean:.2f} [{low:.2f}, {high:.2f}] |")
    lines += ["", "| Paired difference | Mean [95% CI] | Cohen's d_z | p |", "|---|---:|---:|---:|"]
    for label, left, right in CONTROL_COMPARISONS:
        diff = average_mrr(runs[left]) - average_mrr(runs[right])
        mean, low, high = mean_ci(diff)
        _, p = stats.ttest_rel(average_mrr(runs[left]), average_mrr(runs[right]))
        lines.append(f"| {label} | {mean:+.2f} [{low:+.2f}, {high:+.2f}] | {diff.mean() / diff.std(ddof=1):.2f} | {p:.4f} |")
    return lines + [""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", default="outputs/runs", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--tables", nargs="+", default=["main", "anchor", "controls"], choices=["main", "anchor", "controls"])
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown file.")
    args = parser.parse_args()

    needed = {"main": MAIN_TABLE, "anchor": ANCHOR_TABLE, "controls": CONTROL_TABLE}
    run_names = {run for table in args.tables for _, run in needed[table]}
    runs = {run: load_runs(args.runs_dir, run, tuple(args.seeds)) for run in sorted(run_names)}

    lines = [f"Seeds: {', '.join(map(str, args.seeds))}. BEAT2 test gallery N={runs[next(iter(runs))][0]['num_samples']}.", ""]
    if "main" in args.tables:
        lines += retrieval_table("Table 1: BEAT2 retrieval", MAIN_TABLE, "text_contrastive", "semantic_anchors", runs)
    if "anchor" in args.tables:
        lines += retrieval_table("Tables 2/11: anchor content", ANCHOR_TABLE, "random_anchor", "semantic_anchors", runs)
    if "controls" in args.tables:
        lines += control_table(runs)
    text = "\n".join(lines)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
