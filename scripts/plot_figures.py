"""Plot the loss-weight sensitivity (Figs. 2, 4, 5) and cumulative rank distribution (Fig. 6).

    python scripts/plot_figures.py sensitivity --seed 44 --bridge-seed 42
    python scripts/plot_figures.py rank-cdf --runs "Text Contrastive=text_contrastive_seed44" \
        "Ours=semantic_anchors_seed44"
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

WEIGHT_RUN = re.compile(r"semantic_anchors_ls(\d{3})_lp(\d{3})_seed(\d+)$")
BRIDGE_RUN = re.compile(r"semantic_anchors_lb(\d{3})_seed(\d+)$")
DEFAULT_WEIGHTS = {"ls": 0.15, "lp": 0.03, "lb": 0.02}


def mean_mrr(run: Path) -> float:
    return 100.0 * json.loads((run / "metrics" / "beat_test.json").read_text())["mean_mrr"]


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.with_suffix('.pdf')}")


def line(points: dict[float, float], xlabel: str, title: str, path: Path) -> None:
    xs = sorted(points)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(xs, [points[x] for x in xs], marker="o", color="#4c78a8", linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean MRR (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    save(fig, path)


def sensitivity(args: argparse.Namespace) -> None:
    grid: dict[tuple[float, float], float] = {}
    bridge: dict[float, float] = {}
    for run in sorted(p for p in args.runs_dir.iterdir() if p.is_dir()):
        if (match := WEIGHT_RUN.match(run.name)) and int(match.group(3)) == args.seed:
            grid[(int(match.group(1)) / 100, int(match.group(2)) / 100)] = mean_mrr(run)
        elif (match := BRIDGE_RUN.match(run.name)) and int(match.group(2)) == args.bridge_seed:
            bridge[int(match.group(1)) / 100] = mean_mrr(run)
    # The default weights are the main semantic-anchor runs.
    grid.setdefault((DEFAULT_WEIGHTS["ls"], DEFAULT_WEIGHTS["lp"]), mean_mrr(args.runs_dir / f"semantic_anchors_seed{args.seed}"))
    bridge.setdefault(DEFAULT_WEIGHTS["lb"], mean_mrr(args.runs_dir / f"semantic_anchors_seed{args.bridge_seed}"))

    intent = np.array([key[0] for key in grid])
    physical = np.array([key[1] for key in grid])
    score = np.array(list(grid.values()))
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    contour = ax.tricontourf(mtri.Triangulation(physical, intent), score, levels=14, cmap="viridis")
    ax.scatter(physical, intent, c=score, cmap="viridis", edgecolors="white", s=40)
    fig.colorbar(contour, ax=ax, label="Mean MRR (%)")
    ax.set_xlabel(r"Motion-physical weight $\lambda_p$")
    ax.set_ylabel(r"Transcript-intent weight $\lambda_s$")
    ax.set_title("Joint impact of auxiliary weights")
    save(fig, args.output_dir / "loss_weight_heatmap")

    line({lp: s for (ls, lp), s in grid.items() if np.isclose(ls, DEFAULT_WEIGHTS["ls"])},
         r"Motion-physical weight $\lambda_p$", r"$\lambda_s$ = 0.15", args.output_dir / "sweep_lambda_p")
    line({ls: s for (ls, lp), s in grid.items() if np.isclose(lp, DEFAULT_WEIGHTS["lp"])},
         r"Transcript-intent weight $\lambda_s$", r"$\lambda_p$ = 0.03", args.output_dir / "sweep_lambda_s")
    line(bridge, r"Bridge weight $\lambda_b$", r"$\lambda_p$ = 0.03, $\lambda_s$ = 0.15", args.output_dir / "sweep_lambda_b")


def rank_cdf(args: argparse.Namespace) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for spec in args.runs:
        label, _, run = spec.partition("=")
        ranks = np.asarray(json.loads((args.runs_dir / run / "metrics" / "beat_test.json").read_text())["ranks"]["text_to_motion"])
        ranks = np.sort(ranks + 1)
        ax.step(ranks, np.arange(1, ranks.size + 1) / ranks.size, where="post", label=label, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("True-match rank (text to gesture)")
    ax.set_ylabel("Cumulative fraction")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, args.output_dir / "rank_cdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    commands = parser.add_subparsers(dest="command", required=True)
    sweep = commands.add_parser("sensitivity")
    sweep.add_argument("--seed", type=int, default=44, help="Seed of the lambda_s/lambda_p grid.")
    sweep.add_argument("--bridge-seed", type=int, default=42, help="Seed of the lambda_b sweep.")
    cdf = commands.add_parser("rank-cdf")
    cdf.add_argument("--runs", nargs="+", required=True, metavar="LABEL=RUN")
    args = parser.parse_args()
    if args.command == "sensitivity":
        sensitivity(args)
    else:
        rank_cdf(args)


if __name__ == "__main__":
    main()
