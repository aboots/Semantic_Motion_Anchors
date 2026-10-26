#!/usr/bin/env bash
# Train and evaluate every BEAT2 model of Tables 1-3 and 11 and the anchor controls, for all seeds.
# Usage: bash scripts/run_beat2.sh            (SEEDS="42" bash scripts/run_beat2.sh for one seed)
set -euo pipefail
SEEDS="${SEEDS:-42 43 44 45 46 47 48}"

run() {
  python -m semantic_motion_anchors.train --config "$1" --seed "$2"
  python -m semantic_motion_anchors.evaluate --config "$1" --seed "$2"
}

for seed in $SEEDS; do
  for config in configs/text_contrastive.yaml configs/baselines/*.yaml; do
    run "$config" "$seed"
  done
  # Anchor-supervised models start from the Text Contrastive model of the same seed.
  for config in configs/semantic_anchors.yaml configs/controls/*.yaml; do
    run "$config" "$seed"
  done
done
