#!/usr/bin/env bash
# Loss-weight sensitivity of the semantic-anchor objective (Figs. 2, 4 and 5).
# Requires the Text Contrastive models of GRID_SEED and BRIDGE_SEED (scripts/run_beat2.sh).
set -euo pipefail
GRID_SEED="${GRID_SEED:-44}"
BRIDGE_SEED="${BRIDGE_SEED:-42}"
# lambda_s:lambda_p in hundredths; the main setting 15:3 is the semantic_anchors run itself.
GRID="0:3 0:15 3:3 3:15 5:3 5:5 10:3 10:10 10:20 10:30 10:50 15:0 15:5 15:10 15:15 15:20 15:30 15:40 15:50
      20:3 20:10 20:20 30:3 30:10 30:30 40:3 40:40 50:3 50:10 50:50"
BRIDGE="0 5 10 15 20 30 40 50"  # lambda_b in hundredths; 2 is the main setting

weight() { printf '%d.%02d' $(($1 / 100)) $(($1 % 100)); }

run() {
  local seed=$1; shift
  python -m semantic_motion_anchors.train --config configs/semantic_anchors.yaml --seed "$seed" --set "$@"
  python -m semantic_motion_anchors.evaluate --config configs/semantic_anchors.yaml --seed "$seed" --splits beat_test --set "$@"
}

for pair in $GRID; do
  ls=${pair%:*}; lp=${pair#*:}
  run "$GRID_SEED" "name=$(printf 'semantic_anchors_ls%03d_lp%03d' "$ls" "$lp")" \
    "training.losses.intent_transcript.weight=$(weight "$ls")" \
    "training.losses.physical_motion.weight=$(weight "$lp")"
done
for lb in $BRIDGE; do
  run "$BRIDGE_SEED" "name=$(printf 'semantic_anchors_lb%03d' "$lb")" \
    "training.losses.physical_intent_bridge.weight=$(weight "$lb")"
done
python scripts/plot_figures.py sensitivity --seed "$GRID_SEED" --bridge-seed "$BRIDGE_SEED"
