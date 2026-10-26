#!/usr/bin/env bash
# TED-to-TED (Tables 5, 12) and TED <-> BEAT2 (Table 6) evaluation of BEAT2-trained models.
# Table 6 and Tables 5/12 use seed 42; the multi-seed summaries cover the anchor controls.
set -euo pipefail
SEEDS="${SEEDS:-42 43 44 45 46 47 48}"
CONFIGS="configs/text_contrastive.yaml configs/controls/random_anchor_matched.yaml configs/semantic_anchors.yaml
         configs/controls/mismatched_anchor.yaml configs/controls/physical_only.yaml configs/controls/intent_only.yaml"

for seed in $SEEDS; do
  for config in $CONFIGS; do
    python -m semantic_motion_anchors.evaluate_ted --config "$config" --seed "$seed"
    python -m semantic_motion_anchors.evaluate_cross_dataset --config "$config" --seed "$seed"
  done
done

METHODS=("Text Contrastive=text_contrastive" "Random Anchor=random_anchor_matched" "Proposed=semantic_anchors")
python scripts/summarize_cross_dataset.py --seeds 42 --methods "${METHODS[@]}" \
  --blocks "ted_transcript->beat_motion" "ted_transcript->beat_anchor" --output outputs/tables/table6.md
python scripts/summarize_cross_dataset.py --seeds $SEEDS \
  --methods "${METHODS[@]}" "Mismatched Anchors=mismatched_anchor" "Intent Only=intent_only" "Physical Only=physical_only" \
  --blocks "ted_transcript->beat_motion" "ted_transcript->beat_anchor" "ted_transcript->beat_physical" \
           "beat_transcript->ted_motion" "beat_transcript->ted_physical" "beat_motion->ted_transcript" \
           "beat_physical->ted_transcript" "ted_motion->beat_transcript" "ted_physical->beat_transcript" \
           "ted_physical->beat_motion" "beat_physical->ted_motion" \
  --output outputs/tables/cross_dataset_seeds.md
