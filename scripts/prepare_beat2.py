"""Cut BEAT2 (English v2.0.0) into gesture clips with transcripts.

For every gesture interval in `sem/*.txt`, the SMPL-X sequence in `smplxflame_30/` is sliced
with a +-5 s context window and saved as `<sequence>---g<index>---<type>---<label>---w5s.npz`.
The per-sequence index counts all four BEAT2 gesture types (beat, deictic, iconic,
metaphoric), so clip names match the ids in the SEMANTIX anchor file. Words from the
TextGrid `words` tier that overlap the windowed interval form the transcript query.

    python scripts/prepare_beat2.py --beat2-dir data/beat2/beat_english_v2.0.0 \
        --output-dir data/beat2/clips_w5s
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np

GESTURE_TYPES = ("metaphoric", "iconic", "beat", "deictic")
FPS = 30
METADATA_FIELDS = [
    "output_filename",
    "sample_id",
    "sequence_id",
    "target_gesture",
    "gesture_label_original",
    "gesture_index",
    "start_sec_original",
    "end_sec_original",
    "start_sec_windowed",
    "end_sec_windowed",
    "start_frame_windowed",
    "end_frame_windowed",
    "transcript_original",
    "transcript_windowed",
]


def read_textgrid_words(path: Path) -> list[tuple[float, float, str]]:
    """Return (start, end, word) intervals of the `words` tier."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    words: list[tuple[float, float, str]] = []
    in_words_tier = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if 'name = "words"' in line:
            in_words_tier = True
        elif in_words_tier and line.strip().startswith("item ["):
            break
        elif in_words_tier and re.match(r"\s*intervals \[\d+\]:", line):
            xmin = re.search(r"xmin\s*=\s*([\d.]+)", lines[index + 1] if index + 1 < len(lines) else "")
            xmax = re.search(r"xmax\s*=\s*([\d.]+)", lines[index + 2] if index + 2 < len(lines) else "")
            text = re.search(r'text\s*=\s*"([^"]*)"', lines[index + 3] if index + 3 < len(lines) else "")
            if xmin and xmax and text:
                words.append((float(xmin.group(1)), float(xmax.group(1)), text.group(1)))
            index += 4
            continue
        index += 1
    return sorted(words, key=lambda word: word[0])


def words_between(words: list[tuple[float, float, str]], start: float, end: float) -> str:
    return " ".join(text.strip() for begin, finish, text in words if begin < end and finish > start and text.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--beat2-dir", required=True, help="Path to beat_english_v2.0.0/.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--include-beat", action="store_true", help="Also save clips of beat gestures.")
    args = parser.parse_args()

    beat2_dir = Path(args.beat2_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_types = set(GESTURE_TYPES) if args.include_beat else set(GESTURE_TYPES) - {"beat"}
    window = float(args.window_sec)
    suffix = f"---w{int(window)}s"

    rows: list[dict[str, object]] = []
    for sem_file in sorted(os.listdir(beat2_dir / "sem")):
        if not sem_file.endswith(".txt"):
            continue
        sequence = Path(sem_file).stem
        motion_path = beat2_dir / "smplxflame_30" / f"{sequence}.npz"
        if not motion_path.exists():
            continue
        motion = np.load(motion_path, allow_pickle=True)
        num_frames = motion["poses"].shape[0]
        words = read_textgrid_words(beat2_dir / "textgrid" / f"{sequence}.TextGrid")
        gesture_index = 0
        for line in (beat2_dir / "sem" / sem_file).read_text().splitlines():
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            label = parts[0]
            target = next((gesture for gesture in GESTURE_TYPES if gesture in label), None)
            if target is None:
                continue
            try:
                start, end = float(parts[1]), float(parts[2])
            except ValueError:
                continue
            gesture_index += 1
            if target not in saved_types:
                continue

            start_w = max(0.0, start - window)
            end_w = min(num_frames / float(FPS), end + window)
            end_frame = min(int(end_w * FPS), num_frames)
            start_frame = min(int(start_w * FPS), end_frame)
            clip = {
                key: motion[key][start_frame:end_frame]
                if motion[key].ndim > 0 and motion[key].shape[0] == num_frames
                else motion[key]
                for key in motion.files
            }
            filename = f"{sequence}---g{gesture_index:04d}---{target}---{label}{suffix}.npz"
            np.savez(output_dir / filename, **clip)
            rows.append(
                {
                    "output_filename": filename,
                    "sample_id": sequence,
                    "sequence_id": "_".join(sequence.split("_")[2:]),
                    "target_gesture": target,
                    "gesture_label_original": label,
                    "gesture_index": gesture_index,
                    "start_sec_original": start,
                    "end_sec_original": end,
                    "start_sec_windowed": start_w,
                    "end_sec_windowed": end_w,
                    "start_frame_windowed": start_frame,
                    "end_frame_windowed": end_frame,
                    "transcript_original": words_between(words, start, end),
                    "transcript_windowed": words_between(words, start_w, end_w),
                }
            )
        print(f"{sequence}: {gesture_index} gestures")

    with (output_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} clips and metadata.csv to {output_dir}")


if __name__ == "__main__":
    main()
