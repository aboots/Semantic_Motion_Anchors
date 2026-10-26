"""Split each semantic motion anchor into its physical-form and communicative-intent parts (App. A.7).

The unified anchor is passed to Qwen3-8B with a zero-shot prompt that returns a JSON object;
decoding is greedy with a budget of 160 new tokens. Rows that already have a split are skipped,
so the script can be resumed.

    python anchors/split_anchors.py --input anchors.csv --output anchors_split.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

SYSTEM_PROMPT = """You split gesture descriptions into exactly two aspects.

Return JSON with exactly these keys:
- physical_motion
- semantic_intent

Rules:
- physical_motion: only the visible body movement, hand shape, position, direction, timing, posture, and spatial form.
- semantic_intent: only the communicative meaning, emphasis, discourse function, emotion, or concept conveyed by the gesture.
- Keep both fields concise but faithful.
- Do not repeat the same content in both fields.
- If semantic meaning is unclear, set semantic_intent to an empty string.
- Output valid JSON only. No markdown.
"""


def user_prompt(description: str) -> str:
    return (
        "Split this gesture description into physical motion and semantic intent.\n"
        'Return only a JSON object like {"physical_motion":"...","semantic_intent":"..."}.\n'
        f"Description: {description}"
    )


def parse_split(response: str) -> tuple[str, str]:
    """Return (physical_motion, semantic_intent) from the last JSON object in `response`."""
    decoder = json.JSONDecoder()
    candidates = []
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, flags=re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    for index, char in enumerate(response):
        if char == "{":
            try:
                payload, end = decoder.raw_decode(response[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                candidates.append(response[index : index + end])
    if not candidates:
        raise ValueError(f"No JSON object in model output: {response[:200]!r}")
    payload = json.loads(candidates[-1])
    return str(payload.get("physical_motion", "")).strip(), str(payload.get("semantic_intent", "")).strip()


def generate(model, tokenizer, descriptions: list[str], max_new_tokens: int) -> list[str]:
    import torch

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt(text)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for text in descriptions
    ]
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    # Left padding: every generated sequence starts after the full padded prompt width.
    width = inputs["input_ids"].shape[1]
    return [tokenizer.decode(row[width:], skip_special_tokens=False) for row in outputs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path, help="CSV with an anchor column.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--anchor-column", default="anchor")
    parser.add_argument("--id-column", default="anchor_id")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--save-every", type=int, default=200)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    done = {}
    if args.output.exists():
        with args.output.open(encoding="utf-8", newline="") as handle:
            done = {row[args.id_column]: row for row in csv.DictReader(handle) if row.get("split_status") == "ok"}
    for row in rows:
        previous = done.get(row[args.id_column], {})
        row["physical_form"] = previous.get("physical_form", "")
        row["communicative_intent"] = previous.get("communicative_intent", "")
        row["split_status"] = previous.get("split_status", "")

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto")

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    pending = [row for row in rows if row["split_status"] != "ok" and row[args.anchor_column].strip()]
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        try:
            responses = generate(model, tokenizer, [row[args.anchor_column] for row in batch], args.max_new_tokens)
        except Exception:  # retry one by one so a single failure does not drop the batch
            responses = []
            for row in batch:
                try:
                    responses.append(generate(model, tokenizer, [row[args.anchor_column]], args.max_new_tokens)[0])
                except Exception as error:
                    responses.append(f"ERROR {type(error).__name__}: {error}")
        for row, response in zip(batch, responses):
            try:
                row["physical_form"], row["communicative_intent"] = parse_split(response)
                row["split_status"] = "ok"
            except ValueError as error:
                row["split_status"] = f"error: {error}"[:200]
        if (start // args.batch_size + 1) % max(1, args.save_every // args.batch_size) == 0:
            save()
            print(f"{min(start + args.batch_size, len(pending))}/{len(pending)} anchors split")
    save()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
