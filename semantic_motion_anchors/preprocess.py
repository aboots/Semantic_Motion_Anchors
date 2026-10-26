"""Build the clip manifest and cache frozen text-encoder embeddings.

    python -m semantic_motion_anchors.preprocess --config configs/semantic_anchors.yaml

Both are rebuilt automatically when their inputs change: the manifest is tied to the split
settings and the content of the input files, and every embedding file to the encoder, prompt
and exact texts it was computed from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, manifest_path, save_json, text_cache_dir
from .data import MANIFEST_FIELDS, TEXT_SOURCES, build_manifest_rows, read_manifest, source_texts, write_csv_rows


def file_digest(path: str | Path | None) -> str:
    if not path or not Path(path).exists():
        return ""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_fingerprint(cfg: dict[str, Any]) -> str:
    paths = cfg["paths"]
    payload = {
        "split": cfg["data"]["split"],
        "inputs": {key: file_digest(paths.get(key)) for key in ("beat2_metadata", "beat2_anchors", "ted_eval")},
        "clip_dir": str(Path(paths["beat2_metadata"]).resolve().parent),
        "ted_motion": str(Path(paths["ted_motion"]).resolve()) if paths.get("ted_motion") and Path(paths["ted_motion"]).exists() else "",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def text_fingerprint(model_name: str, prompt: str | None, uids: list[str], texts: list[str]) -> str:
    digest = hashlib.sha256(f"{model_name}\n{prompt or ''}\n".encode("utf-8"))
    for uid, text in zip(uids, texts):
        digest.update(f"{uid}\t{text}\n".encode("utf-8"))
    return digest.hexdigest()


def cache_is_current(path: Path, fingerprint: str) -> bool:
    if not path.exists():
        return False
    return str(np.load(path, allow_pickle=False).get("fingerprint", "")) == fingerprint


def encode_texts(model, texts: list[str], batch_size: int, prompt_name: str | None) -> np.ndarray:
    kwargs = {"prompt_name": prompt_name} if prompt_name else {}
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        **kwargs,
    )
    return embeddings.astype(np.float32)


def load_text_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, model_kwargs={"attn_implementation": "eager"})
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Config overrides.")
    parser.add_argument("--force", action="store_true", help="Rebuild the manifest and all embeddings.")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)

    manifest = manifest_path(cfg)
    info_path = manifest.with_suffix(".json")
    fingerprint = manifest_fingerprint(cfg)
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    if args.force or not manifest.exists() or info.get("fingerprint") != fingerprint:
        rows = build_manifest_rows(cfg)
        write_csv_rows(manifest, rows, MANIFEST_FIELDS)
        save_json(info_path, {"fingerprint": fingerprint, "splits": dict(Counter(row["split"] for row in rows))})
        print(f"Wrote {manifest} ({len(rows)} rows)")
    rows = read_manifest(manifest)
    print(f"Splits: {dict(Counter(row['split'] for row in rows))}")

    encoder_cfg = cfg["text_encoder"]
    model = None
    for source in TEXT_SOURCES:
        texts = source_texts(cfg, rows, source)
        prompt = encoder_cfg.get("transcript_prompt_name" if source == "transcript" else "anchor_prompt_name")
        uids, values = list(texts), list(texts.values())
        fingerprint = text_fingerprint(encoder_cfg["model_name"], prompt, uids, values)
        cache_path = text_cache_dir(cfg) / f"{source}.npz"
        if not args.force and cache_is_current(cache_path, fingerprint):
            print(f"Using {cache_path}")
            continue
        if model is None:
            model = load_text_encoder(encoder_cfg["model_name"])
        embeddings = encode_texts(model, values, int(encoder_cfg["batch_size"]), prompt)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, uid=np.asarray(uids), embedding=embeddings, fingerprint=np.asarray(fingerprint))
        print(f"Wrote {cache_path} ({len(uids)} texts)")


if __name__ == "__main__":
    main()
